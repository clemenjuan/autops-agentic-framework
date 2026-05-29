"""Animated satellite ground track visualization.

Generates a video or GIF of the satellite's operational ground track for a
single episode, color-coded by mode, with ground station visibility overlay.

Dependencies:
  - matplotlib (required, already in project)
  - cartopy (optional; falls back to plain equirectangular if unavailable)
  - ffmpeg binary (required for MP4 output; install via conda/brew/apt)
  - pillow (required for GIF output; `uv add pillow`)

Usage::

    from src.orchestration.animation import animate_ground_track
    from src.orchestration.results_loader import load_results, results_to_step_df

    results_raw = load_results("data/results/eventsat_cen_sda_symb_hd_ah/results.json")
    step_df = results_to_step_df(results_raw)

    anim = animate_ground_track(
        step_df, results_raw,
        episode_id=0,
        duration_steps=1440,   # 1 day
        step_stride=2,
        fps=30,
        save_path="data/figures/ground_track_ep0.mp4",
    )
"""

from __future__ import annotations

import logging
import math
import random as _random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

from src.orchestration.plotting import MODE_COLORS, MODE_ORDER, apply_style, PALETTE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cartopy availability check
# ---------------------------------------------------------------------------
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _CARTOPY_AVAILABLE = True
except ImportError:
    _CARTOPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_R_EARTH_KM = 6378.137
_MU_KM3_S2 = 398600.4418
_OMEGA_EARTH_RAD_S = 7.2921159e-5  # Earth rotation rate

# Default simulation epoch (matches context.py)
_DEFAULT_EPOCH = datetime(2026, 6, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Orbit reconstruction
# ---------------------------------------------------------------------------

def _orbit_from_results(
    results_raw: Dict[str, Any],
    episode_id: int,
) -> Optional[Dict[str, Any]]:
    """Return the orbital elements persisted for ``episode_id``, if present.

    Reads ``results["episodes"][i]["orbital_elements"]`` (written by the
    experiment runner). Returns ``None`` for older results that predate
    orbital-element persistence, so the caller can fall back to RNG replay.
    """
    for ep in results_raw.get("episodes", []) or []:
        if ep.get("episode_id") == episode_id:
            orbit = ep.get("orbital_elements")
            return dict(orbit) if orbit else None
    return None


def _reconstruct_orbital_elements(
    scenario_config: Dict[str, Any],
    experiment_seed: int,
    episode_id: int,
) -> Dict[str, Any]:
    """Reproduce the exact RAAN/ArgP/TA drawn by the launch lottery.

    Replicates the exact RNG sequence from ``eventsat_env.py::reset()``:
    - ``random.seed(seed)`` where ``seed = experiment_seed + episode_id``
    - Three consecutive ``random.uniform(0, 360)`` draws: RAAN, ArgP, TA

    Uses the **global** ``random`` module (not a ``Random()`` instance) to
    match the environment exactly.
    """
    seed = experiment_seed + episode_id
    _random.seed(seed)

    orbit = dict(scenario_config.get("orbit", {}))
    if orbit.get("launch_lottery", False):
        orbit["raan_deg"] = _random.uniform(0, 360)
        orbit["arg_perigee_deg"] = _random.uniform(0, 360)
        orbit["true_anomaly_deg"] = _random.uniform(0, 360)
        logger.debug(
            "Reconstructed launch lottery — RAAN=%.1f°, ArgP=%.1f°, TA=%.1f°",
            orbit["raan_deg"], orbit["arg_perigee_deg"], orbit["true_anomaly_deg"],
        )
    return orbit


# ---------------------------------------------------------------------------
# Propagation — Orekit path
# ---------------------------------------------------------------------------

def _propagate_orekit(
    orbit: Dict[str, Any],
    n_steps: int,
    step_s: float,
    epoch: datetime,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ground track using the Orekit J2 propagator.

    Returns (lats_deg, lons_deg) arrays of length n_steps.
    """
    from src.environment.orbital.propagator import (
        create_j2_propagator,
        _get_earth,
        _datetime_to_absolute,
    )

    a_km = _R_EARTH_KM + orbit.get("altitude_km", 400.0)
    prop = create_j2_propagator(
        a_km=a_km,
        e=orbit.get("eccentricity", 0.001),
        i_deg=orbit.get("inclination_deg", 97.4),
        raan_deg=orbit.get("raan_deg", 0.0),
        argp_deg=orbit.get("arg_perigee_deg", 0.0),
        ta_deg=orbit.get("true_anomaly_deg", 0.0),
        epoch=epoch,
    )
    earth = _get_earth()
    start_date = _datetime_to_absolute(epoch)

    lats: list[float] = []
    lons: list[float] = []
    for k in range(n_steps):
        state = prop.propagate(start_date.shiftedBy(float(k * step_s)))
        gp = earth.transform(
            state.getPVCoordinates().getPosition(),
            state.getFrame(),
            state.getDate(),
        )
        lats.append(math.degrees(gp.getLatitude()))
        lons.append(math.degrees(gp.getLongitude()))

    return np.array(lats), np.array(lons)


# ---------------------------------------------------------------------------
# Propagation — Keplerian fallback (no Orekit)
# ---------------------------------------------------------------------------

def _propagate_keplerian(
    orbit: Dict[str, Any],
    n_steps: int,
    step_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Analytical two-body ground track (no J2, no Orekit required).

    Accurate enough for a presentation animation.  RAAN precession (~1°/day
    for SSO) is not modelled — use Orekit path for physical accuracy.

    Returns (lats_deg, lons_deg) arrays of length n_steps.
    """
    a = _R_EARTH_KM + orbit.get("altitude_km", 400.0)  # km
    inc = math.radians(orbit.get("inclination_deg", 97.4))
    raan0 = math.radians(orbit.get("raan_deg", 0.0))
    argp0 = math.radians(orbit.get("arg_perigee_deg", 0.0))
    ta0 = math.radians(orbit.get("true_anomaly_deg", 0.0))
    n = math.sqrt(_MU_KM3_S2 / a**3)  # mean motion rad/s

    lats = np.empty(n_steps)
    lons = np.empty(n_steps)
    for k in range(n_steps):
        t = k * step_s
        # Argument of latitude (circular orbit: M ≈ ν for e≈0)
        u = argp0 + ta0 + n * t
        lat = math.asin(math.sin(inc) * math.sin(u))
        lon_inertial = raan0 + math.atan2(math.cos(inc) * math.sin(u), math.cos(u))
        # Earth rotation: convert inertial → Earth-fixed longitude
        lon = lon_inertial - _OMEGA_EARTH_RAD_S * t
        # Normalise to [-π, π]
        lon = (lon + math.pi) % (2 * math.pi) - math.pi
        lats[k] = math.degrees(lat)
        lons[k] = math.degrees(lon)

    return lats, lons


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _propagate_ground_track(
    orbit: Dict[str, Any],
    n_steps: int,
    step_s: float,
    epoch: datetime,
) -> Tuple[np.ndarray, np.ndarray]:
    """Try Orekit J2 propagation; fall back to analytical Keplerian."""
    from src.environment.orbital.propagator import is_available as _orekit_ok
    if _orekit_ok():
        try:
            logger.info("Propagating ground track with Orekit J2 propagator.")
            return _propagate_orekit(orbit, n_steps, step_s, epoch)
        except Exception as exc:
            logger.warning("Orekit propagation failed, using Keplerian fallback: %s", exc)
    logger.info("Propagating ground track with analytical Keplerian (no J2).")
    return _propagate_keplerian(orbit, n_steps, step_s)


# ---------------------------------------------------------------------------
# Ground station footprint geometry
# ---------------------------------------------------------------------------

def _ground_station_footprint_deg(altitude_km: float, min_elevation_deg: float) -> float:
    """Return half-angle (degrees) of ground visibility footprint.

    Uses spherical Earth approximation.  For 400 km altitude and 10° min
    elevation, the footprint radius is ~14°.
    """
    el = math.radians(min_elevation_deg)
    rho = math.acos(math.cos(el) * _R_EARTH_KM / (_R_EARTH_KM + altitude_km))
    return math.degrees(rho - el)


# ---------------------------------------------------------------------------
# Map axes setup
# ---------------------------------------------------------------------------

def _setup_map_axes(fig: plt.Figure) -> Any:
    """Create and return a map axes — cartopy if available, plain otherwise."""
    if _CARTOPY_AVAILABLE:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_global()
        ax.add_feature(cfeature.NaturalEarthFeature(
            "physical", "land", "110m",
            facecolor="#e8e8e8", edgecolor="#aaaaaa", linewidth=0.4,
        ))
        ax.add_feature(cfeature.NaturalEarthFeature(
            "physical", "ocean", "110m",
            facecolor="#cde8f0", edgecolor="none",
        ))
        ax.add_feature(cfeature.NaturalEarthFeature(
            "cultural", "admin_0_boundary_lines_land", "110m",
            facecolor="none", edgecolor="#cccccc", linewidth=0.3,
        ))
        ax.gridlines(
            draw_labels=True, linewidth=0.3, color="#999999",
            alpha=0.5, linestyle="--",
            xlabel_style={"size": 6}, ylabel_style={"size": 6},
        )
        return ax
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (°)", fontsize=7)
        ax.set_ylabel("Latitude (°)", fontsize=7)
        ax.set_facecolor("#cde8f0")
        ax.axhspan(-90, 90, facecolor="#e8e8e8", alpha=0.0)
        for lon in range(-180, 181, 30):
            ax.axvline(lon, color="#cccccc", linewidth=0.3, linestyle="--")
        for lat in range(-90, 91, 30):
            ax.axhline(lat, color="#cccccc", linewidth=0.3, linestyle="--")
        ax.tick_params(labelsize=6)
        return ax


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def animate_ground_track(
    step_df: pd.DataFrame,
    results_raw: Dict[str, Any],
    *,
    episode_id: int = 0,
    duration_steps: Optional[int] = 1440,
    step_stride: int = 2,
    fps: int = 30,
    trail_steps: int = 93,
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 150,
    save_path: Optional[Union[str, Path]] = None,
) -> FuncAnimation:
    """Animate the satellite ground track for a single episode.

    Args:
        step_df: Step-level DataFrame (from ``results_to_step_df``). Must
            contain ``episode_id``, ``step``, ``resolved_mode``,
            ``battery_soc``, ``anomaly``, ``ground_pass_active``.
        results_raw: Raw ``results.json`` dict (provides config, seed,
            and scenario path for orbit reconstruction).
        episode_id: Which episode to animate (0-indexed).
        duration_steps: Number of simulation steps to animate (default 1440
            = 1 day at 60 s/step). None = full episode.
        step_stride: Render every Nth step (2 → half the frames, 2× faster
            to generate and smaller file).
        fps: Frames per second of output video.
        trail_steps: Number of past steps shown as a fading trail (default
            93 ≈ one orbital period at 60 s/step).
        figsize: Figure size in inches.
        dpi: Render resolution (150 is fine for video; use 300 for stills).
        save_path: Output path. ``.mp4`` requires ffmpeg; ``.gif`` requires
            pillow (``uv add pillow``). If None, returns animation without
            saving.

    Returns:
        ``matplotlib.animation.FuncAnimation`` object.
    """
    apply_style()

    # --- Load config from results ---
    config = results_raw.get("config", {})
    experiment_seed: int = config.get("seed", 42)
    step_s: float = float(config.get("environment", {}).get("timestep_seconds", 60))

    # Load scenario YAML for orbit + ground station params
    import yaml
    scenario_config: Dict[str, Any] = {}
    scenario_file = (
        config.get("environment", {})
        .get("scenario_config", {})
        .get("scenario_file", "")
    )
    if scenario_file:
        p = Path(scenario_file)
        if not p.is_absolute():
            # Try relative to repo root
            p = Path(__file__).parent.parent.parent / p
        if p.exists():
            with open(p) as f:
                scenario_config = yaml.safe_load(f) or {}

    # Prefer the orbital elements persisted with the run (faithful to the exact
    # simulated orbit). Fall back to replaying the RNG draw order only for older
    # results that predate orbital-element persistence.
    orbit = _orbit_from_results(results_raw, episode_id)
    if orbit is None:
        orbit = _reconstruct_orbital_elements(scenario_config, experiment_seed, episode_id)
    elif scenario_config:
        # Backfill fixed fields (altitude/inclination/epoch) the persisted dict
        # may omit, without overwriting the persisted lottery draws.
        merged = dict(scenario_config.get("orbit", {}))
        merged.update(orbit)
        orbit = merged
    gs = scenario_config.get("communications", {}).get("ground_station", {})
    gs_lat: float = gs.get("latitude_deg", 48.0483)
    gs_lon: float = gs.get("longitude_deg", 11.6567)
    gs_min_el: float = gs.get("min_elevation_deg", 10.0)
    altitude_km: float = orbit.get("altitude_km", 400.0)
    footprint_deg: float = _ground_station_footprint_deg(altitude_km, gs_min_el)

    # --- Filter episode data ---
    edf = step_df[step_df["episode_id"] == episode_id].sort_values("step").reset_index(drop=True)
    if duration_steps is not None:
        edf = edf.iloc[:duration_steps]
    n_steps = len(edf)
    if n_steps == 0:
        raise ValueError(f"No steps found for episode_id={episode_id}")

    # --- Propagate orbit ---
    lats, lons = _propagate_ground_track(orbit, n_steps, step_s, _DEFAULT_EPOCH)

    # --- Extract step arrays ---
    modes = edf["resolved_mode"].fillna("charging").values
    battery = edf["battery_soc"].fillna(0.8).values
    anomaly = edf["anomaly"].fillna("").values
    pass_active = edf["ground_pass_active"].fillna(0).astype(bool).values

    # --- Frame indices ---
    frame_indices = list(range(0, n_steps, step_stride))
    n_frames = len(frame_indices)

    # --- Figure setup ---
    fig = plt.figure(figsize=figsize, layout="constrained")
    ax = _setup_map_axes(fig)

    # Text overlays
    txt_top = ax.text(
        0.01, 0.97, "", transform=ax.transAxes,
        fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="none"),
        zorder=10,
    )
    txt_bot = ax.text(
        0.01, 0.03, "", transform=ax.transAxes,
        fontsize=7, va="bottom", ha="left", color="#444444",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65, edgecolor="none"),
        zorder=10,
    )

    # Ground station marker
    gs_kwargs = dict(
        marker="^", s=80, color="#d62728", zorder=8,
        label="Ottobrunn GS",
    )
    if _CARTOPY_AVAILABLE:
        ax.scatter([gs_lon], [gs_lat], transform=ccrs.PlateCarree(), **gs_kwargs)
    else:
        ax.scatter([gs_lon], [gs_lat], **gs_kwargs)

    # Visibility footprint — filled circle (light) + dashed edge
    _patch_kwargs = {"transform": ccrs.PlateCarree()} if _CARTOPY_AVAILABLE else {}
    footprint_fill = plt.Circle(
        (gs_lon, gs_lat), footprint_deg,
        facecolor="#ffaaaa", edgecolor="none",
        alpha=0.0, zorder=6,
        **_patch_kwargs,
    )
    footprint_circle = plt.Circle(
        (gs_lon, gs_lat), footprint_deg,
        facecolor="none", edgecolor="#d62728", linewidth=1.2,
        linestyle="--", alpha=0.0, zorder=7,
        **_patch_kwargs,
    )
    ax.add_patch(footprint_fill)
    ax.add_patch(footprint_circle)

    # Trail scatter (updated each frame — uses scatter so antimeridian wrapping is fine)
    trail_scatter = ax.scatter(
        [], [], s=20, zorder=5,
        **({"transform": ccrs.PlateCarree()} if _CARTOPY_AVAILABLE else {}),
    )

    # Satellite marker
    sat_scatter = ax.scatter(
        [], [], s=120, zorder=9, edgecolors="white", linewidths=0.8,
        **({"transform": ccrs.PlateCarree()} if _CARTOPY_AVAILABLE else {}),
    )

    # Mode legend
    legend_handles = [
        mpatches.Patch(
            facecolor=MODE_COLORS[m], alpha=0.85,
            label=m.replace("payload_", "").replace("_", " ").title(),
        )
        for m in MODE_ORDER if m in MODE_COLORS
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=5.5,
        ncol=len(MODE_ORDER),
        frameon=True, fancybox=False,
        edgecolor="#cccccc", facecolor="white", framealpha=0.85,
        handlelength=0.9, handletextpad=0.3, columnspacing=0.5,
    )

    # --- Animation functions ---
    def init():
        trail_scatter.set_offsets(np.empty((0, 2)))
        sat_scatter.set_offsets(np.empty((0, 2)))
        sat_scatter.set_facecolors([])
        footprint_fill.set_alpha(0.0)
        footprint_circle.set_alpha(0.0)
        txt_top.set_text("")
        txt_bot.set_text("")
        return trail_scatter, sat_scatter, footprint_fill, footprint_circle, txt_top, txt_bot

    def update(frame_num: int):
        i = frame_indices[frame_num]

        # Trail
        t_start = max(0, i - trail_steps)
        t_lons = lons[t_start:i + 1]
        t_lats = lats[t_start:i + 1]
        t_modes = modes[t_start:i + 1]
        t_len = len(t_lons)
        alphas = np.linspace(0.08, 1.0, t_len)
        trail_colors = np.array([
            [*_hex_to_rgb(MODE_COLORS.get(m, "#888888")), a]
            for m, a in zip(t_modes, alphas)
        ])
        trail_scatter.set_offsets(np.column_stack([t_lons, t_lats]))
        trail_scatter.set_facecolors(trail_colors)
        trail_scatter.set_edgecolors("none")

        # Satellite
        sat_scatter.set_offsets([[lons[i], lats[i]]])
        sat_scatter.set_facecolors([MODE_COLORS.get(modes[i], "#888888")])

        # Ground station visibility circle
        footprint_fill.set_alpha(0.25 if pass_active[i] else 0.0)
        footprint_circle.set_alpha(0.85 if pass_active[i] else 0.0)

        # Text overlays
        mode_label = modes[i].replace("payload_", "").replace("_", " ").upper()
        batt_pct = int(round(battery[i] * 100))
        mission_day = (i * step_s) / 86400
        txt_top.set_text(
            f"Mode: {mode_label}    Battery: {batt_pct}%    Step {i + 1}/{n_steps}"
        )
        anom_str = str(anomaly[i]) if anomaly[i] else "None"
        pass_str = "PASS ACTIVE" if pass_active[i] else ""
        txt_bot.set_text(
            f"Anomaly: {anom_str}    Day {mission_day:.2f}    {pass_str}"
        )

        return trail_scatter, sat_scatter, footprint_fill, footprint_circle, txt_top, txt_bot

    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        init_func=init,
        interval=1000 / fps,
        blit=False,  # blit=True can cause issues with cartopy transforms
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_path.suffix.lower()
        from matplotlib.animation import FFMpegWriter
        if suffix != ".gif" and not FFMpegWriter.isAvailable():
            logger.warning("ffmpeg not available. Saving as GIF instead of %s", save_path.name)
            save_path = save_path.with_suffix(".gif")
            suffix = ".gif"
        if suffix == ".gif":
            anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps, dpi=dpi,
                      extra_args=["-vcodec", "libx264", "-crf", "22"])
        logger.info("Saved animation to %s", save_path)

    return anim


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert '#rrggbb' to (r, g, b) floats in [0, 1]."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

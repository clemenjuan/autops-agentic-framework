
import rerun as rr
import rerun.blueprint as rrb
import argparse
import os          # DEBUG ONLY: REMOVE LATER -- only used by _viewer_executable
import shlex
import sys
import numpy as np

from typing import Optional, Sequence, Dict, Callable, List
from pathlib import Path
from src.environment.orbital.adcs.constants import R_EARTH, MU_EARTH
from src.environment.orbital.adcs.eventsat import satellite


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.episode_io import Recording

M_TO_KM = 1e-3

# The orbit view's marker is not to scale. In order to make it visible the
# marker is sized as a fraction of Earth's radius and only its proportions are real
MARKER_FRACTION = 0.1

# --- DEBUG ONLY: REMOVE BEFORE COMMITTING ---------------------------------
# Put here whatever you would otherwise type after `to_rerun` on the command
# line, as one string. Non-empty means it wins over sys.argv, so hitting "Run
# and Debug" with no launch.json arguments still parses a real argument list.
# Relative paths here resolve against the debugger's working directory, which
# VS Code sets to the workspace root unless launch.json says otherwise.
# Set back to "" (or delete this block and the `or None` in __main__) to go
# back to the normal CLI.
DEBUG_ARGV = "data/records/adcs_ep_checkpoint_seed42.npz"
# --------------------------------------------------------------------------


# --- DEBUG ONLY: REMOVE BEFORE COMMITTING ---------------------------------
# Whole function goes when the debug block does.
def _viewer_executable() -> Path:
    """The viewer binary rerun-sdk ships, located without consulting PATH.

    `spawn=True` finds the viewer by searching PATH for `rerun` and has no
    fallback (see rerun/_spawn.py). `uv run` puts .venv/Scripts on PATH so the
    CLI works; the debugger launches .venv/Scripts/python.exe directly without
    activating the venv, so PATH never gets it and spawn raises
    "Failed to find Rerun Viewer executable in PATH". The binary is installed
    either way -- only the lookup fails -- so point at it directly.
    """
    exe = "rerun.exe" if os.name == "nt" else "rerun"
    return Path(rr.__file__).parent.parent / "rerun_cli" / exe
# --------------------------------------------------------------------------

############################################################################
# Helpers
############################################################################
def _timelines(rec:Recording) -> List[rr.TimeColumn]:
    indexes=[
        rr.TimeColumn("step", sequence=rec.step),
        rr.TimeColumn("t", duration=rec.t_rel)
    ]
    return indexes 


############################################################################
# Blueprint
############################################################################
# Display names for the plot groups. Keys are `PLOT_GROUPS` entries, which are
# also the `--plots` choices and the tail of each view's origin.
_VIEW_NAMES = {
    "pointing":     "Pointing",
    "rates":        "Body rates",
    "wheels":       "Wheel speeds",
    "wheel_torque": "Wheel torque",
    "mtq":          "Magnetorquers",
    "reward":       "Reward",
    "policy":       "Policy I/O",
    "misc":         "Other",
}

# What `blueprint_default` shows when `--plots` says nothing: did it point, did
# the wheels run out, what was it paid. Everything else is logged and one click
# away in the viewer's entity tree.
DEFAULT_PLOTS = ("pointing", "wheels", "reward/dense")


def _plot_views(groups:Sequence[str]) -> List[rrb.TimeSeriesView]:
    """One time-series view per plot group.

    `origin` is the whole selection mechanism: a view rooted at
    `plots/wheels` shows every series logged below it, which is why
    `log_scalars` puts the group in the entity path. For a finer cut, pass
    `contents` instead -- it takes globs, with a leading '-' to exclude, e.g.
    contents=["plots/pointing/**", "- plots/pointing/target_idx"].
    """
    return [
        rrb.TimeSeriesView(origin=f"plots/{g}", name=_VIEW_NAMES.get(g, g))
        for g in groups
    ]


def blueprint_default(plots:Optional[Sequence[str]] = None) -> rrb.Blueprint:
    """Three areas: orbit and attitude on the left, the chosen plots on the right."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(origin="world", name="Orbit"),
                rrb.Spatial3DView(origin="body", name="Attitude"),
            ),
            rrb.Vertical(*_plot_views(plots or DEFAULT_PLOTS)),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def blueprint_attitude(plots:Optional[Sequence[str]] = None) -> rrb.Blueprint:
    """No orbit view: the closeup, with the plots that explain what it is doing.

    Reward is split here rather than shown whole, because the event bonuses
    (500, 100) flatten the dense terms (1.0, 0.1) onto the same axis.
    """
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="body", name="Attitude"),
            rrb.Vertical(*_plot_views(plots or ("pointing", "rates", "wheel_torque", "mtq"))),
            column_shares=[2, 4],
        ),
        collapse_panels=True,
    )


def blueprint_reward(plots:Optional[Sequence[str]] = None) -> rrb.Blueprint:
    """For reading the policy rather than the spacecraft."""
    if plots:
        return rrb.Blueprint(rrb.Vertical(*_plot_views(plots)), collapse_panels=True)
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.TimeSeriesView(origin="plots/reward/dense", name="Reward (dense)"),
            rrb.TimeSeriesView(origin="plots/reward/event", name="Reward (events)"),
            rrb.TimeSeriesView(origin="plots/reward/cumulative", name="Return"),
            rrb.TimeSeriesView(origin="plots/policy", name="Policy I/O"),
        ),
        collapse_panels=True,
    )


BLUEPRINTS: Dict[str, Callable[..., rrb.Blueprint]] ={
    "default":  blueprint_default,
    "attitude": blueprint_attitude,
    "reward":   blueprint_reward,
}

############################################################################
# Orbit
############################################################################
def log_earth() -> None:
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    half_size= [R_EARTH * M_TO_KM] * 3 
    rr.log(
        "world/earth",
        rr.Ellipsoids3D(half_sizes=half_size),
        static=True
    )

def _orbit_ring(r0: np.ndarray, v0:np.ndarray, n: int = 360) ->np.ndarray:
    """The array needed to build the orbital ring for any object

    Args:
        r0: Position at t0, ECI [m], shape (3,).
        v0: Velocity at t0, ECI [m/s], shape (3,).
        n: Points around the ring.

    Returns:
        Points around the closed orbit, ECI [m], shape (n, 3).

    """

    # Orbit-plane basis. The normal comes first because the other two are only
    # defined relative to it: x towards the satellite at t0, y completing the
    # right-handed set and so pointing along the motion.
    h = np.cross(r0, v0)
    n_hat = h / np.linalg.norm(h)
    x_hat = r0 / np.linalg.norm(r0)
    y_hat = np.cross(n_hat, x_hat)
    
    # vis-viva. Using |r0| instead would be within metres here, but only
    # because the orbit is circular -- this stays right if it ever is not.
    a = 1.0 / (2.0 / np.linalg.norm(r0) - v0 @ v0 / MU_EARTH)
    
    # endpoint=True (the default) makes theta[-1] coincide with theta[0],
    # which is what closes the strip instead of leaving a seam.
    theta = np.linspace(0, 2 * np.pi, n)
    ring = a * (np.cos(theta)[:, None] * x_hat + np.sin(theta)[:, None] * y_hat)

    return ring

def log_orbit_path(rec:Recording) -> None:
    """The whole orbit the satellite is on, as one closed ring.

    Derived rather than recorded. An ADCS episode is seconds long and sweeps
    well under a degree, so `r_eci` on its own draws a short tick beside the
    Earth and reads as nothing. The state vector at t0 pins the orbit exactly,
    and at the eccentricity these episodes fly (~1e-5) a circle in the orbit
    plane is indistinguishable from the ellipse.

    Taken from the recording, not from `eventsat.orbit`: the launch lottery
    randomises RAAN/ArgP/TA per episode, so the nominal config would draw a
    different ring than the one actually flown.
    """
    r0 = rec["r_eci"][0]
    v0 = rec["v_eci"][0]

    rr.log(
        "world/orbit",
        rr.LineStrips3D(_orbit_ring(r0,v0) * M_TO_KM),
        static=True
    )

def log_target_orbits(rec:Recording) -> None:
    """The orbits the targets are on as closed rings"""

    target_positions = rec["target_positions"] #(T, num_targets, 3)
    target_velocities = rec["target_velocities"] #(T, num_targets, 3)
    num_targets = target_positions.shape[1]

    for i in range(num_targets):
        rr.log(
            f"world/targets/{i}/orbit",
            rr.LineStrips3D(
                _orbit_ring(r0=target_positions[0][i], v0=target_velocities[0][i]) * M_TO_KM,
                colors=TARGET_PALETTE[0]),
            static = True
            )

def log_orbit_track(rec:Recording) -> None:
    """Where the satellite actually was, over the episode.

    A separate entity from `world/orbit` because it answers a different
    question: a ~165 km arc, invisible at orbit scale on its own, but the
    thing worth looking at once the satellite marker moves along it.
    """
    rr.log(
        "world/track",
        rr.LineStrips3D(rec["r_eci"] * M_TO_KM),
        static=True
    )

def log_orbit_motion(rec:Recording) -> None:
    rr.send_columns(
        "world/satellite",
        indexes= _timelines(rec),
        columns=rr.Transform3D.columns(translation=rec["r_eci"] * M_TO_KM),
    )

def log_satellite_marker() -> None:
    """A stand-in box for the satellite, as a child of the orbit transform.

    Child of `world/satellite`, so it inherits that transform and needs to know
    nothing about time -- the same reason the visibility cone and the pointing
    arrows can hang here later and move for free.

    Deliberately not to scale (see MARKER_FRACTION), but the proportions are
    EventSat's own, so the box still reads as an orientation once a rotation
    joins the translation on the parent transform.
    """
    dims = satellite.dimensions
    marker = dims / dims.max() * MARKER_FRACTION * R_EARTH * M_TO_KM
    rr.log(
        "world/satellite/marker",
        rr.Boxes3D(sizes=marker, colors=[0xFF, 0xA5, 0x00]),
        static=True
    )

TARGET_PALETTE = np.array([[120,120,120], [40,160,60], [255,80,40]], dtype=np.uint8)

def log_targets(rec:Recording)->None:
    pos = rec["target_positions"] * M_TO_KM #(T, num_targets, 3)
    num_targets = pos.shape[1]  
    colors = TARGET_PALETTE[rec["target_status"]] # (T, num_targets, 3)

    rr.send_columns(
        "world/targets/markers",
        indexes=_timelines(rec),
        columns=rr.Points3D.columns(
            positions=pos.reshape(-1,3),
            colors=colors.reshape(-1,3),
    ).partition([num_targets] * pos.shape[0])
    )


############################################################################
# Attitude
############################################################################

def log_satellite_geometry() -> None:
    """The satellite at true size, for the attitude closeup.

    Is independend of the marker used in the orbital figure and thus not inherits 
    it's motion

    Sized from `satellite.dimensions` -- note the config flags the axis order as
    unconfirmed against CMO p.16, so the box is the right size but possibly the
    wrong way round.
    """
    rr.log("body", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "body/satellite/box",
        rr.Boxes3D(sizes=satellite.dimensions, colors=[0xFF, 0xA5, 0x00]),
        static=True
    )

def log_attitude(rec:Recording) -> None:
    """Turn the closeup box with the recorded attitude.

    Rotation only, no translation: that is what keeps the box at the origin
    while it turns, and what makes this view about attitude alone.

    On the quaternion, which looks wrong and is not
    -----------------------------------------------
    Two things differ between the archive and rerun, and only one needs fixing.

    Component order does: the channel is scalar-first (w, x, y, z), rerun is
    scalar-last (x, y, z, w). Hence the roll.

    Direction looks like it does, and does not. `Transform3D` asks "the geometry
    below this node is written in the node's own frame -- where does it sit in
    the parent's?", i.e. body -> ECI, while the channel is named ECI -> body.
    But that name belongs to `dcm_eci_to_body(q)`, not to `q`: under this
    codebase's Hamilton convention the textbook rotation matrix R(q) already
    *is* body -> ECI, which is why `dynamics.dcm_eci_to_body` returns R(q).T to
    get the direction it is named after (dynamics.py:53-68). Rerun applies that
    same textbook R(q). So the two transposes that would have cancelled never
    happen, and the reorder is the whole conversion.

    Checked rather than assumed: Rotation.from_quat(np.roll(q, -1)) reproduces
    dcm_eci_to_body(q).T exactly on this data. Worth re-checking if the
    kinematics convention in `dynamics` ever changes, because getting this
    backwards raises no error -- it renders a plausible, wrong attitude.

    """
    rr.send_columns(
        "body/satellite",
        indexes=_timelines(rec),
        columns=rr.Transform3D.columns(
            quaternion=np.roll(rec["q_eci_body"], -1, axis=1),
        ),
    )

############################################################################
# Scalar Plots
############################################################################

# Channels that are not plotted. The spatial ones are the 3D views' business;
# `t` is the timeline itself, so plotting it draws a diagonal and nothing else.
_NOT_PLOTTED = ("t", "r_eci", "v_eci", "q_eci_body", "target_q_eci_body")

# Channel -> group. A channel missing here lands in "misc" rather than being
# dropped, so a channel added to `adcs_record` still shows up somewhere without
# this file changing; promote it to a real group once it has earned one.
_GROUP_OF = {
    "pointing_error_deg": "pointing",
    "hold_timer":         "pointing",
    "body_rate":          "rates",
    "omega_body":         "rates",
    "wheel_speeds":       "wheels",
    "wheel_torque":       "wheel_torque",
    "mtq_dipole":         "mtq",
    "action":             "policy",
    "obs":                "policy",
}

# Group names are what `--plots` accepts and what a blueprint points a
# TimeSeriesView's `origin` at. Static, unlike group membership.
PLOT_GROUPS = ("pointing", "rates", "wheels", "wheel_torque", "mtq",
               "reward_dense", "reward_total", "policy", "misc")


def _component_names(rec:Recording, channel:str, width:int) -> List[str]:
    """Labels for one column of a (N, k) channel.

    Fanning a multi-component channel into one entity path per component is
    what makes each line separately toggleable and separately coloured; these
    are the leaf names those paths get. Unknown channels fall back to indices,
    which is ugly but never wrong.
    """
    n = rec.n_wheels
    if channel == "omega_body":
        return ["x", "y", "z"]
    if channel in ("wheel_speeds", "wheel_torque"):
        return [f"wheel_{i}" for i in range(width)]
    if channel == "mtq_dipole":
        return [f"mtq_{ax}" for ax in "xyz"[:width]]
    if channel == "action":
        # The one channel still carrying both actuators, because it is the
        # policy's own output vector: wheel torques first, dipoles after, all
        # normalised to +-1 so they share an axis honestly.
        return [f"wheel_{i}" for i in range(n)] + [f"mtq_{ax}" for ax in "xyz"[: width - n]]
    if channel == "obs":
        # concat([q_error(4), omega_body(3), wheel_speeds(4)]), see
        # adcs_gymnasium_wrapper.py:214. Normalised, so these are not the same
        # numbers as the `rates` and `wheels` groups -- that is the point of
        # keeping them: a scaling bug shows up here and nowhere else.
        return (["q_err_w", "q_err_x", "q_err_y", "q_err_z"]
                + ["omega_x", "omega_y", "omega_z"]
                + [f"wheel_{i}" for i in range(width - 7)])
    return [str(i) for i in range(width)]


def _send_scalar(path:str, series:np.ndarray, indexes:List[rr.TimeColumn]) -> None:
    """One scalar series, as a column rather than a row at a time."""
    rr.send_columns(path, indexes=indexes, columns=rr.Scalars.columns(scalars=series))


def log_threshold(path:str, value:float, name:str, rec:Recording,
                  indexes:List[rr.TimeColumn]) -> None:
    """One horizontal reference line, as a constant series.

    Rerun's time-series views have no horizontal-line primitive, so a threshold
    is an ordinary series that happens not to vary. Styled thin and grey by the
    static `SeriesLines` so it reads as a reference rather than as data -- the
    styling is static because it describes the series, not any one sample.
    """
    rr.log(path, rr.SeriesLines(colors=[140, 140, 140], widths=1.0, names=name), static=True)
    _send_scalar(path, np.full(len(rec), float(value)), indexes)


def log_thresholds(rec:Recording) -> None:
    """The limits the episode was actually flown against.

    Logged under each group's own subtree, so a view rooted at `plots/wheels`
    picks up its limit lines without naming them -- the same `origin` rule that
    selects the data.

    Limits absent from older archives come back as None from `Recording` rather
    than raising, so each block is guarded: a file recorded before `build_meta`
    grew a field still plots, just without that line.
    """
    indexes = _timelines(rec)

    log_threshold("plots/pointing/limits/tolerance", rec.tolerance_deg, "tolerance", rec, indexes)
    log_threshold("plots/pointing/limits/hold_time", rec.hold_time, "hold time", rec, indexes)
    log_threshold("plots/rates/limits/slew_penalty", rec.body_rate_thresh, "slew penalty", rec, indexes)

    for sign in (+1, -1):
        log_threshold(f"plots/wheels/limits/max_speed_{'pos' if sign > 0 else 'neg'}",
                      sign * rec.wheel_max_speed, "wheel limit", rec, indexes)
        # Normalised commands, so the rail is at +-1 by construction, not config.
        log_threshold(f"plots/policy/limits/action_{'pos' if sign > 0 else 'neg'}",
                      sign * 1.0, "action rail", rec, indexes)

    # Per-actuator, and per-axis within that: nothing requires a cluster to be
    # built from identical wheels, so these are arrays rather than scalars.
    if rec.wheel_max_torque is not None:
        for sign in (+1, -1):
            log_threshold(f"plots/wheel_torque/limits/max_{'pos' if sign > 0 else 'neg'}",
                          sign * rec.wheel_max_torque.max(), "wheel torque limit", rec, indexes)
    if rec.mtq_max_dipole is not None:
        for sign in (+1, -1):
            log_threshold(f"plots/mtq/limits/max_{'pos' if sign > 0 else 'neg'}",
                          sign * rec.mtq_max_dipole.max(), "MTQ limit", rec, indexes)


def log_scalars(rec:Recording) -> None:
    """Every numeric channel in the archive, as a time series.

    Swept from `rec.arrays`. A channel added to `adcs_record` plots itself, so 
    everything is logged. Which groups are shown is the blueprint's decision,
    so toggling a plot on is a click rather than another run.
    """
    indexes = _timelines(rec)

    # Reward first, so the generic sweep below can skip what it has claimed.
    _send_scalar("plots/reward/total", rec["reward"], indexes)
    _send_scalar("plots/reward/cumulative", rec.cum_reward, indexes)
    for term, series in rec.dense_terms.items():
        _send_scalar(f"plots/reward/dense/{term}", series, indexes)
    for term, series in rec.event_terms.items():
        _send_scalar(f"plots/reward/event/{term}", series, indexes)

    for name in sorted(rec.arrays):
        if name in _NOT_PLOTTED or name == "reward" or name.startswith("reward_"):
            continue
        data = rec[name]
        # `phase` is <U16; nothing else in the archive is non-numeric. Bools
        # (terminated, truncated) go through as 0/1, which plots fine.
        if data.dtype.kind not in "fiub":
            continue

        series = data.astype(np.float64)
        group = _GROUP_OF.get(name, "misc")
        if series.ndim == 1:
            _send_scalar(f"plots/{group}/{name}", series, indexes)
        else:
            for i, label in enumerate(_component_names(rec, name, series.shape[1])):
                _send_scalar(f"plots/{group}/{name}/{label}", series[:, i], indexes)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        prog="to_rerun",
        description="Show all recorded data in rerun viewer",
    )

    # Positional: there is no sensible default recording, while every other
    # argument has one.
    parser.add_argument(
        "record",
        type=Path,
        help="episode .npz, as written by adcs_record",
    )

    parser.add_argument(
            "--blueprint",
            type=str,
            choices=list(BLUEPRINTS),
            default="default",
            help="Describes what layout the window should open with",
        )

    # Selects which plot views the layout opens with, not what gets logged --
    # every channel is sent regardless, so turning a group on afterwards is a
    # click in the viewer rather than another run.
    parser.add_argument(
        "--plots",
        type=str,
        default=None,
        metavar=",".join(PLOT_GROUPS[:3]) + ",...",
        help=f"comma-separated plot groups to show, from: {', '.join(PLOT_GROUPS)} "
             "(default: whatever the blueprint chooses)",
    )

    args = parser.parse_args(argv)

    if args.plots is not None:
        args.plots = [g.strip() for g in args.plots.split(",") if g.strip()]
        unknown = [g for g in args.plots if g not in PLOT_GROUPS]
        if unknown:
            # parser.error, not a raise: this is a bad command line, so it earns
            # the usage message and exit code 2 that argparse gives.
            parser.error(
                f"unknown plot group(s) {', '.join(unknown)}; "
                f"choose from {', '.join(PLOT_GROUPS)}"
            )
    return args

def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    #Open the recording
    rec = Recording.load(args.record)

    
    # --- DEBUG ONLY: REMOVE BEFORE COMMITTING -----------------------------
    # Restore the line below and delete the two after it. `rr.init` has no
    # executable_path parameter, which is why this has to be two calls.
    #rr.init("rerun_viewer_test", spawn=True)
    rr.init("rerun_viewer_test")
    rr.spawn(executable_path=str(_viewer_executable()))
    # ----------------------------------------------------------------------

    # This creates blueprints which has everything prearanged 
    # and overwrites how the user left the viewer 
    rr.send_blueprint(BLUEPRINTS[args.blueprint](args.plots))
   
    # All static object created and logged 
    log_earth()
    log_orbit_path(rec=rec)
    log_target_orbits(rec=rec)
    log_orbit_track(rec=rec)
    log_satellite_marker()
    log_satellite_geometry()
    # log_series_styling()
    # log_ground_targets()


    # Temporal pass
    log_orbit_motion(rec=rec)
    log_targets(rec=rec)
    log_attitude(rec=rec)
    log_scalars(rec=rec)
    log_thresholds(rec=rec)
    # log_events()

    return 0



if __name__ == "__main__":
    # DEBUG ONLY: drop the conditional and call main() once DEBUG_ARGV goes.
    # A fallback, not an override: DEBUG_ARGV applies only when the command
    # line carried nothing, which is the debugger's case. Passing it whenever
    # it is non-empty would swallow every real flag -- argparse ignores
    # sys.argv entirely once main() is handed an argv.
    # raise(main())
    raise SystemExit(main(shlex.split(DEBUG_ARGV) if len(sys.argv) == 1 else None))

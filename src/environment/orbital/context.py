"""Orbital context — pre-computed orbital events for an episode.

Bundles eclipse intervals and ground passes into a single object
that the environment queries at each step. Events are pre-computed
at reset() time for the full episode duration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.environment.orbital.eclipse import (
    EclipseInterval,
    compute_eclipses_orekit,
    compute_eclipses_simplified,
    is_in_sunlight,
)
from src.environment.orbital.ground_access import (
    GroundPass,
    compute_passes_orekit,
    compute_passes_simplified,
)

logger = logging.getLogger(__name__)


@dataclass
class OrbitalContext:
    """Pre-computed orbital events for a simulation episode.

    Attributes:
        eclipses: List of eclipse intervals (step ranges in shadow).
        ground_passes: List of ground station passes.
        mode: Which propagation mode was used ('orekit' or 'simplified').
    """

    eclipses: List[EclipseInterval] = field(default_factory=list)
    ground_passes: List[GroundPass] = field(default_factory=list)
    mode: str = "simplified"

    def is_in_sunlight(self, step: int) -> bool:
        """Check if satellite is in sunlight at this step."""
        return is_in_sunlight(step, self.eclipses)

    def is_ground_pass_active(self, step: int) -> bool:
        """Check if a ground pass is active at this step."""
        for gp in self.ground_passes:
            if gp.start_step <= step <= gp.end_step:
                return True
        return False

    def get_current_pass(self, step: int) -> Optional[GroundPass]:
        """Return the active ground pass, or None."""
        for gp in self.ground_passes:
            if gp.start_step <= step <= gp.end_step:
                return gp
        return None


def compute_orbital_context(
    orbit_config: Dict[str, Any],
    comms_config: Dict[str, Any],
    step_s: float,
    total_steps: int,
    epoch: Optional[datetime] = None,
    require_orekit: bool = False,
) -> OrbitalContext:
    """Compute orbital context for an episode.

    Tries Orekit first if available and orbit config provides sufficient
    parameters. Falls back to simplified models otherwise.

    Args:
        orbit_config: Orbit parameters from scenario YAML.
        comms_config: Communications config (ground station, passes).
        step_s: Timestep duration in seconds.
        total_steps: Total simulation steps.
        epoch: Simulation start epoch (for Orekit).
        require_orekit: If True and Orekit is not available or fails,
            log a warning instead of silently falling back. Ground passes
            will be stochastic rather than physics-based.

    Returns:
        OrbitalContext with pre-computed events.
    """
    from src.environment.orbital.propagator import is_available

    duration_s = total_steps * step_s

    # Try Orekit propagation
    if is_available() and _has_orekit_params(orbit_config):
        try:
            return _compute_orekit_context(
                orbit_config, comms_config, step_s, total_steps, duration_s, epoch
            )
        except Exception as e:
            if require_orekit:
                logger.error(
                    "Orekit required but failed — ground passes will be stochastic: %s", e
                )
            else:
                logger.warning("Orekit computation failed, falling back: %s", e)
    elif require_orekit:
        logger.warning(
            "Orekit not available — ground passes will be stochastic, not physics-based. "
            "Install orekit-jpype for deterministic pass computation."
        )

    # Simplified fallback
    return _compute_simplified_context(
        orbit_config, comms_config, step_s, total_steps
    )


def _has_orekit_params(orbit_config: Dict[str, Any]) -> bool:
    """Check if orbit config has enough data for Orekit propagation."""
    has_tle = "tle_line1" in orbit_config and "tle_line2" in orbit_config
    has_kep = "altitude_km" in orbit_config and "inclination_deg" in orbit_config
    return has_tle or has_kep


def _compute_orekit_context(
    orbit_config: Dict[str, Any],
    comms_config: Dict[str, Any],
    step_s: float,
    total_steps: int,
    duration_s: float,
    epoch: Optional[datetime],
) -> OrbitalContext:
    """Compute context using Orekit."""
    from src.environment.orbital.propagator import (
        create_j2_propagator,
        create_keplerian_propagator,
        create_tle_propagator,
    )

    if epoch is None:
        epoch = datetime(2026, 6, 1, tzinfo=timezone.utc)

    # Create propagator — TLE takes priority; otherwise select by 'propagator' key
    if "tle_line1" in orbit_config:
        propagator = create_tle_propagator(
            orbit_config["tle_line1"], orbit_config["tle_line2"]
        )
    else:
        earth_radius_km = 6378.137
        a_km = earth_radius_km + orbit_config.get("altitude_km", 500)
        propagator_type = orbit_config.get("propagator", "j2")
        kep_kwargs = dict(
            a_km=a_km,
            e=orbit_config.get("eccentricity", 0.001),
            i_deg=orbit_config.get("inclination_deg", 97.4),
            raan_deg=orbit_config.get("raan_deg", 0.0),
            argp_deg=orbit_config.get("arg_perigee_deg", 0.0),
            ta_deg=orbit_config.get("true_anomaly_deg", 0.0),
            epoch=epoch,
        )
        if propagator_type == "j2":
            propagator = create_j2_propagator(**kep_kwargs)
        else:
            propagator = create_keplerian_propagator(**kep_kwargs)

    # Compute eclipses
    eclipses = compute_eclipses_orekit(propagator, duration_s, step_s)

    # Compute ground passes
    gs = comms_config.get("ground_station", {})
    station_lat = gs.get("latitude_deg", 48.0483)
    station_lon = gs.get("longitude_deg", 11.6567)
    min_el = gs.get("min_elevation_deg", 10.0)
    dl_rate = comms_config.get("sband", {}).get("downlink_rate_kbps", 128.0)

    ground_passes = compute_passes_orekit(
        propagator,
        station_lat_deg=station_lat,
        station_lon_deg=station_lon,
        min_elevation_deg=min_el,
        duration_s=duration_s,
        step_s=step_s,
        downlink_rate_kbps=dl_rate,
    )

    logger.info(
        "Orekit context: %d eclipses, %d ground passes over %d steps",
        len(eclipses),
        len(ground_passes),
        total_steps,
    )

    return OrbitalContext(
        eclipses=eclipses,
        ground_passes=ground_passes,
        mode="orekit",
    )


def _compute_simplified_context(
    orbit_config: Dict[str, Any],
    comms_config: Dict[str, Any],
    step_s: float,
    total_steps: int,
) -> OrbitalContext:
    """Compute context using simplified analytical models."""
    orbital_period_s = orbit_config.get("orbital_period_s", 5676)
    eclipse_fraction = orbit_config.get("eclipse_fraction", 0.36)

    eclipses = compute_eclipses_simplified(
        orbital_period_s, eclipse_fraction, step_s, total_steps
    )

    passes_cfg = comms_config.get("passes", {})
    ground_passes = compute_passes_simplified(
        step_s=step_s,
        total_steps=total_steps,
        passes_min_per_day=passes_cfg.get("min_per_day", 2),
        passes_max_per_day=passes_cfg.get("max_per_day", 3),
        pass_min_dur_s=passes_cfg.get("min_duration_s", 22.0),
        pass_max_dur_s=passes_cfg.get("max_duration_s", 422.0),
        avg_data_per_day_mb=passes_cfg.get("avg_data_per_day_mb", 12.0),
    )

    logger.info(
        "Simplified context: %d eclipses, %d ground passes over %d steps",
        len(eclipses),
        len(ground_passes),
        total_steps,
    )

    return OrbitalContext(
        eclipses=eclipses,
        ground_passes=ground_passes,
        mode="simplified",
    )

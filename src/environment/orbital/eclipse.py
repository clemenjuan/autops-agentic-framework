"""Eclipse computation.

Computes sunlight/eclipse intervals for a satellite orbit.
Uses Orekit EclipseDetector when available, falls back to a
simple phase-fraction model otherwise.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EclipseInterval:
    """A single eclipse (shadow) interval."""

    start_step: int
    end_step: int


def compute_eclipses_orekit(
    propagator: Any,
    duration_s: float,
    step_s: float,
) -> List[EclipseInterval]:
    """Compute eclipse intervals using Orekit step-by-step propagation.

    Steps through the propagation and checks whether the satellite is
    in Earth's shadow at each timestep using the Sun-Earth geometry.

    Args:
        propagator: An Orekit propagator (Keplerian or TLE).
        duration_s: Total simulation duration in seconds.
        step_s: Timestep in seconds.

    Returns:
        List of EclipseInterval with step indices.
    """
    from src.environment.orbital.propagator import get_sun, get_earth_body

    sun = get_sun()
    earth = get_earth_body()

    start_date = propagator.getInitialState().getDate()
    total_steps = int(duration_s / step_s)

    eclipses: List[EclipseInterval] = []
    in_eclipse = False
    eclipse_start = 0

    for step in range(total_steps):
        t = step * step_s
        state = propagator.propagate(start_date.shiftedBy(t))
        sat_pos = state.getPVCoordinates().getPosition()
        frame = state.getFrame()
        date = state.getDate()

        # Sun position in the same frame
        sun_pos = sun.getPVCoordinates(date, frame).getPosition()

        # Check if satellite is in Earth's shadow using geometric test:
        # Project satellite position onto Sun direction. If the satellite
        # is behind Earth relative to the Sun, it's in eclipse.
        earth_radius = earth.getEquatorialRadius()

        # Vector from Earth center to satellite
        sat_r = sat_pos.getNorm()
        # Vector from Earth center to Sun
        sun_r = sun_pos.getNorm()

        # Angle between satellite and sun as seen from Earth center
        cos_angle = sat_pos.dotProduct(sun_pos) / (sat_r * sun_r)

        # Simple cylindrical shadow model:
        # In eclipse if satellite is on the opposite side of Earth from Sun
        # and within Earth's shadow cone
        if cos_angle < 0:  # satellite is on dark side
            # Check if satellite is within the shadow cylinder
            # perpendicular distance from the Earth-Sun line
            sin_angle = math.sqrt(1.0 - cos_angle * cos_angle)
            perp_dist = sat_r * sin_angle
            if perp_dist < earth_radius:
                if not in_eclipse:
                    in_eclipse = True
                    eclipse_start = step
            else:
                if in_eclipse:
                    eclipses.append(EclipseInterval(eclipse_start, step - 1))
                    in_eclipse = False
        else:
            if in_eclipse:
                eclipses.append(EclipseInterval(eclipse_start, step - 1))
                in_eclipse = False

    if in_eclipse:
        eclipses.append(EclipseInterval(eclipse_start, total_steps - 1))

    return eclipses


def compute_eclipses_simplified(
    orbital_period_s: float,
    eclipse_fraction: float,
    step_s: float,
    total_steps: int,
) -> List[EclipseInterval]:
    """Compute eclipse intervals using a simple phase-fraction model.

    The orbit is divided into sunlight and eclipse phases based on the
    eclipse fraction. Eclipse occurs at the start of each orbit period.

    Args:
        orbital_period_s: Orbital period in seconds.
        eclipse_fraction: Fraction of orbit spent in eclipse (0-1).
        step_s: Timestep in seconds.
        total_steps: Total number of simulation steps.

    Returns:
        List of EclipseInterval with step indices.
    """
    period_steps = max(1, int(orbital_period_s / step_s))
    eclipse_steps = max(1, int(period_steps * eclipse_fraction))

    eclipses: List[EclipseInterval] = []
    step = 0
    while step < total_steps:
        orbit_start = step
        eclipse_end = min(orbit_start + eclipse_steps - 1, total_steps - 1)
        eclipses.append(EclipseInterval(orbit_start, eclipse_end))
        step += period_steps

    return eclipses


def is_in_sunlight(step: int, eclipses: List[EclipseInterval]) -> bool:
    """Check if a given step is in sunlight (not in any eclipse interval)."""
    for ec in eclipses:
        if ec.start_step <= step <= ec.end_step:
            return False
    return True

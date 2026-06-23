"""Ground station access computation.

Computes visibility windows between a satellite and ground stations.
Uses Orekit elevation computation when available, falls back to
stochastic pass generation otherwise.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GroundPass:
    """A single ground station pass.

    ``start_s`` / ``end_s`` are second-accurate AOS/LOS times (the 10° crossings,
    linearly interpolated from the elevation samples); ``start_step`` / ``end_step``
    are the integer-step span they fall in (kept for back-compat). Sub-timestep
    contact uses the seconds; a 22 s pass therefore credits ~22 s, not a full step.
    """

    start_step: int
    end_step: int
    max_elevation_deg: float = 0.0
    data_budget_mb: float = 0.0
    start_s: float = 0.0
    end_s: float = 0.0


def compute_passes_orekit(
    propagator: Any,
    station_lat_deg: float,
    station_lon_deg: float,
    min_elevation_deg: float,
    duration_s: float,
    step_s: float,
    downlink_rate_kbps: float = 128.0,
) -> List[GroundPass]:
    """Compute ground passes using Orekit elevation computation.

    Steps through the propagation and checks satellite elevation
    above the ground station at each timestep.

    Args:
        propagator: An Orekit propagator.
        station_lat_deg: Ground station latitude in degrees.
        station_lon_deg: Ground station longitude in degrees.
        min_elevation_deg: Minimum elevation for visibility (degrees).
        duration_s: Total simulation duration in seconds.
        step_s: Timestep in seconds.
        downlink_rate_kbps: Downlink data rate for budget computation.

    Returns:
        List of GroundPass with step indices.
    """
    from src.orbital.propagator import make_ground_station_frame

    station_frame = make_ground_station_frame(station_lat_deg, station_lon_deg)
    start_date = propagator.getInitialState().getDate()
    total_steps = int(duration_s / step_s)

    passes: List[GroundPass] = []
    in_pass = False
    pass_start = 0
    max_el = 0.0
    start_s = 0.0
    el_prev = None  # elevation at the previous coarse sample (for AOS/LOS interpolation)

    def _make_pass(p_start_step: int, p_end_step: int, p_start_s: float,
                   p_end_s: float, p_max_el: float) -> GroundPass:
        contact_s = max(0.0, p_end_s - p_start_s)
        data_mb = downlink_rate_kbps / 8.0 * contact_s / 1000.0
        return GroundPass(
            start_step=p_start_step, end_step=p_end_step,
            max_elevation_deg=p_max_el, data_budget_mb=data_mb,
            start_s=p_start_s, end_s=p_end_s,
        )

    for step in range(total_steps):
        t = step * step_s
        state = propagator.propagate(start_date.shiftedBy(t))
        pos = state.getPVCoordinates().getPosition()
        topo = station_frame.getTrackingCoordinates(
            pos, state.getFrame(), state.getDate()
        )
        elevation = math.degrees(topo.getElevation())

        if elevation >= min_elevation_deg:
            if not in_pass:
                in_pass = True
                pass_start = step
                max_el = elevation
                # AOS: interpolate the 10° crossing between the previous (below) sample
                # and this (above) sample. At t=0 there is no prior sample → use t.
                if el_prev is not None and elevation > el_prev:
                    frac = (min_elevation_deg - el_prev) / (elevation - el_prev)
                    start_s = (step - 1) * step_s + frac * step_s
                else:
                    start_s = t
            else:
                max_el = max(max_el, elevation)
        else:
            if in_pass:
                # LOS: interpolate the 10° crossing between the last-above sample
                # (step-1) and this (below) sample.
                if el_prev is not None and el_prev > elevation:
                    frac = (el_prev - min_elevation_deg) / (el_prev - elevation)
                    end_s = (step - 1) * step_s + frac * step_s
                else:
                    end_s = (step - 1) * step_s
                passes.append(_make_pass(pass_start, step - 1, start_s, end_s, max_el))
                in_pass = False
        el_prev = elevation

    if in_pass:
        passes.append(
            _make_pass(pass_start, total_steps - 1, start_s, (total_steps - 1) * step_s, max_el)
        )

    return passes


def compute_passes_simplified(
    step_s: float,
    total_steps: int,
    passes_min_per_day: int = 2,
    passes_max_per_day: int = 3,
    pass_min_dur_s: float = 22.0,
    pass_max_dur_s: float = 422.0,
    avg_data_per_day_mb: float = 12.0,
) -> List[GroundPass]:
    """Generate stochastic ground passes (simplified model).

    This is the fallback when Orekit is not available. Passes are
    randomly distributed throughout each day with durations drawn
    from the configured range.

    Args:
        step_s: Timestep in seconds.
        total_steps: Total number of simulation steps.
        passes_min_per_day: Minimum passes per day.
        passes_max_per_day: Maximum passes per day.
        pass_min_dur_s: Minimum pass duration in seconds.
        pass_max_dur_s: Maximum pass duration in seconds.
        avg_data_per_day_mb: Average downlink data per day.

    Returns:
        List of GroundPass with step indices.
    """
    passes: List[GroundPass] = []
    steps_per_day = int(86400 / step_s)
    total_days = max(1, total_steps // steps_per_day)

    for day in range(total_days):
        n_passes = random.randint(passes_min_per_day, passes_max_per_day)
        day_start = day * steps_per_day
        for _ in range(n_passes):
            dur_s = random.uniform(pass_min_dur_s, pass_max_dur_s)
            dur_steps = max(1, int(dur_s / step_s))
            start = day_start + random.randint(
                0, max(1, steps_per_day - dur_steps - 1)
            )
            data_mb = avg_data_per_day_mb / n_passes
            start_s = start * step_s
            passes.append(
                GroundPass(
                    start_step=start,
                    end_step=start + dur_steps,
                    max_elevation_deg=0.0,
                    data_budget_mb=data_mb,
                    start_s=start_s,
                    end_s=start_s + dur_s,  # true (sub-step-accurate) duration
                )
            )

    passes.sort(key=lambda p: p.start_step)
    return passes

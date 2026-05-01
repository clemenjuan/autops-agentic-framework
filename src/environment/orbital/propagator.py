"""Orekit propagator wrapper.

Thin abstraction over orekit-jpype for orbital propagation.
All Orekit calls are isolated here so the rest of the codebase
never imports Orekit directly.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

OREKIT_AVAILABLE = False
_orekit_initialized = False

try:
    import orekit_jpype
    import jpype

    if not jpype.isJVMStarted():
        orekit_jpype.initVM()

    from pathlib import Path as _Path
    from orekit_jpype.pyhelpers import setup_orekit_data
    # Use absolute path so it works regardless of CWD (e.g. when called from notebooks/)
    _orekit_data = str(_Path(__file__).parent.parent.parent.parent / "orekit-data.zip")
    setup_orekit_data(filenames=_orekit_data, from_pip_library=True)

    from org.orekit.frames import FramesFactory, TopocentricFrame
    from org.orekit.time import TimeScalesFactory, AbsoluteDate
    from org.orekit.bodies import (
        CelestialBodyFactory,
        GeodeticPoint,
        OneAxisEllipsoid,
    )
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.propagation.analytical import KeplerianPropagator, EcksteinHechlerPropagator
    from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
    from org.orekit.utils import Constants, IERSConventions

    OREKIT_AVAILABLE = True
    _orekit_initialized = True
    logger.info("Orekit initialized successfully.")
except Exception as e:
    logger.debug("Orekit not available: %s", e)


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
MU_EARTH = 398600.4418e9  # m^3/s^2


def is_available() -> bool:
    """Return True if Orekit is loaded and ready."""
    return OREKIT_AVAILABLE


def _get_utc():
    return TimeScalesFactory.getUTC()


def _get_earth():
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    return OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )


def _datetime_to_absolute(dt: datetime) -> Any:
    utc = _get_utc()
    return AbsoluteDate(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute,
        float(dt.second + dt.microsecond / 1e6),
        utc,
    )


def create_keplerian_propagator(
    a_km: float,
    e: float,
    i_deg: float,
    raan_deg: float,
    argp_deg: float,
    ta_deg: float,
    epoch: datetime,
) -> Any:
    """Create an analytical Keplerian propagator from orbital elements."""
    if not OREKIT_AVAILABLE:
        raise RuntimeError("Orekit is not available.")

    frame = FramesFactory.getEME2000()
    date = _datetime_to_absolute(epoch)

    orbit = KeplerianOrbit(
        a_km * 1000.0,
        e,
        math.radians(i_deg),
        math.radians(argp_deg),
        math.radians(raan_deg),
        math.radians(ta_deg),
        PositionAngleType.TRUE,
        frame,
        date,
        Constants.WGS84_EARTH_MU,
    )
    return KeplerianPropagator(orbit)


def create_j2_propagator(
    a_km: float,
    e: float,
    i_deg: float,
    raan_deg: float,
    argp_deg: float,
    ta_deg: float,
    epoch: datetime,
) -> Any:
    """Create an analytical J2 propagator using EcksteinHechler.

    Models J2 secular perturbation — critical for SSO RAAN precession
    (~0.98 deg/day at 400 km, 97.4 deg inclination). Falls back to
    Keplerian (two-body) if the J2 propagator fails.

    Uses EcksteinHechlerPropagator which handles near-circular orbits
    robustly (unlike BrouwerLyddane which can fail to converge for
    certain initial conditions).
    """
    if not OREKIT_AVAILABLE:
        raise RuntimeError("Orekit is not available.")

    frame = FramesFactory.getEME2000()
    date = _datetime_to_absolute(epoch)
    orbit = KeplerianOrbit(
        a_km * 1000.0,
        e,
        math.radians(i_deg),
        math.radians(argp_deg),
        math.radians(raan_deg),
        math.radians(ta_deg),
        PositionAngleType.TRUE,
        frame,
        date,
        Constants.WGS84_EARTH_MU,
    )
    try:
        return EcksteinHechlerPropagator(
            orbit,
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
            Constants.WGS84_EARTH_MU,
            Constants.WGS84_EARTH_C20,
            0.0, 0.0, 0.0, 0.0,  # J3-J6 = 0 (J2-only)
        )
    except Exception as exc:
        logger.warning("J2 propagator (EcksteinHechler) failed, falling back to Keplerian: %s", exc)
        return KeplerianPropagator(orbit)


def create_tle_propagator(tle_line1: str, tle_line2: str) -> Any:
    """Create a TLE (SGP4/SDP4) propagator."""
    if not OREKIT_AVAILABLE:
        raise RuntimeError("Orekit is not available.")

    tle = TLE(tle_line1, tle_line2)
    return TLEPropagator.selectExtrapolator(tle)


def get_sun():
    """Return Orekit Sun body."""
    return CelestialBodyFactory.getSun()


def get_earth_body():
    """Return Orekit Earth OneAxisEllipsoid."""
    return _get_earth()


def make_ground_station_frame(
    lat_deg: float, lon_deg: float, alt_m: float = 0.0
) -> Any:
    """Create a TopocentricFrame for a ground station."""
    if not OREKIT_AVAILABLE:
        raise RuntimeError("Orekit is not available.")

    earth = _get_earth()
    point = GeodeticPoint(
        math.radians(lat_deg), math.radians(lon_deg), alt_m
    )
    return TopocentricFrame(earth, point, "ground_station")

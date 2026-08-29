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

import os

import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

OREKIT_AVAILABLE = False
_orekit_initialized = False
_orekit_load_error: Optional[str] = None

try:
    import orekit_jpype
    import jpype

    if "JAVA_HOME" not in os.environ:
        try:
            import jdk4py
            os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
        except ImportError:
            pass

    if not jpype.isJVMStarted():
        orekit_jpype.initVM()

    from pathlib import Path as _Path
    from orekit_jpype.pyhelpers import setup_orekit_data
    # Use absolute path so it works regardless of CWD (e.g. when called from notebooks/)
    _orekit_data = str(_Path(__file__).parent.parent.parent.parent / "orekit-data.zip")
    setup_orekit_data(filenames=_orekit_data, from_pip_library=False)

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
    from org.orekit.utils import Constants, IERSConventions, OccultationEngine
    from org.orekit.models.earth import GeoMagneticFieldFactory
    from org.orekit.models.earth.atmosphere import HarrisPriester

    OREKIT_AVAILABLE = True
    _orekit_initialized = True
    logger.info("Orekit initialized successfully (data: %s).", _orekit_data)
except Exception as e:
    _orekit_load_error = repr(e)
    logger.warning(
        "Orekit failed to initialize; get_environment will return zeros. Reason: %s", e
    )


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
# MU_EARTH = 398600.4418e9  # m^3/s^2


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

    frame = _get_eci_frame()
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

    frame = _get_eci_frame()
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

@dataclass
class EnvironmentData:
    """Orbital and environmental quantities at one instant, in the ECI frame.

    Bundles everything the ADCS simulation needs from the propagator at a given
    time: the orbital state plus the geomagnetic field, sun direction, eclipse
    condition, and atmospheric density at the satellite's position.

    Attributes:
        r_eci: Position in the ECI frame [m], shape (3,).
        v_eci: Velocity in the ECI frame [m/s], shape (3,).
        b_field_eci: Geomagnetic field vector in the ECI frame [T], shape (3,).
        sun_vector_eci: Unit vector from the satellite to the sun in the ECI
            frame, shape (3,).
        eclipse: True when the satellite is in Earth's shadow.
        atmospheric_density: Local atmospheric mass density [kg/m^3].
    """

    r_eci: np.ndarray
    v_eci: np.ndarray
    b_field_eci: np.ndarray
    sun_vector_eci: np.ndarray
    eclipse: bool
    atmospheric_density: float


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type-only; avoids any import cycle and keeps Orekit out of configs
    from src.environment.orbital.adcs.configs import OrbitConfig


def _get_eci_frame():
    """
    To avoid mistakes, all frames can be changed from one place.
    """
    return FramesFactory.getGCRF()


def _vec3_to_np(v) -> np.ndarray:
    """Orekit Vector3D -> numpy (3,)"""
    return np.array([v.getX(), v.getY(), v.getZ()])

def _raan_from_ltan(epoch_date: Any, ltan_hours: float, sun: Any, frame: Any) -> float:
    """RAAN [deg] for a desired local time of the ascending node.

    Ω = α_sun + 15*(LTAN - 12), with α_sun the Sun's right ascension at epoch
    in the ECI frame. Uses the true ephemeris Sun (equation-of-time effect
    <~4 deg, negligible here). Result wrapped to [0, 360).
    """
    sun_pos = _vec3_to_np(sun.getPosition(epoch_date, frame))
    ra_sun_deg = np.degrees(np.arctan2(sun_pos[1], sun_pos[0]))
    return float((ra_sun_deg + 15.0 * (ltan_hours - 12.0)) % 360.0)

def _b_field_eci(pos_gcrf: Any, date: Any, earth: Any, igrf: Any, eci_frame: Any) -> np.ndarray:
    """Geomagnetic field at the spacecraft, expressed in ECI, in Tesla.

    ECI position -> geodetic (lat, lon, alt) on the WGS84 ellipsoid; IGRF gives
    the field in the local North-East-Down frame in nanoTesla; rotate NED->ECI
    via the geodetic basis vectors (transformed ITRF->ECI); convert nT -> Tesla.
    """
    gp = earth.transform(pos_gcrf, eci_frame, date)          # ECI pos -> geodetic
    elements = igrf.calculateField(                          # lat/lon rad, alt m
        gp.getLatitude(), gp.getLongitude(), gp.getAltitude()
    )
    b_ned = elements.getFieldVector()                        # (N, E, D), nanoTesla

    # Local NED basis (unit vectors in ITRF) transformed into the ECI frame.
    itrf = earth.getBodyFrame()
    to_eci = itrf.getTransformTo(eci_frame, date)
    north = _vec3_to_np(to_eci.transformVector(gp.getNorth()))
    east = _vec3_to_np(to_eci.transformVector(gp.getEast()))
    nadir = _vec3_to_np(to_eci.transformVector(gp.getNadir()))

    b_tesla = b_ned.getX() * north + b_ned.getY() * east + b_ned.getZ() * nadir
    return b_tesla                                    # nanoTesla -> Tesla

@dataclass
class _PropagatorContext:
    #to give get_environment clean access (_ctx.propagator, _ctx.epoch, _ctx.frame).
    propagator: Any   # Orekit analytical propagator
    epoch: Any        # Orekit AbsoluteDate; simulation t=0
    frame: Any        # Orekit ECI Frame (from _get_eci_frame)
    sun: Any          # Orekit CelestialBody (Sun), for the Sun vector
    occultation: Any  # Orekit OccultationEngine (Sun occulted by Earth), for eclipse
    earth: Any        # Orekit OneAxisEllipsoid (WGS84); geodetic conversion + ITRF frame
    igrf: Any         # Orekit GeoMagneticField (IGRF) at the epoch decimal year
    atmosphere: Any   # Orekit HarrisPriester atmosphere, for density

_ctx: Optional[_PropagatorContext] = None


def configure(orbit: "OrbitConfig") -> None:
    """Build the orbit propagator from config and store it for get_environment.
    """
    global _ctx
    if not OREKIT_AVAILABLE:
        raise RuntimeError(
            f"Orekit unavailable; the propagator cannot be configured. "
            f"Load error: {_orekit_load_error}"
        )

    epoch_date = _datetime_to_absolute(orbit.epoch)
    frame = _get_eci_frame()
    sun = CelestialBodyFactory.getSun()

    if orbit.ltan_hours is not None:
        raan_deg = _raan_from_ltan(epoch_date, orbit.ltan_hours, sun, frame)
        logger.info("Derived RAAN=%.3f deg from LTAN=%.2f h", raan_deg, orbit.ltan_hours)
    else:
        raan_deg = orbit.raan_deg

    a_km = Constants.WGS84_EARTH_EQUATORIAL_RADIUS / 1000.0 + orbit.altitude_km
    args = (
        a_km, orbit.eccentricity, orbit.inclination_deg,
        raan_deg, orbit.arg_perigee_deg, orbit.true_anomaly_deg, orbit.epoch,
    )
    if orbit.propagator_type == "j2":
        prop = create_j2_propagator(*args)
    elif orbit.propagator_type == "keplerian":
        prop = create_keplerian_propagator(*args)
    else:
        raise ValueError(f"Unknown propagator_type: {orbit.propagator_type!r}")

    earth = _get_earth()
    occultation = OccultationEngine(sun, Constants.SUN_RADIUS, earth)

    # IGRF at the epoch's decimal year (computed in Python to avoid the
    # version-dependent getDecimalYear argument order).
    year_start = datetime(orbit.epoch.year, 1, 1, tzinfo=orbit.epoch.tzinfo)
    year_end = datetime(orbit.epoch.year + 1, 1, 1, tzinfo=orbit.epoch.tzinfo)
    decimal_year = orbit.epoch.year + (
        (orbit.epoch - year_start).total_seconds()
        / (year_end - year_start).total_seconds()
    )

    try:
        igrf = GeoMagneticFieldFactory.getIGRF(decimal_year)
    except Exception as exc:
        igrf = None
        logger.warning(
            "IGRF unavailable; b_field_eci will be zero. Add IGRF.COF to "
            "orekit-data. Reason: %s", exc
        )

    # Harris-Priester atmosphere: embedded density table
    atmosphere = HarrisPriester(sun, earth, 6.0)

    _ctx = _PropagatorContext(
        propagator=prop,
        epoch=epoch_date,
        frame=frame,
        sun=sun,
        occultation=occultation,
        earth=earth,
        igrf=igrf,
        atmosphere=atmosphere,
    )


def get_environment(t: float) -> EnvironmentData:
    """Orbital + environmental data at time t [s] since the simulation epoch.
    """

    if _ctx is None:
        raise RuntimeError(
            "get_environment() called before configure(). There is no neutral "
            "default environment: zero field, zero density and zero Sun vector "
            "produce a run that completes and looks successful while modelling "
            "no physics at all."
        )

    target = _ctx.epoch.shiftedBy(float(t))
    state = _ctx.propagator.propagate(target)
    pv = state.getPVCoordinates(_ctx.frame)
    pos_v3d = pv.getPosition()
    r_eci = _vec3_to_np(pos_v3d)
    v_eci = _vec3_to_np(pv.getVelocity())

    sun_pos = _vec3_to_np(_ctx.sun.getPosition(target, _ctx.frame))
    sun_rel = sun_pos - r_eci
    sun_vector_eci = sun_rel / np.linalg.norm(sun_rel)

    angles = _ctx.occultation.angles(state)
    eclipse = bool(
        angles.getSeparation() - angles.getLimbRadius()
        + angles.getOccultedApparentRadius() < 0.0
    )

    if _ctx.igrf is not None:
        b_field_eci = _b_field_eci(pos_v3d, target, _ctx.earth, _ctx.igrf, _ctx.frame)
    else:
        b_field_eci = np.zeros(3)

    atmospheric_density = float(_ctx.atmosphere.getDensity(target, pos_v3d, _ctx.frame))

    return EnvironmentData(
        r_eci=r_eci,
        v_eci=v_eci,
        b_field_eci=b_field_eci,
        sun_vector_eci=sun_vector_eci,
        eclipse=eclipse,
        atmospheric_density=atmospheric_density,
    )
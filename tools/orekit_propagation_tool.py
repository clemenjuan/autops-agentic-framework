"""
Orekit High-Fidelity Propagation Tool

Numerical propagation with configurable force models, maneuver computation,
state conversions, and ground track analysis.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import math

OREKIT_AVAILABLE = False

try:
    from agent.data_pipeline.fetchers.orekit_setup import init_orekit
    if init_orekit():
        from org.orekit.frames import FramesFactory, TopocentricFrame
        from org.orekit.time import TimeScalesFactory, AbsoluteDate
        from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid, GeodeticPoint
        from org.orekit.orbits import KeplerianOrbit, CartesianOrbit, PositionAngleType, OrbitType
        from org.orekit.propagation.analytical import KeplerianPropagator
        from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
        from org.orekit.propagation.numerical import NumericalPropagator
        from org.orekit.propagation import SpacecraftState
        from org.orekit.forces.gravity.potential import GravityFieldFactory
        from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction
        from org.orekit.forces.drag import DragForce, IsotropicDrag
        from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient
        from org.orekit.models.earth.atmosphere import NRLMSISE00
        from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData
        from org.orekit.utils import Constants, IERSConventions, PVCoordinates
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
        from java.util import Arrays
        OREKIT_AVAILABLE = True
except ImportError as e:
    print(f"Orekit import failed: {e}")


# Constants
MU_EARTH = 398600.4418  # km^3/s^2
EARTH_RADIUS = 6378.137  # km


def get_utc():
    """Get UTC time scale."""
    return TimeScalesFactory.getUTC()


def get_frames():
    """Get commonly used reference frames."""
    if not OREKIT_AVAILABLE:
        return None
    return {
        'eme2000': FramesFactory.getEME2000(),
        'gcrf': FramesFactory.getGCRF(),
        'itrf': FramesFactory.getITRF(IERSConventions.IERS_2010, True),
        'teme': FramesFactory.getTEME()
    }


def get_earth():
    """Get Earth body model."""
    if not OREKIT_AVAILABLE:
        return None
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    return OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                           Constants.WGS84_EARTH_FLATTENING, itrf)


def datetime_to_absolute(dt: datetime) -> "AbsoluteDate":
    """Convert Python datetime to Orekit AbsoluteDate."""
    utc = get_utc()
    return AbsoluteDate(dt.year, dt.month, dt.day, dt.hour, dt.minute, 
                        float(dt.second + dt.microsecond / 1e6), utc)


def absolute_to_datetime(ad: "AbsoluteDate") -> datetime:
    """Convert Orekit AbsoluteDate to Python datetime."""
    utc = get_utc()
    components = ad.getComponents(utc)
    date = components.getDate()
    time = components.getTime()
    return datetime(date.getYear(), date.getMonth(), date.getDay(),
                   time.getHour(), time.getMinute(), int(time.getSecond()),
                   tzinfo=timezone.utc)


def keplerian_to_cartesian(a_km: float, e: float, i_deg: float, raan_deg: float,
                           argp_deg: float, ta_deg: float, epoch: datetime) -> Dict[str, Any]:
    """Convert Keplerian elements to Cartesian state."""
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not available'}
    
    try:
        frame = FramesFactory.getEME2000()
        date = datetime_to_absolute(epoch)
        
        orbit = KeplerianOrbit(
            a_km * 1000,  # Convert to meters
            e,
            math.radians(i_deg),
            math.radians(argp_deg),
            math.radians(raan_deg),
            math.radians(ta_deg),
            PositionAngleType.TRUE,
            frame,
            date,
            Constants.WGS84_EARTH_MU
        )
        
        pv = orbit.getPVCoordinates()
        pos = pv.getPosition()
        vel = pv.getVelocity()
        
        return {
            'position_eci_km': {'x': pos.getX()/1000, 'y': pos.getY()/1000, 'z': pos.getZ()/1000},
            'velocity_eci_km_s': {'x': vel.getX()/1000, 'y': vel.getY()/1000, 'z': vel.getZ()/1000},
            'epoch': epoch.isoformat()
        }
    except Exception as e:
        return {'error': str(e)}


def cartesian_to_keplerian(pos_km: Dict, vel_km_s: Dict, epoch: datetime) -> Dict[str, Any]:
    """Convert Cartesian state to Keplerian elements."""
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not available'}
    
    try:
        frame = FramesFactory.getEME2000()
        date = datetime_to_absolute(epoch)
        
        position = Vector3D(pos_km['x']*1000, pos_km['y']*1000, pos_km['z']*1000)
        velocity = Vector3D(vel_km_s['x']*1000, vel_km_s['y']*1000, vel_km_s['z']*1000)
        pv = PVCoordinates(position, velocity)
        
        orbit = KeplerianOrbit(pv, frame, date, Constants.WGS84_EARTH_MU)
        
        return {
            'semi_major_axis_km': orbit.getA() / 1000,
            'eccentricity': orbit.getE(),
            'inclination_deg': math.degrees(orbit.getI()),
            'raan_deg': math.degrees(orbit.getRightAscensionOfAscendingNode()),
            'arg_perigee_deg': math.degrees(orbit.getPerigeeArgument()),
            'true_anomaly_deg': math.degrees(orbit.getTrueAnomaly()),
            'mean_anomaly_deg': math.degrees(orbit.getMeanAnomaly()),
            'period_min': orbit.getKeplerianPeriod() / 60,
            'apogee_km': orbit.getA() * (1 + orbit.getE()) / 1000 - EARTH_RADIUS,
            'perigee_km': orbit.getA() * (1 - orbit.getE()) / 1000 - EARTH_RADIUS,
            'epoch': epoch.isoformat()
        }
    except Exception as e:
        return {'error': str(e)}


def propagate_tle(tle_line1: str, tle_line2: str, target_time: datetime) -> Dict[str, Any]:
    """Propagate TLE using SGP4/SDP4 (medium fidelity)."""
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not available'}
    
    try:
        tle = TLE(tle_line1, tle_line2)
        propagator = TLEPropagator.selectExtrapolator(tle)
        target = datetime_to_absolute(target_time)
        
        state = propagator.propagate(target)
        pv = state.getPVCoordinates()
        pos = pv.getPosition()
        vel = pv.getVelocity()
        
        # Get ground track
        earth = get_earth()
        geo = earth.transform(pos, state.getFrame(), state.getDate())
        
        return {
            'position_eci_km': {'x': pos.getX()/1000, 'y': pos.getY()/1000, 'z': pos.getZ()/1000},
            'velocity_eci_km_s': {'x': vel.getX()/1000, 'y': vel.getY()/1000, 'z': vel.getZ()/1000},
            'ground_track': {
                'latitude_deg': math.degrees(geo.getLatitude()),
                'longitude_deg': math.degrees(geo.getLongitude()),
                'altitude_km': geo.getAltitude() / 1000
            },
            'epoch': target_time.isoformat()
        }
    except Exception as e:
        return {'error': str(e)}


def create_numerical_propagator(initial_state: "SpacecraftState", 
                                 force_models: Dict[str, bool],
                                 spacecraft_mass: float = 1000.0,
                                 drag_area: float = 10.0,
                                 drag_cd: float = 2.2,
                                 srp_area: float = 10.0,
                                 srp_cr: float = 1.5) -> "NumericalPropagator":
    """Create numerical propagator with configurable force models."""
    
    min_step = 0.001
    max_step = 1000.0
    init_step = 60.0
    pos_tolerance = 1.0
    
    integrator = DormandPrince853Integrator(min_step, max_step, pos_tolerance, pos_tolerance)
    integrator.setInitialStepSize(init_step)
    
    propagator = NumericalPropagator(integrator)
    propagator.setInitialState(initial_state)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    
    # Gravity field (always included)
    gravity_degree = force_models.get('gravity_degree', 20)
    gravity_order = force_models.get('gravity_order', 20)
    gravity_provider = GravityFieldFactory.getNormalizedProvider(gravity_degree, gravity_order)
    earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    propagator.addForceModel(
        HolmesFeatherstoneAttractionModel(earth_frame, gravity_provider)
    )
    
    # Third body - Moon
    if force_models.get('moon', True):
        propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))
    
    # Third body - Sun
    if force_models.get('sun', True):
        propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getSun()))
    
    # Atmospheric drag
    if force_models.get('drag', False):
        try:
            sun = CelestialBodyFactory.getSun()
            earth = get_earth()
            weather_data = CssiSpaceWeatherData("SpaceWeather-All-v1.2.txt")
            atmosphere = NRLMSISE00(weather_data, sun, earth)
            spacecraft = IsotropicDrag(drag_area, drag_cd)
            propagator.addForceModel(DragForce(atmosphere, spacecraft))
        except Exception:
            pass  # Skip drag if weather data unavailable
    
    # Solar radiation pressure
    if force_models.get('srp', False):
        try:
            sun = CelestialBodyFactory.getSun()
            spacecraft = IsotropicRadiationSingleCoefficient(srp_area, srp_cr)
            propagator.addForceModel(
                SolarRadiationPressure(sun, get_earth(), spacecraft)
            )
        except Exception:
            pass
    
    return propagator


def propagate_numerical(tle_line1: str, tle_line2: str, duration_hours: float,
                        step_seconds: float = 60.0,
                        force_models: Optional[Dict] = None) -> Dict[str, Any]:
    """High-fidelity numerical propagation with configurable force models."""
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not available'}
    
    if force_models is None:
        force_models = {'gravity_degree': 20, 'gravity_order': 20, 'moon': True, 'sun': True}
    
    try:
        # Get initial state from TLE
        tle = TLE(tle_line1, tle_line2)
        tle_prop = TLEPropagator.selectExtrapolator(tle)
        initial_state = tle_prop.getInitialState()
        
        # Create numerical propagator
        propagator = create_numerical_propagator(initial_state, force_models)
        
        start = initial_state.getDate()
        earth = get_earth()
        trajectory = []
        
        t = 0.0
        end_offset = duration_hours * 3600
        
        while t <= end_offset:
            state = propagator.propagate(start.shiftedBy(t))
            pv = state.getPVCoordinates()
            pos = pv.getPosition()
            
            # Ground track
            geo = earth.transform(pos, state.getFrame(), state.getDate())
            
            trajectory.append({
                'time_offset_sec': t,
                'position_eci_km': {'x': pos.getX()/1000, 'y': pos.getY()/1000, 'z': pos.getZ()/1000},
                'ground_track': {
                    'lat': math.degrees(geo.getLatitude()),
                    'lon': math.degrees(geo.getLongitude()),
                    'alt_km': geo.getAltitude() / 1000
                }
            })
            t += step_seconds
        
        return {
            'status': 'success',
            'propagation_type': 'numerical',
            'force_models': force_models,
            'points': len(trajectory),
            'trajectory': trajectory
        }
    except Exception as e:
        return {'error': str(e)}


def compute_impulsive_maneuver(pos_km: Dict, vel_km_s: Dict, delta_v_km_s: Dict,
                                epoch: datetime) -> Dict[str, Any]:
    """Compute state after impulsive maneuver."""
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not available'}
    
    try:
        # Pre-maneuver state
        pre_elements = cartesian_to_keplerian(pos_km, vel_km_s, epoch)
        
        # Apply delta-v
        new_vel = {
            'x': vel_km_s['x'] + delta_v_km_s['x'],
            'y': vel_km_s['y'] + delta_v_km_s['y'],
            'z': vel_km_s['z'] + delta_v_km_s['z']
        }
        
        # Post-maneuver state
        post_elements = cartesian_to_keplerian(pos_km, new_vel, epoch)
        
        dv_mag = math.sqrt(delta_v_km_s['x']**2 + delta_v_km_s['y']**2 + delta_v_km_s['z']**2)
        
        return {
            'delta_v_magnitude_km_s': dv_mag,
            'pre_maneuver': pre_elements,
            'post_maneuver': post_elements,
            'new_velocity_km_s': new_vel
        }
    except Exception as e:
        return {'error': str(e)}


def compute_hohmann_transfer(r1_km: float, r2_km: float, i1_deg: float = 0, 
                              i2_deg: float = 0) -> Dict[str, Any]:
    """Compute Hohmann transfer with optional plane change."""
    v1 = math.sqrt(MU_EARTH / r1_km)
    v2 = math.sqrt(MU_EARTH / r2_km)
    
    a_transfer = (r1_km + r2_km) / 2
    v_transfer_perigee = math.sqrt(MU_EARTH * (2/r1_km - 1/a_transfer))
    v_transfer_apogee = math.sqrt(MU_EARTH * (2/r2_km - 1/a_transfer))
    
    dv1 = abs(v_transfer_perigee - v1)
    dv2 = abs(v2 - v_transfer_apogee)
    
    # Plane change at apogee (more efficient)
    di = abs(i2_deg - i1_deg)
    if di > 0:
        dv_plane = 2 * v_transfer_apogee * math.sin(math.radians(di/2))
        dv2 = math.sqrt(dv2**2 + dv_plane**2 - 2*dv2*dv_plane*math.cos(math.radians(di/2)))
    
    transfer_period = 2 * math.pi * math.sqrt(a_transfer**3 / MU_EARTH)
    transfer_time = transfer_period / 2
    
    return {
        'dv1_km_s': dv1,
        'dv2_km_s': dv2,
        'total_dv_km_s': dv1 + dv2,
        'transfer_time_sec': transfer_time,
        'transfer_time_min': transfer_time / 60,
        'transfer_sma_km': a_transfer,
        'plane_change_deg': di
    }


def compute_bielliptic_transfer(r1_km: float, r2_km: float, rb_km: float) -> Dict[str, Any]:
    """Compute bi-elliptic transfer (more efficient for large radius ratios)."""
    v1 = math.sqrt(MU_EARTH / r1_km)
    
    # First transfer ellipse
    a1 = (r1_km + rb_km) / 2
    v1_transfer = math.sqrt(MU_EARTH * (2/r1_km - 1/a1))
    dv1 = abs(v1_transfer - v1)
    
    # At intermediate point
    vb1 = math.sqrt(MU_EARTH * (2/rb_km - 1/a1))
    
    # Second transfer ellipse
    a2 = (rb_km + r2_km) / 2
    vb2 = math.sqrt(MU_EARTH * (2/rb_km - 1/a2))
    dv2 = abs(vb2 - vb1)
    
    # Final circularization
    v2_transfer = math.sqrt(MU_EARTH * (2/r2_km - 1/a2))
    v2_circular = math.sqrt(MU_EARTH / r2_km)
    dv3 = abs(v2_circular - v2_transfer)
    
    # Transfer times
    t1 = math.pi * math.sqrt(a1**3 / MU_EARTH)
    t2 = math.pi * math.sqrt(a2**3 / MU_EARTH)
    
    return {
        'dv1_km_s': dv1,
        'dv2_km_s': dv2,
        'dv3_km_s': dv3,
        'total_dv_km_s': dv1 + dv2 + dv3,
        'transfer_time_sec': t1 + t2,
        'intermediate_radius_km': rb_km
    }


def compute_station_keeping(a_km: float, e: float, i_deg: float,
                             duration_days: float = 365) -> Dict[str, Any]:
    """Estimate station-keeping delta-v budget."""
    # Atmospheric drag (for LEO)
    alt_km = a_km - EARTH_RADIUS
    if alt_km < 1000:
        # Simplified drag estimate
        scale_height = 50  # km, varies with altitude
        rho_factor = math.exp(-alt_km / scale_height)
        dv_drag = 0.1 * rho_factor * duration_days  # Simplified model
    else:
        dv_drag = 0
    
    # Solar radiation pressure (for GEO)
    if alt_km > 30000:
        dv_srp = 0.05 * duration_days / 365  # ~50 m/s per year for GEO
    else:
        dv_srp = 0
    
    # Inclination maintenance (for GEO)
    if alt_km > 30000:
        dv_inc = 50 * duration_days / 365  # ~50 m/s per year
    else:
        dv_inc = 0
    
    # Eccentricity maintenance
    dv_ecc = 2 * duration_days / 365  # ~2 m/s per year
    
    return {
        'duration_days': duration_days,
        'dv_drag_km_s': dv_drag / 1000,
        'dv_srp_km_s': dv_srp / 1000,
        'dv_inclination_km_s': dv_inc / 1000,
        'dv_eccentricity_km_s': dv_ecc / 1000,
        'total_dv_km_s': (dv_drag + dv_srp + dv_inc + dv_ecc) / 1000,
        'orbit_altitude_km': alt_km
    }


def compute_ground_track(tle_line1: str, tle_line2: str, duration_hours: float = 2,
                          step_seconds: float = 30) -> Dict[str, Any]:
    """Compute satellite ground track."""
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not available'}
    
    try:
        tle = TLE(tle_line1, tle_line2)
        propagator = TLEPropagator.selectExtrapolator(tle)
        earth = get_earth()
        
        start = propagator.getInitialState().getDate()
        track = []
        
        t = 0.0
        end = duration_hours * 3600
        
        while t <= end:
            state = propagator.propagate(start.shiftedBy(t))
            pos = state.getPVCoordinates().getPosition()
            geo = earth.transform(pos, state.getFrame(), state.getDate())
            
            track.append({
                'time_offset_sec': t,
                'lat': math.degrees(geo.getLatitude()),
                'lon': math.degrees(geo.getLongitude()),
                'alt_km': geo.getAltitude() / 1000
            })
            t += step_seconds
        
        return {'status': 'success', 'points': len(track), 'ground_track': track}
    except Exception as e:
        return {'error': str(e)}


def compute_visibility(tle_line1: str, tle_line2: str, ground_lat: float, ground_lon: float,
                        min_elevation_deg: float = 10, duration_hours: float = 24) -> Dict[str, Any]:
    """Compute visibility windows from a ground station."""
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not available'}
    
    try:
        tle = TLE(tle_line1, tle_line2)
        propagator = TLEPropagator.selectExtrapolator(tle)
        earth = get_earth()
        
        # Ground station
        station_point = GeodeticPoint(math.radians(ground_lat), math.radians(ground_lon), 0.0)
        station_frame = TopocentricFrame(earth, station_point, "station")
        
        start = propagator.getInitialState().getDate()
        passes = []
        in_pass = False
        pass_start = None
        max_el = 0
        
        t = 0.0
        step = 30.0  # 30 second steps
        end = duration_hours * 3600
        
        while t <= end:
            state = propagator.propagate(start.shiftedBy(t))
            pos = state.getPVCoordinates().getPosition()
            
            topo = station_frame.getTrackingCoordinates(pos, state.getFrame(), state.getDate())
            elevation = math.degrees(topo.getElevation())
            
            if elevation >= min_elevation_deg:
                if not in_pass:
                    in_pass = True
                    pass_start = t
                    max_el = elevation
                else:
                    max_el = max(max_el, elevation)
            else:
                if in_pass:
                    passes.append({
                        'start_offset_sec': pass_start,
                        'end_offset_sec': t,
                        'duration_sec': t - pass_start,
                        'max_elevation_deg': max_el
                    })
                    in_pass = False
            
            t += step
        
        return {
            'status': 'success',
            'ground_station': {'lat': ground_lat, 'lon': ground_lon},
            'min_elevation_deg': min_elevation_deg,
            'passes': passes
        }
    except Exception as e:
        return {'error': str(e)}


def compute_visibility_from_orbit(orbit_elements: Dict[str, Any], epoch: datetime,
                                   ground_lat: float, ground_lon: float,
                                   min_elevation_deg: float = 10, duration_hours: float = 24,
                                   force_models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute visibility windows from orbital elements (Keplerian or Cartesian)."""
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not available'}
    
    try:
        frame = FramesFactory.getEME2000()
        earth = get_earth()
        date = datetime_to_absolute(epoch)
        
        # Determine if input is Keplerian or Cartesian
        if 'semi_major_axis_km' in orbit_elements or 'a_km' in orbit_elements:
            # Keplerian elements
            a_km = orbit_elements.get('semi_major_axis_km') or orbit_elements.get('a_km')
            e = orbit_elements.get('eccentricity') or orbit_elements.get('e', 0)
            i_deg = orbit_elements.get('inclination_deg') or orbit_elements.get('i_deg')
            raan_deg = orbit_elements.get('raan_deg') or orbit_elements.get('omega_deg')
            argp_deg = orbit_elements.get('arg_perigee_deg') or orbit_elements.get('w_deg')
            ta_deg = orbit_elements.get('true_anomaly_deg') or orbit_elements.get('ta_deg') or orbit_elements.get('mean_anomaly_deg', 0)
            
            orbit = KeplerianOrbit(
                a_km * 1000, e, math.radians(i_deg), math.radians(raan_deg),
                math.radians(argp_deg), math.radians(ta_deg),
                PositionAngleType.TRUE, frame, date, Constants.WGS84_EARTH_MU
            )
        elif 'position_km' in orbit_elements or 'pos_km' in orbit_elements:
            # Cartesian elements
            pos = orbit_elements.get('position_km') or orbit_elements.get('pos_km')
            vel = orbit_elements.get('velocity_km_s') or orbit_elements.get('vel_km_s')
            
            pv = PVCoordinates(
                Vector3D(float(pos['x']) * 1000, float(pos['y']) * 1000, float(pos['z']) * 1000),
                Vector3D(float(vel['x']) * 1000, float(vel['y']) * 1000, float(vel['z']) * 1000)
            )
            orbit = CartesianOrbit(pv, frame, date, Constants.WGS84_EARTH_MU)
        else:
            return {'error': 'Invalid orbit elements. Provide Keplerian (semi_major_axis_km, eccentricity, etc.) or Cartesian (position_km, velocity_km_s)'}
        
        # Create propagator
        if force_models:
            initial_state = SpacecraftState(orbit)
            propagator = create_numerical_propagator(initial_state, force_models)
        else:
            propagator = KeplerianPropagator(orbit)
        
        # Ground station
        station_point = GeodeticPoint(math.radians(ground_lat), math.radians(ground_lon), 0.0)
        station_frame = TopocentricFrame(earth, station_point, "station")
        
        start = propagator.getInitialState().getDate()
        passes = []
        in_pass = False
        pass_start = None
        max_el = 0
        
        t = 0.0
        step = 30.0
        end = duration_hours * 3600
        
        while t <= end:
            state = propagator.propagate(start.shiftedBy(t))
            pos = state.getPVCoordinates().getPosition()
            
            topo = station_frame.getTrackingCoordinates(pos, state.getFrame(), state.getDate())
            elevation = math.degrees(topo.getElevation())
            
            if elevation >= min_elevation_deg:
                if not in_pass:
                    in_pass = True
                    pass_start = t
                    max_el = elevation
                else:
                    max_el = max(max_el, elevation)
            else:
                if in_pass:
                    passes.append({
                        'start_offset_sec': pass_start,
                        'end_offset_sec': t,
                        'duration_sec': t - pass_start,
                        'max_elevation_deg': max_el
                    })
                    in_pass = False
            
            t += step
        
        return {
            'status': 'success',
            'ground_station': {'lat': ground_lat, 'lon': ground_lon},
            'min_elevation_deg': min_elevation_deg,
            'passes': passes
        }
    except Exception as e:
        return {'error': str(e)}


async def execute(params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Orekit propagation tool."""
    action = params.get('action', 'propagate')
    
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not available. Run: uv pip install orekit-jpype'}
    
    if action == 'propagate':
        tle1 = params.get('tle_line1')
        tle2 = params.get('tle_line2')
        target = params.get('target_time')
        if not tle1 or not tle2:
            return {'error': 'tle_line1 and tle_line2 required'}
        if isinstance(target, str):
            target = datetime.fromisoformat(target.replace('Z', '+00:00'))
        elif target is None:
            target = datetime.now(timezone.utc)
        return propagate_tle(tle1, tle2, target)
    
    elif action == 'propagate_numerical':
        tle1 = params.get('tle_line1')
        tle2 = params.get('tle_line2')
        hours = params.get('duration_hours', 2)
        step = params.get('step_seconds', 60)
        force_models = params.get('force_models', {'gravity_degree': 20, 'moon': True, 'sun': True})
        if not tle1 or not tle2:
            return {'error': 'tle_line1 and tle_line2 required'}
        return propagate_numerical(tle1, tle2, hours, step, force_models)
    
    elif action == 'keplerian_to_cartesian':
        return keplerian_to_cartesian(
            params.get('semi_major_axis_km'),
            params.get('eccentricity'),
            params.get('inclination_deg'),
            params.get('raan_deg'),
            params.get('arg_perigee_deg'),
            params.get('true_anomaly_deg'),
            datetime.fromisoformat(params.get('epoch', datetime.now(timezone.utc).isoformat()))
        )
    
    elif action == 'cartesian_to_keplerian':
        pos = params.get('position_km')
        vel = params.get('velocity_km_s')
        epoch = params.get('epoch', datetime.now(timezone.utc).isoformat())
        if isinstance(epoch, str):
            epoch = datetime.fromisoformat(epoch.replace('Z', '+00:00'))
        return cartesian_to_keplerian(pos, vel, epoch)
    
    elif action == 'compute_hohmann':
        return compute_hohmann_transfer(
            params.get('initial_radius_km'),
            params.get('target_radius_km'),
            params.get('initial_inclination_deg', 0),
            params.get('target_inclination_deg', 0)
        )
    
    elif action == 'compute_bielliptic':
        return compute_bielliptic_transfer(
            params.get('initial_radius_km'),
            params.get('target_radius_km'),
            params.get('intermediate_radius_km')
        )
    
    elif action == 'compute_impulsive':
        pos = params.get('position_km')
        vel = params.get('velocity_km_s')
        dv = params.get('delta_v_km_s')
        epoch = datetime.fromisoformat(params.get('epoch', datetime.now(timezone.utc).isoformat()))
        return compute_impulsive_maneuver(pos, vel, dv, epoch)
    
    elif action == 'station_keeping':
        return compute_station_keeping(
            params.get('semi_major_axis_km'),
            params.get('eccentricity', 0),
            params.get('inclination_deg', 0),
            params.get('duration_days', 365)
        )
    
    elif action == 'ground_track':
        tle1 = params.get('tle_line1')
        tle2 = params.get('tle_line2')
        if not tle1 or not tle2:
            return {'error': 'tle_line1 and tle_line2 required'}
        return compute_ground_track(tle1, tle2, 
                                    params.get('duration_hours', 2),
                                    params.get('step_seconds', 30))
    
    elif action == 'visibility':
        tle1 = params.get('tle_line1')
        tle2 = params.get('tle_line2')
        if not tle1 or not tle2:
            return {'error': 'tle_line1 and tle_line2 required'}
        return compute_visibility(tle1, tle2,
                                  params.get('ground_lat'),
                                  params.get('ground_lon'),
                                  params.get('min_elevation_deg', 10),
                                  params.get('duration_hours', 24))
    
    else:
        return {'error': f'Unknown action: {action}. Available: propagate, propagate_numerical, keplerian_to_cartesian, cartesian_to_keplerian, compute_hohmann, compute_bielliptic, compute_impulsive, station_keeping, ground_track, visibility'}

import requests
import os
import math
from datetime import datetime, timedelta, timezone

API_BASE_URL = os.getenv('SATELLITE_API_URL', 'http://localhost:8000')

try:
    from tools.orekit_propagation_tool import propagate_tle, propagate_numerical, cartesian_to_keplerian, OREKIT_AVAILABLE
except ImportError:
    OREKIT_AVAILABLE = False


def parse_tle_elements(tle_line1: str, tle_line2: str) -> dict:
    """Parse orbital elements directly from TLE lines (no Orekit needed)."""
    # TLE Line 2 contains orbital elements
    # Format: 2 NNNNN III.IIII RRR.RRRR EEEEEEE AAA.AAAA MMM.MMMM NN.NNNNNNNN
    try:
        inc = float(tle_line2[8:16].strip())          # Inclination (degrees)
        raan = float(tle_line2[17:25].strip())        # RAAN (degrees)
        ecc = float('0.' + tle_line2[26:33].strip())  # Eccentricity (decimal assumed)
        argp = float(tle_line2[34:42].strip())        # Argument of perigee (degrees)
        mean_anom = float(tle_line2[43:51].strip())   # Mean anomaly (degrees)
        mean_motion = float(tle_line2[52:63].strip()) # Mean motion (rev/day)
        
        # Calculate semi-major axis from mean motion
        # n = sqrt(mu/a^3) -> a = (mu/n^2)^(1/3)
        mu = 398600.4418  # km^3/s^2
        n_rad_s = mean_motion * 2 * math.pi / 86400  # Convert rev/day to rad/s
        a_km = (mu / (n_rad_s ** 2)) ** (1/3)
        
        # Calculate apogee and perigee
        earth_radius = 6378.137  # km
        apogee_km = a_km * (1 + ecc) - earth_radius
        perigee_km = a_km * (1 - ecc) - earth_radius
        
        # Period
        period_min = 1440 / mean_motion  # minutes
        
        return {
            'semi_major_axis_km': round(a_km, 3),
            'eccentricity': round(ecc, 7),
            'inclination_deg': round(inc, 4),
            'raan_deg': round(raan, 4),
            'arg_perigee_deg': round(argp, 4),
            'mean_anomaly_deg': round(mean_anom, 4),
            'mean_motion_rev_day': round(mean_motion, 8),
            'period_min': round(period_min, 2),
            'apogee_altitude_km': round(apogee_km, 2),
            'perigee_altitude_km': round(perigee_km, 2)
        }
    except Exception as e:
        return {'error': f'Failed to parse TLE: {str(e)}'}

async def get_satellite(params):
    """Get satellite metadata and current orbital elements."""
    norad_id = params.get('norad_id')
    include_orbit = params.get('include_orbit', True)
    
    if not norad_id:
        return {'error': 'norad_id parameter required'}
    
    try:
        # Get metadata
        resp = requests.get(f"{API_BASE_URL}/satellites/{norad_id}", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        
        if 'error' in data:
            return {'error': data['error']}
        
        result = {'status': 'success', 'satellite': data}
        
        # Also fetch current orbital elements from TLE
        if include_orbit:
            try:
                tle_resp = requests.get(f"{API_BASE_URL}/tle/{norad_id}/history", params={'days': 1}, timeout=5.0)
                tle_resp.raise_for_status()
                tle_data = tle_resp.json()
                
                if tle_data.get('history'):
                    tle = tle_data['history'][0]
                    orbital_elements = parse_tle_elements(tle['tle_line1'], tle['tle_line2'])
                    result['orbital_elements'] = orbital_elements
                    result['tle_epoch'] = tle.get('epoch')
            except Exception:
                pass  # Orbital elements optional
        
        return result
    except requests.exceptions.ConnectionError:
        return {
            'error': 'Satellite API server not running',
            'message': f'Cannot connect to {API_BASE_URL}. Start the server with: python run_satellite_api.py'
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {'error': f'Satellite with NORAD ID {norad_id} not found'}
        return {'error': f'Failed to fetch satellite data: {str(e)}'}
    except Exception as e:
        return {'error': f'Failed to fetch satellite data: {str(e)}'}

async def get_orbital_elements(params):
    """Get current orbital elements (Keplerian) for a satellite."""
    norad_id = params.get('norad_id')
    
    if not norad_id:
        return {'error': 'norad_id parameter required'}
    
    try:
        resp = requests.get(f"{API_BASE_URL}/tle/{norad_id}/history", params={'days': 1}, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        
        if not data.get('history'):
            return {'error': f'No TLE found for NORAD ID {norad_id}'}
        
        tle = data['history'][0]
        elements = parse_tle_elements(tle['tle_line1'], tle['tle_line2'])
        
        if 'error' in elements:
            return elements
        
        return {
            'status': 'success',
            'norad_id': norad_id,
            'epoch': tle.get('epoch'),
            'orbital_elements': elements
        }
    except requests.exceptions.ConnectionError:
        return {
            'error': 'Satellite API server not running',
            'message': f'Cannot connect to {API_BASE_URL}. Start the server with: python run_satellite_api.py'
        }
    except Exception as e:
        return {'error': f'Failed to get orbital elements: {str(e)}'}


async def get_tle_history(params):
    norad_id = params.get('norad_id')
    days = params.get('days', 30)
    
    if not norad_id:
        return {'error': 'norad_id parameter required'}
    
    try:
        resp = requests.get(f"{API_BASE_URL}/tle/{norad_id}/history", params={'days': days}, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return {'status': 'success', 'data': data}
    except requests.exceptions.ConnectionError:
        return {
            'error': 'Satellite API server not running',
            'message': f'Cannot connect to {API_BASE_URL}. Start the server with: python run_satellite_api.py'
        }
    except Exception as e:
        return {'error': f'Failed to fetch TLE history: {str(e)}'}

async def get_maneuvers(params):
    satellite_id = params.get('satellite_id')
    min_confidence = params.get('min_confidence', 0.0)
    days = params.get('days', 30)
    
    query_params = {'min_confidence': min_confidence, 'days': days}
    if satellite_id:
        query_params['satellite_id'] = satellite_id
    
    try:
        resp = requests.get(f"{API_BASE_URL}/maneuvers", params=query_params, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return {'status': 'success', 'data': data}
    except requests.exceptions.ConnectionError:
        return {
            'error': 'Satellite API server not running',
            'message': f'Cannot connect to {API_BASE_URL}. Start the server with: python run_satellite_api.py'
        }
    except Exception as e:
        return {'error': f'Failed to fetch maneuvers: {str(e)}'}

async def predict_position(params):
    """Predict satellite position at a future time using Orekit."""
    norad_id = params.get('norad_id')
    target_time = params.get('target_time')
    
    if not norad_id:
        return {'error': 'norad_id parameter required'}
    
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not available for high-precision prediction'}
    
    try:
        resp = requests.get(f"{API_BASE_URL}/tle/{norad_id}/history", params={'days': 1}, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        
        if not data.get('history'):
            return {'error': f'No TLE found for NORAD ID {norad_id}'}
        
        tle = data['history'][0]
        if isinstance(target_time, str):
            target_time = datetime.fromisoformat(target_time)
        elif target_time is None:
            target_time = datetime.utcnow() + timedelta(hours=1)
        
        result = propagate_tle(tle['tle_line1'], tle['tle_line2'], target_time)
        result['norad_id'] = norad_id
        return {'status': 'success', 'prediction': result}
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}


async def calculate_passes(params):
    """Calculate visibility passes for a satellite over a ground location using Orekit.
    
    Flexible input options:
    1. TLE directly: tle_line1, tle_line2 (optional: tle_epoch)
    2. Orbital elements: semi_major_axis_km, eccentricity, inclination_deg, etc. (required: epoch)
    3. NORAD ID: norad_id (fetches TLE from API)
    
    Ground station options:
    1. Location name: location='ottobrunn' or 'oberpfaffenhofen'
    2. Coordinates: ground_lat, ground_lon (or sensor_lat, sensor_lon)
    3. Ground station dict: ground_station={'lat': X, 'lon': Y}
    """
    hours_ahead = params.get('hours_ahead', 24)
    min_elevation = params.get('min_elevation', 10)
    
    # Known locations
    LOCATIONS = {
        'ottobrunn': (48.052453328999675, 11.654863033454957),
        'oberpfaffenhofen': (48.0864, 11.2800),
    }
    
    # Get ground station coordinates
    ground_station = params.get('ground_station')
    location_name = None
    if isinstance(ground_station, dict):
        ground_lat = ground_station.get('lat') or ground_station.get('latitude')
        ground_lon = ground_station.get('lon') or ground_station.get('longitude')
        location_name = ground_station.get('name', 'custom')
    else:
        location = params.get('location', '').lower()
        if location in LOCATIONS:
            ground_lat, ground_lon = LOCATIONS[location]
            location_name = location
        else:
            ground_lat = params.get('sensor_lat') or params.get('ground_lat') or params.get('latitude')
            ground_lon = params.get('sensor_lon') or params.get('ground_lon') or params.get('longitude')
            location_name = 'custom'
    
    if ground_lat is None or ground_lon is None:
        return {'error': 'Ground station required: use location parameter (ottobrunn, oberpfaffenhofen), ground_lat/ground_lon, or ground_station dict'}
    
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not available. Install with: uv pip install orekit-jpype'}
    
    try:
        from tools.orekit_propagation_tool import compute_visibility, compute_visibility_from_orbit
        from datetime import datetime
        
        # Determine input type and compute passes
        tle_line1 = params.get('tle_line1')
        tle_line2 = params.get('tle_line2')
        norad_id = params.get('norad_id')
        
        # Option 1: TLE provided directly
        if tle_line1 and tle_line2:
            result = compute_visibility(
                tle_line1, tle_line2,
                ground_lat, ground_lon,
                min_elevation, hours_ahead
            )
            if result.get('status') == 'success':
                return {
                    'status': 'success',
                    'input_type': 'tle',
                    'ground_station': {'lat': ground_lat, 'lon': ground_lon, 'name': location_name},
                    'hours_ahead': hours_ahead,
                    'min_elevation_deg': min_elevation,
                    'passes': result.get('passes', []),
                    'tle_epoch': params.get('tle_epoch')
                }
            return result
        
        # Option 2: Orbital elements provided
        orbit_elements = params.get('orbital_elements') or params.get('orbit_elements')
        if orbit_elements or any(k in params for k in ['semi_major_axis_km', 'a_km', 'inclination_deg', 'i_deg']):
            if orbit_elements:
                elements = orbit_elements
            else:
                elements = {
                    'semi_major_axis_km': params.get('semi_major_axis_km') or params.get('a_km'),
                    'eccentricity': params.get('eccentricity') or params.get('e', 0),
                    'inclination_deg': params.get('inclination_deg') or params.get('i_deg'),
                    'raan_deg': params.get('raan_deg') or params.get('omega_deg'),
                    'arg_perigee_deg': params.get('arg_perigee_deg') or params.get('w_deg'),
                    'true_anomaly_deg': params.get('true_anomaly_deg') or params.get('ta_deg') or params.get('mean_anomaly_deg', 0)
                }
            
            epoch_str = params.get('epoch') or params.get('tle_epoch')
            if isinstance(epoch_str, str):
                epoch = datetime.fromisoformat(epoch_str.replace('Z', '+00:00'))
            elif isinstance(epoch_str, datetime):
                epoch = epoch_str
            else:
                return {'error': 'epoch required when using orbital elements'}
            
            force_models = params.get('force_models')
            result = compute_visibility_from_orbit(
                elements, epoch, ground_lat, ground_lon,
                min_elevation, hours_ahead, force_models
            )
            
            if result.get('status') == 'success':
                return {
                    'status': 'success',
                    'input_type': 'orbital_elements',
                    'ground_station': {'lat': ground_lat, 'lon': ground_lon, 'name': location if 'location' in locals() else 'custom'},
                    'hours_ahead': hours_ahead,
                    'min_elevation_deg': min_elevation,
                    'passes': result.get('passes', []),
                    'epoch': epoch_str
                }
            return result
        
        # Option 3: NORAD ID (fetch TLE from API)
        if norad_id:
            resp = requests.get(f"{API_BASE_URL}/tle/{norad_id}/history", params={'days': 1}, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            
            if not data.get('history'):
                return {'error': f'No TLE found for NORAD ID {norad_id}'}
            
            tle = data['history'][0]
            result = compute_visibility(
                tle['tle_line1'], tle['tle_line2'],
                ground_lat, ground_lon,
                min_elevation, hours_ahead
            )
            
            if result.get('status') == 'success':
                return {
                    'status': 'success',
                    'input_type': 'norad_id',
                    'norad_id': norad_id,
                    'ground_station': {'lat': ground_lat, 'lon': ground_lon, 'name': location_name},
                    'hours_ahead': hours_ahead,
                    'min_elevation_deg': min_elevation,
                    'passes': result.get('passes', []),
                    'tle_epoch': tle.get('epoch')
                }
            return result
        
        return {'error': 'No valid input provided. Use tle_line1/tle_line2, orbital_elements with epoch, or norad_id'}
        
    except requests.exceptions.ConnectionError:
        return {'error': 'Satellite API server not running. Start with: python run_satellite_api.py'}
    except Exception as e:
        return {'error': f'Pass calculation failed: {str(e)}'}


async def get_orbit_trajectory(params):
    """Get past and predicted orbit trajectory for visualization."""
    norad_id = params.get('norad_id')
    past_minutes = params.get('past_minutes', 45)
    prediction_minutes = params.get('prediction_minutes', 90)
    
    if not norad_id:
        return {'error': 'norad_id parameter required'}
    
    try:
        resp = requests.get(f"{API_BASE_URL}/tle/{norad_id}/history", params={'days': 1}, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        
        if not data.get('history'):
            return {'error': f'No TLE found for NORAD ID {norad_id}'}
        
        tle = data['history'][0]
        
        if OREKIT_AVAILABLE:
            total_hours = (past_minutes + prediction_minutes) / 60.0
            trajectory = propagate_numerical(tle['tle_line1'], tle['tle_line2'], total_hours, 60)
            
            past_count = past_minutes
            return {
                'status': 'success',
                'norad_id': norad_id,
                'past_trajectory': trajectory[:past_count],
                'prediction_trajectory': trajectory[past_count:]
            }
        else:
            return {
                'status': 'partial',
                'norad_id': norad_id,
                'tle': tle,
                'message': 'Orekit not available - use frontend satellite.js for propagation'
            }
    except Exception as e:
        return {'error': f'Trajectory fetch failed: {str(e)}'}


async def execute(params):
    action = params.get('action', 'get_satellite')
    
    actions = {
        'get_satellite': get_satellite,
        'get_orbital_elements': get_orbital_elements,
        'get_tle_history': get_tle_history,
        'get_maneuvers': get_maneuvers,
        'predict_position': predict_position,
        'calculate_passes': calculate_passes,
        'get_orbit_trajectory': get_orbit_trajectory
    }
    
    if action not in actions:
        return {'error': f'Unknown action: {action}', 'available': list(actions.keys())}
    
    return await actions[action](params)

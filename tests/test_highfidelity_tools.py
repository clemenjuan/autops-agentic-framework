"""
Test high-fidelity orbital mechanics tools.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio


# ISS TLE for testing
ISS_TLE1 = "1 25544U 98067A   24015.50000000  .00016717  00000-0  30000-3 0  9993"
ISS_TLE2 = "2 25544  51.6400 100.0000 0007000  90.0000 270.0000 15.50000000400000"


def test_propagate_tle():
    """Test basic TLE propagation."""
    from tools.orekit_propagation_tool import propagate_tle
    from datetime import datetime, timezone
    
    target = datetime(2024, 1, 20, 12, 0, 0, tzinfo=timezone.utc)
    result = propagate_tle(ISS_TLE1, ISS_TLE2, target)
    
    assert 'error' not in result, f"Error: {result.get('error')}"
    assert 'position_eci_km' in result
    assert 'ground_track' in result
    print(f"Position: {result['position_eci_km']}")
    print(f"Ground track: {result['ground_track']}")


def test_state_conversion():
    """Test Keplerian to Cartesian conversion."""
    from tools.orekit_propagation_tool import keplerian_to_cartesian, cartesian_to_keplerian
    from datetime import datetime, timezone
    
    epoch = datetime(2024, 1, 20, 12, 0, 0, tzinfo=timezone.utc)
    
    # ISS-like orbit
    result = keplerian_to_cartesian(
        a_km=6778,
        e=0.0007,
        i_deg=51.64,
        raan_deg=100.0,
        argp_deg=90.0,
        ta_deg=270.0,
        epoch=epoch
    )
    
    assert 'error' not in result, f"Error: {result.get('error')}"
    print(f"Cartesian state: {result}")
    
    # Convert back
    back = cartesian_to_keplerian(
        result['position_eci_km'],
        result['velocity_eci_km_s'],
        epoch
    )
    
    assert 'error' not in back, f"Error: {back.get('error')}"
    print(f"Keplerian elements: {back}")
    assert abs(back['semi_major_axis_km'] - 6778) < 1, "SMA mismatch"


def test_hohmann_transfer():
    """Test Hohmann transfer computation."""
    from tools.orekit_propagation_tool import compute_hohmann_transfer
    
    # LEO to GEO transfer
    result = compute_hohmann_transfer(
        r1_km=6778,  # LEO
        r2_km=42164,  # GEO
        i1_deg=28.5,
        i2_deg=0  # Plane change to equatorial
    )
    
    print(f"Hohmann transfer: {result}")
    assert result['total_dv_km_s'] > 0
    assert result['transfer_time_min'] > 0


def test_bielliptic_transfer():
    """Test bi-elliptic transfer computation."""
    from tools.orekit_propagation_tool import compute_bielliptic_transfer
    
    result = compute_bielliptic_transfer(
        r1_km=6778,
        r2_km=42164,
        rb_km=100000  # High intermediate orbit
    )
    
    print(f"Bi-elliptic transfer: {result}")
    assert result['total_dv_km_s'] > 0


def test_ground_track():
    """Test ground track computation."""
    from tools.orekit_propagation_tool import compute_ground_track
    
    result = compute_ground_track(ISS_TLE1, ISS_TLE2, duration_hours=1.5)
    
    assert result.get('status') == 'success', f"Error: {result.get('error')}"
    print(f"Ground track points: {result['points']}")
    assert result['points'] > 0
    
    # Check first and last points
    track = result['ground_track']
    print(f"First point: lat={track[0]['lat']:.2f}, lon={track[0]['lon']:.2f}")
    print(f"Last point: lat={track[-1]['lat']:.2f}, lon={track[-1]['lon']:.2f}")


def test_visibility():
    """Test visibility computation from ground station."""
    from tools.orekit_propagation_tool import compute_visibility
    
    # Ottobrunn coordinates
    result = compute_visibility(
        ISS_TLE1, ISS_TLE2,
        ground_lat=48.052453328999675,
        ground_lon=11.654863033454957,
        min_elevation_deg=10,
        duration_hours=24
    )
    
    assert result.get('status') == 'success', f"Error: {result.get('error')}"
    print(f"Visibility passes from Ottobrunn: {len(result['passes'])}")
    for p in result['passes'][:3]:
        print(f"  Pass: {p['duration_sec']/60:.1f} min, max el: {p['max_elevation_deg']:.1f} deg")


async def test_calculate_passes_flexible():
    """Test flexible calculate_passes with different input methods."""
    from tools.satellite_data_tool import calculate_passes
    from datetime import datetime, timezone
    
    print("\n=== Testing Flexible Overpass Calculation ===")
    
    # Test 1: Direct TLE input
    print("\n1. Testing with direct TLE input...")
    result1 = await calculate_passes({
        'action': 'calculate_passes',
        'tle_line1': ISS_TLE1,
        'tle_line2': ISS_TLE2,
        'location': 'ottobrunn',
        'hours_ahead': 24,
        'min_elevation': 10
    })
    assert result1.get('status') == 'success', f"Error: {result1.get('error')}"
    assert result1.get('input_type') == 'tle'
    print(f"   ✓ TLE input: {len(result1.get('passes', []))} passes found")
    
    # Test 2: Orbital elements input
    print("\n2. Testing with orbital elements input...")
    result2 = await calculate_passes({
        'action': 'calculate_passes',
        'semi_major_axis_km': 6778.0,
        'eccentricity': 0.0007,
        'inclination_deg': 51.64,
        'raan_deg': 100.0,
        'arg_perigee_deg': 90.0,
        'true_anomaly_deg': 0.0,
        'epoch': '2024-01-15T12:00:00Z',
        'ground_lat': 48.052453328999675,
        'ground_lon': 11.654863033454957,
        'hours_ahead': 24,
        'min_elevation': 10
    })
    assert result2.get('status') == 'success', f"Error: {result2.get('error')}"
    assert result2.get('input_type') == 'orbital_elements'
    print(f"   ✓ Orbital elements input: {len(result2.get('passes', []))} passes found")
    
    # Test 3: Ground station as dict
    print("\n3. Testing with ground station dict...")
    result3 = await calculate_passes({
        'action': 'calculate_passes',
        'tle_line1': ISS_TLE1,
        'tle_line2': ISS_TLE2,
        'ground_station': {'lat': 48.052453328999675, 'lon': 11.654863033454957},
        'hours_ahead': 24,
        'min_elevation': 10
    })
    assert result3.get('status') == 'success', f"Error: {result3.get('error')}"
    print(f"   ✓ Ground station dict: {len(result3.get('passes', []))} passes found")
    
    print("\n=== All flexibility tests passed! ===")


async def test_execute_function():
    """Test the async execute function."""
    from tools.orekit_propagation_tool import execute as prop_execute
    
    result = await prop_execute({
        'action': 'propagate',
        'tle_line1': ISS_TLE1,
        'tle_line2': ISS_TLE2
    })
    print(f"Execute propagate: {result.get('ground_track', {})}")
    assert 'error' not in result


if __name__ == "__main__":
    print("=" * 70)
    print("Testing High-Fidelity Orbital Mechanics Tools")
    print("=" * 70)
    
    tests = [
        ("TLE Propagation", test_propagate_tle),
        ("State Conversion", test_state_conversion),
        ("Hohmann Transfer", test_hohmann_transfer),
        ("Bi-elliptic Transfer", test_bielliptic_transfer),
        ("Ground Track", test_ground_track),
        ("Visibility", test_visibility),
        ("Flexible Overpass Calculation", lambda: asyncio.run(test_calculate_passes_flexible())),
        ("Execute Function", lambda: asyncio.run(test_execute_function())),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            test_func()
            print(f"PASSED: {name}")
            passed += 1
        except Exception as e:
            print(f"FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

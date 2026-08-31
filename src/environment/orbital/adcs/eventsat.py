"""EventSat mission configuration.

Concrete sensor and actuator instances for the EventSat 6U CubeSat. 
The the class configuration is in the configs.py file.
Here the exect sensor/actuator values are supplied.

By adding/edditing instances, the whole
ADCS sim can be used for different CubeSat missions, 
with no change to the simulation code elsewhere.
"""

import numpy as np

from datetime import datetime, timezone

from src.environment.orbital.adcs.configs import (
    ActuatorSuite,
    CoarseSunSensorConfig,
    EarthHorizonConfig,
    FineSunSensorConfig,
    MagnetometerConfig,
    RateGyroConfig,
    MagnetorquerConfig,
    ReactionWheelConfig,
    SensorSuite,
    SatelliteConfig,
    OrbitConfig,
    SimulationConfig,
)


# -----------------------------------------------------------------------------
# Magnetometers: CubeMag Deployable (2 sensors) + CubeMag Compact
# -----------------------------------------------------------------------------
_MAG_NOISE_1SIGMA_DEPLOYABLE = 50e-9 / 3.0 # CubeMag PD, p.11
_MAG_NOISE_1SIGMA_COMPACT = 120e-9 / 3.0 # CubeMag PD, p.11
_MAG_UPDATE_RATE_HZ = 5.0 # CubeMag PD, p.11
_MAG_MEASUREMENT_RANGE = 8e-4 # CubeMag PD, p.11
_MAG_DEPLOYABLE_TO_SENSOR = np.eye(3)
_MAG_COMPACT_TO_SENSOR = np.eye(3)
_magnetometers = [ 
    MagnetometerConfig(
        name="deployable_primary",
        body_to_sensor=_MAG_DEPLOYABLE_TO_SENSOR, # PLACEHOLDER
        noise_std=np.full(3, _MAG_NOISE_1SIGMA_DEPLOYABLE), # CubeMag PD, p.11
        bias=np.full(3, 100e-9), # PLACEHOLDER
        update_rate_hz=_MAG_UPDATE_RATE_HZ,
        max_field=_MAG_MEASUREMENT_RANGE,

    ),

    MagnetometerConfig(
        name="deployable_secondary",
        body_to_sensor=_MAG_DEPLOYABLE_TO_SENSOR, # PLACEHOLDER
        noise_std=np.full(3, _MAG_NOISE_1SIGMA_DEPLOYABLE), # CubeMag PD, p.11
        bias=np.full(3, 200e-9), # PLACEHOLDER
        update_rate_hz=_MAG_UPDATE_RATE_HZ,
        max_field=_MAG_MEASUREMENT_RANGE,
    
    ),
    
    MagnetometerConfig(
        name="compact",
        body_to_sensor=_MAG_COMPACT_TO_SENSOR, # PLACEHOLDER
        noise_std=np.full(3, _MAG_NOISE_1SIGMA_COMPACT), # CubeMag PD, p.11
        bias=np.full(3, 300e-9), # PLACEHOLDER
        update_rate_hz=_MAG_UPDATE_RATE_HZ,
        max_field=_MAG_MEASUREMENT_RANGE,
    ),
]

# -----------------------------------------------------------------------------
# Fine sun sensors: 2x CubeSense
# -----------------------------------------------------------------------------    

_fine_sun_sensors = [
    FineSunSensorConfig(
        name="fss0",
        body_to_sensor=np.eye(3), # PLACEHOLDER
        fov_half_angle=np.deg2rad(90), # ICD 3.1.2 Fig. 3
        incident_angle_noise_std=np.deg2rad(0.2/2), # PD p.10 gives 0.2°/axis at 2σ
        bias_std=np.deg2rad(0.05), # PLACEHOLDER
        max_slew_rate=np.deg2rad(70), # PD p.10
    ),
    FineSunSensorConfig(
        name="fss1",
        body_to_sensor=np.eye(3), # PLACEHOLDER
        fov_half_angle=np.deg2rad(90), # ICD 3.1.2 Fig. 3
        incident_angle_noise_std=np.deg2rad(0.2/2), # PD p.10 gives 0.2°/axis at 2σ
        bias_std=np.deg2rad(0.05), # PLACEHOLDER(PD gives ≤0.05°/axis as a worst case and not a σ, so reading it as 1σ likely overstates the bias)
        max_slew_rate=np.deg2rad(70), # PD p.10
    ),
]

# -----------------------------------------------------------------------------
# Coarse sun sensor array: 10 photodiodes
# -----------------------------------------------------------------------------
# Exact Position of diodes is still unknown!

_S3 = 1.0 / np.sqrt(3.0)
_coarse_sun_sensor = CoarseSunSensorConfig(
    name="css_array",
    normals=np.array([
    [ 1.0,  0.0,  0.0],      # six face normals
    [-1.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0],
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [ 0.0,  0.0, -1.0],
    [ _S3,  _S3,  _S3],      # four tetrahedral corners
    [-_S3, -_S3,  _S3],
    [-_S3,  _S3, -_S3],
    [ _S3, -_S3, -_S3],
    ])
)    


# -----------------------------------------------------------------------------
# Earth horizon sensor
# -----------------------------------------------------------------------------
_earth_horizon_sensor = EarthHorizonConfig(
    name="hss0",
    body_to_sensor=np.eye(3),
    fov_half_angle=np.deg2rad(60.0),
    noise_std=np.deg2rad(0.2),
)

# -----------------------------------------------------------------------------
# Gyro !!! ALL VALUES HERE ARE PLACEHOLDERS
# -----------------------------------------------------------------------------

_gyr0 = RateGyroConfig(
    name="GYR0",
    body_to_sensor=np.eye(3),                              
    arw=np.deg2rad(0.15) / 60.0,                           
    rrw=np.deg2rad(10.0) / 3600.0 / np.sqrt(3600.0),       
    bias_initial_std=np.deg2rad(0.5), 
    update_rate_hz=5.0,  # PLACEHOLDER                     
    max_rate=np.deg2rad(400.0),                            
)

# only used as back up, so it will not be included in the Sensor Suite
_gyr1 = RateGyroConfig(
    name="GYR1",
    body_to_sensor=np.eye(3),                              
    arw=np.deg2rad(0.30) / 60.0,
    rrw=np.deg2rad(20.0) / 3600.0 / np.sqrt(3600.0),
    bias_initial_std=np.deg2rad(1.0),
    update_rate_hz=5.0,  # PLACEHOLDER
    max_rate=np.deg2rad(400.0),
)

# -----------------------------------------------------------------------------
# Sensor Suite
# -----------------------------------------------------------------------------

sensors = SensorSuite(
    magnetometers=_magnetometers,
    fine_sun_sensors=_fine_sun_sensors,
    coarse_sun_sensor=_coarse_sun_sensor,
    earth_horizon_sensor=_earth_horizon_sensor,
    star_trackers=[], rate_gyros=[_gyr0],
)


# -----------------------------------------------------------------------------
# Reaction wheels: CubeWheel CW0057 Pyramid (4 wheels)
# -----------------------------------------------------------------------------
"""
Spin-axis unit vectors, PYRAMID local frame (CubeWheel ICD p.18, 22;
CubeADCS ICD p.100). Skew 26.57 deg = arctan(1/2) from the pyramid base

!!! PYRAMID -> BODY MOUNTING TRANSFORM IS UNKNOWN !!!
CMO section 7.5.3.2 (p.42) - No information
PYRAMID_TO_BODY = identity for the moment
"""

_S5 = 1.0 / np.sqrt(5.0)
_wheel_axes_pyramid = np.array(
    [
        [ 2.0 * _S5,  0.0,        _S5],   
        [ 0.0,        2.0 * _S5,  _S5],   
        [-2.0 * _S5,  0.0,        _S5],   
        [ 0.0,       -2.0 * _S5,  _S5],   
    ]
)

PYRAMID_TO_BODY = np.eye(3)  # PLACEHOLDER

_WHEEL_INERTIA = 9.51e-6      # kg*m^2, 9510 g*mm^2  (CubeWheel ICD p.16)
_WHEEL_MAX_TORQUE = 4.0e-3    # N*m     (CMO p.23, CW0057 Pyramid)
_WHEEL_MAX_MOMENTUM = 5.7e-3  # N*m*s   (CubeWheel PD p.11)

_reaction_wheels = [
    ReactionWheelConfig(
        name=f"rwl{i}",
        spin_axis_body=PYRAMID_TO_BODY @ _wheel_axes_pyramid[i],
        max_torque=_WHEEL_MAX_TORQUE,
        max_momentum=_WHEEL_MAX_MOMENTUM,
        wheel_inertia=_WHEEL_INERTIA,
    )
    for i in range(4)
]

# -----------------------------------------------------------------------------
# Magnetorquers: 3x CubeTorquer CR0006, one per SBC axis (CMO p.40).
# -----------------------------------------------------------------------------

_MTQ_MAX_DIPOLE = 0.6  # A*m^2 (CMO p.23; CubeTorquer PD p.10; CubeADCS ICD p.43)

_magnetorquers = [
    MagnetorquerConfig(name="mtq0", axis_body=np.array([1.0, 0.0, 0.0]),
                       max_dipole=_MTQ_MAX_DIPOLE),
    MagnetorquerConfig(name="mtq1", axis_body=np.array([0.0, 1.0, 0.0]),
                       max_dipole=_MTQ_MAX_DIPOLE),
    MagnetorquerConfig(name="mtq2", axis_body=np.array([0.0, 0.0, 1.0]),
                       max_dipole=_MTQ_MAX_DIPOLE),
]

# -----------------------------------------------------------------------------
# Actuator Suite
# -----------------------------------------------------------------------------

actuators = ActuatorSuite(
    reaction_wheels=_reaction_wheels,
    magnetorquers=_magnetorquers,
)

# -----------------------------------------------------------------------------
# Satellite Config
# -----------------------------------------------------------------------------

_inertia_full = np.array([
    [ 0.06887448665, -0.00085492255,  0.01973127957],
    [-0.00085492255,  0.09684835523, -0.00104368190],
    [ 0.01973127957, -0.00104368190,  0.04289885960],
])

_wheel_axes = np.column_stack([w.spin_axis_body for w in _reaction_wheels])   # (3,4)
_wheel_inertia = np.array([w.wheel_inertia for w in _reaction_wheels])        # (4,)

satellite = SatelliteConfig(
    name="EventSat",
    mass=8.42508,                                  # kg   (CMO p.15)
    inertia_full=_inertia_full,                    #      (CMO p.14-15)
    com_offset=np.array([0.00017, 0.00126, -0.01371]),   # m (CMO p.15)
    cop_offset=np.zeros(3),                        # m, CoP = geometric origin (CMO p.15)
    wheel_axes=_wheel_axes,
    wheel_inertia=_wheel_inertia,
    dimensions=np.array([0.3665, 0.1005, 0.227]),  # Double check if this is correct (CMO p.16 gives "366.5 x 100.5 x 227 mm" without stating the axis order) 
    drag_coeff=2.2,                                # literature default (Cook, G. E. (1965))
    reflectivity=1.3,                              # literature default (Paluszek Eq. 8.35)
    residual_dipole=np.array([0.02, 0.02, 0.02])   # UNSOURCED placeholder
)

# -----------------------------------------------------------------------------
# Orbit Config
# -----------------------------------------------------------------------------

orbit = OrbitConfig(
    epoch=datetime(2024, 12, 31, 10, 30, 0, tzinfo=timezone.utc), # For IGRF-13 validity
    altitude_km=450.0,         # CMO p.14 - not yet confirmed
    eccentricity=0.0,          # Assumption
    inclination_deg=97.4,      # CMO p.14
    raan_deg=0.0,              # propagator.configure() derives RAAN from LTAN and this value is unused
    arg_perigee_deg=0.0,       # undefined at e=0; any value is equivalent
    true_anomaly_deg=0.0,
    ltan_hours=10.5,           # no info in CMO -ask!
    propagator_type="j2",      # Eckstein-Hechler
)

# -----------------------------------------------------------------------------
# Mission Config
# -----------------------------------------------------------------------------
# Will be later moved to a different file, maybe
# 0.2 s = the 5 Hz flight control loop (CubeADCS ICD p.42; CubeMag PD p.11).
sim = SimulationConfig( 
    step_s = 0.2,   
)
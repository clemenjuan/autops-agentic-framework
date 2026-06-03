"""EventSat mission configuration.

Concrete sensor and actuator instances for the EventSat 6U CubeSat. 
The the class configuration is in the configs.py file.
Here the exect sensor/actuator values are supplied.

By adding/edditing instances, the whole
ADCS sim can be used for different CubeSat missions, 
with no change to the simulation code elsewhere.

All numeric values in are placeholders (for now), 
but structurally correct.
"""

import numpy as np

from src.environment.orbital.adcs.configs import (
    ActuatorSuite,
    CoarseSunSensorConfig,
    EarthHorizonConfig,
    FineSunSensorConfig,
    MagnetometerConfig,
    MagnetorquerConfig,
    ReactionWheelConfig,
    SensorSuite,
)


# -----------------------------------------------------------------------------
# Magnetometers: CubeMag Deployable (2 sensors) + CubeMag Compact
# -----------------------------------------------------------------------------
_magnetometers = [
    MagnetometerConfig(
        name="deployable",
        body_to_sensor=np.eye(3),
        noise_std=np.array([50e-9, 50e-9, 50e-9]),
        bias=np.array([100e-9, 100e-9, 100e-9]),
    ),
    #think about what would be the best way to add the second deployable sensor
    MagnetometerConfig(
        name="compact",
        body_to_sensor=np.eye(3),
        noise_std=np.array([100e-9, 100e-9, 100e-9]),
        bias=np.array([200e-9, 200e-9, 200e-9]),
    ),
]

# -----------------------------------------------------------------------------
# Fine sun sensors: 2x CubeSense
# -----------------------------------------------------------------------------
_fine_sun_sensors = [
    FineSunSensorConfig(
        name="fss_a",
        body_to_sensor=np.eye(3),
        fov_half_angle=np.deg2rad(60.0),
        noise_std=np.deg2rad(0.1),
    ),
    FineSunSensorConfig(
        name="fss_b",
        body_to_sensor=np.eye(3),
        fov_half_angle=np.deg2rad(60.0),
        noise_std=np.deg2rad(0.1),
    ),
]

# -----------------------------------------------------------------------------
# Coarse sun sensor array: 10 photodiodes
# -----------------------------------------------------------------------------
_coarse_sun_sensor = CoarseSunSensorConfig(
    name="css_array",
    normals=np.array(
        [
            [ 1.0,  0.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0,  1.0],
            [ 0.0,  0.0, -1.0],
            [ 1.0,  1.0,  0.0],
            [-1.0,  1.0,  0.0],
            [ 1.0, -1.0,  0.0],
            [-1.0, -1.0,  0.0],
        ]
    ),
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
# Sensor Suite
# -----------------------------------------------------------------------------

sensors = SensorSuite(
    magnetometers=_magnetometers,
    fine_sun_sensors=_fine_sun_sensors,
    coarse_sun_sensor=_coarse_sun_sensor,
    earth_horizon_sensor=_earth_horizon_sensor,
    star_trackers=[]
)


# -----------------------------------------------------------------------------
# Reaction wheels: four-wheel pyramid
# -----------------------------------------------------------------------------

_p = 1.0 / np.sqrt(3.0)
_reaction_wheels = [
    ReactionWheelConfig(
        name="wheel_1",
        spin_axis_body=np.array([ 1.0,  1.0, 1.0]) * _p,
        max_torque=1.0e-3,
        max_momentum=1.0e-2,
        wheel_inertia=1.0e-5,
    ),
    ReactionWheelConfig(
        name="wheel_2",
        spin_axis_body=np.array([-1.0,  1.0, 1.0]) * _p,
        max_torque=1.0e-3,
        max_momentum=1.0e-2,
        wheel_inertia=1.0e-5,
    ),
    ReactionWheelConfig(
        name="wheel_3",
        spin_axis_body=np.array([-1.0, -1.0, 1.0]) * _p,
        max_torque=1.0e-3,
        max_momentum=1.0e-2,
        wheel_inertia=1.0e-5,
    ),
    ReactionWheelConfig(
        name="wheel_4",
        spin_axis_body=np.array([ 1.0, -1.0, 1.0]) * _p,
        max_torque=1.0e-3,
        max_momentum=1.0e-2,
        wheel_inertia=1.0e-5,
    )
]

# -----------------------------------------------------------------------------
# Magnetorquers: three rods, one per body axis
# -----------------------------------------------------------------------------
_magnetorquers = [
    MagnetorquerConfig(
        name="mtq_x",
        axis_body=np.array([1.0, 0.0, 0.0]), 
        max_dipole=0.13
        ),
    MagnetorquerConfig(
        name="mtq_y", 
        axis_body=np.array([0.0, 1.0, 0.0]), 
        max_dipole=0.13
        ),
    MagnetorquerConfig(
        name="mtq_z", 
        axis_body=np.array([0.0, 0.0, 1.0]), 
        max_dipole=0.13
        )
]

# -----------------------------------------------------------------------------
# Actuator Suite
# -----------------------------------------------------------------------------

actuators = ActuatorSuite(
    reaction_wheels=_reaction_wheels,
    magnetorquers=_magnetorquers,
)
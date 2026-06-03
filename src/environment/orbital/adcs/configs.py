"""EventSat ADCS configuration.

Sensor and actuator set up as dataclasses, to enable 
configurable design trough the eventsat.py file that contains 
the specific instances for the relevant configuration.

Here the parameters for each sensor/actuator are defined.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


# =============================================================================
# Sensor classes set up
# =============================================================================
@dataclass
class MagnetometerConfig:
    """Magnetometer configuration.

    Attributes:
        name: Human-readable identifier.
        body_to_sensor: 3x3 rotation matrix, body frame to sensor frame,
            describing how the unit is mounted.
        noise_std: Per-axis measurement noise standard deviation (1-sigma)
            [T], shape (3,).
        bias: Per-axis constant bias [T], shape (3,).
    """

    name: str
    body_to_sensor: np.ndarray
    noise_std: np.ndarray
    bias: np.ndarray


@dataclass
class FineSunSensorConfig:
    """Fine sun sensor configuration.

    Attributes:
        name: Human-readable identifier.
        body_to_sensor: 3x3 rotation matrix, body frame to sensor frame.
        fov_half_angle: Half-angle of the conical field of view [rad]. The
            sun is only seen when within this cone of the boresight.
        noise_std: Angular measurement noise standard deviation (1-sigma)
            [rad].
    """

    name: str
    body_to_sensor: np.ndarray
    fov_half_angle: float
    noise_std: float


@dataclass
class CoarseSunSensorConfig:
    """Coarse sun sensor array configuration.

    Attributes:
        name: Human-readable identifier.
        normals: Outward unit normals of each photodiode cell in the body
            frame, shape (n_cells, 3). EventSat has ten cells.
    """

    name: str
    normals: np.ndarray


@dataclass
class EarthHorizonConfig:
    """Earth horizon (nadir) sensor configuration.

    Attributes:
        name: Human-readable identifier.
        body_to_sensor: 3x3 rotation matrix, body frame to sensor frame.
        fov_half_angle: Half-angle of the conical field of view [rad]; a
            nadir lock is only available when nadir falls within it.
        noise_std: Angular measurement noise standard deviation (1-sigma)
            [rad].
    """

    name: str
    body_to_sensor: np.ndarray
    fov_half_angle: float
    noise_std: float

@dataclass
class StarTrackerConfig:
    """Star tracker configuration.

    Not included as an instrument on EventSat.

    Star tracker outputs a full attitude estimate
    (a quaternion), not a single direction and can be blinded
    by bright bodies.

    Attributes:
        name: Human-readable identifier.
        body_to_sensor: 3x3 rotation matrix, body frame to sensor frame.
        fov_half_angle: Half-angle of the conical field of view [rad].
        noise_std: Per-axis attitude noise standard deviation (1-sigma) [rad],
            shape (3,). Typically anisotropic.
        sun_exclusion_angle: Minimum allowed angle between boresight and sun
            [rad]; within this cone the tracker is blinded and returns no
            solution.
    """

    name: str
    body_to_sensor: np.ndarray
    fov_half_angle: float
    noise_std: np.ndarray
    sun_exclusion_angle: float

# =============================================================================
# Actuator classes set up
# =============================================================================
@dataclass
class ReactionWheelConfig:
    """Per-wheel configuration.

    Attributes:
        name: Human-readable identifier.
        spin_axis_body: Unit vector along the wheel spin axis in the body
            frame, shape (3,).
        max_torque: Maximum commandable torque magnitude [N·m].
        max_momentum: Angular momentum at saturation [N·m·s].
        wheel_inertia: Wheel inertia about its spin axis [kg·m²].
    """

    name: str
    spin_axis_body: np.ndarray
    max_torque: float
    max_momentum: float
    wheel_inertia: float


@dataclass
class MagnetorquerConfig:
    """Per-rod configuration.

    Attributes:
        name: Human-readable identifier.
        axis_body: Unit vector along the rod axis in the body frame,
            shape (3,).
        max_dipole: Maximum commandable magnetic dipole moment [A·m²].
    """

    name: str
    axis_body: np.ndarray
    max_dipole: float


# =============================================================================
# Suite containers
# =============================================================================
"""A container that bundles all of the sensors/actuators into a single object,
so the rest of the code can pass it around as one unit.
"""

@dataclass
class SensorSuite:
    """All sensors equipped on the satellite.

    Attributes:
        magnetometers: Magnetometer configurations.
        fine_sun_sensors: Fine sun sensor configurations.
        coarse_sun_sensor: The single coarse sun sensor array.
        earth_horizon_sensor: The single earth horizon sensor.
    """

    magnetometers: List[MagnetometerConfig]
    fine_sun_sensors: List[FineSunSensorConfig]
    coarse_sun_sensor: CoarseSunSensorConfig
    earth_horizon_sensor: EarthHorizonConfig
    star_trackers: List[StarTrackerConfig]


@dataclass
class ActuatorSuite:
    """All actuators equipped on the satellite.

    Attributes:
        reaction_wheels: Reaction wheel configurations.
        magnetorquers: Magnetorquer configurations.
    """

    reaction_wheels: List[ReactionWheelConfig]
    magnetorquers: List[MagnetorquerConfig]
"""ADCS configuration.

Sensor and actuator set up as dataclasses, to enable 
configurable design trough the eventsat.py file that contains
the specific instances for the relevant configuration.

Here the parameters for each sensor/actuator are defined.
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np
from datetime import datetime


# =============================================================================
# Sensor classes set up
# =============================================================================
@dataclass
class MagnetometerConfig:
    """Magnetometer configuration.

    Attributes:
        name: Readable identifier.
        body_to_sensor: 3x3 rotation matrix, body frame to sensor frame,
            describing how the unit is mounted.
        noise_std: Per-axis measurement noise standard deviation (1-sigma)
            [T], shape (3,).
        bias: Per-axis constant bias [T], shape (3,).
        update_rate_hz: Rate at which the unit produces new measurements [Hz].
        max_field: Per-axis saturation limit [T].
    """

    name: str
    body_to_sensor: np.ndarray
    noise_std: np.ndarray
    bias: np.ndarray
    update_rate_hz: float
    max_field: float


@dataclass
class FineSunSensorConfig:
    """Fine sun sensor configuration.

    Models a CMOS imager simplified to a pinhole model (High-Accuracy Image 
    Centroiding Algorithm for CMOS-Based Digital Sun Sensors (Kim et al., Eq. 2)),
    reporting the sun direction as two image-plane incident angles (alpha,
    beta) which are converted to a unit vector in the sensor frame.

    Attributes:
        name: Human-readable identifier.
        body_to_sensor: 3x3 rotation matrix, body frame to sensor frame.
        fov_half_angle: Half-angle of the conical detection region [rad].
            must be between 0 and pi/2 rad, due to the pinhole model simplification
        incident_angle_noise_std: Noise standard deviation (1-SIGMA) on each
            incident angle [rad]. 
        bias_std: Standard deviation [rad] for drawing the per-unit turn-on
            offset on each incident angle. This is the spread, not the bias.
        max_slew_rate: Body rate above which the unit reports no detection
            [rad/s]. Hard cutoff, CubeSun PD p.10.
    """

    name: str
    body_to_sensor: np.ndarray
    fov_half_angle: float
    incident_angle_noise_std: float
    bias_std: float
    max_slew_rate: float

    def __post_init__(self) -> None:
        if self.fov_half_angle > np.pi/2 or self.fov_half_angle <= 0:
            raise ValueError(f"fov_half_angle must not be bigger than pi/2 rad (pinhole model) , got {self.fov_half_angle}")


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

@dataclass(frozen=True)
class RateGyroConfig:
    """MEMS rate gyro with a Farrenkopf error model. (Farrenkopf, JGC 1(4))

    Attributes:
        name: Instance identifier.
        body_to_sensor: Body to sensor frame rotation, shape (3, 3).
        arw: Angle random walk [rad/sqrt(s)] - white noise on the rate.
        rrw: Rate random walk [rad/s^1.5] - drives the bias drift.
        bias_initial_std: Turn-on bias repeatability [rad/s], one sigma.
        update_rate_hz: Rate at which the unit produces new measurements [Hz].
        max_rate: Measurement saturation limit [rad/s].
    """

    name: str
    body_to_sensor: np.ndarray
    arw: float
    rrw: float
    bias_initial_std: float
    update_rate_hz: float
    max_rate: float

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
        friction_viscous: Viscous friction coefficient sigma_v
            [N·m·s/rad], the speed-proportional term of Paluszek Eq. 10.14
        friction_coulomb: Coulomb friction torque f_c [N·m], constant in
            magnitude and opposing motion (Paluszek Eq. 10.14).
    """

    name: str
    spin_axis_body: np.ndarray
    max_torque: float
    max_momentum: float
    wheel_inertia: float
    friction_viscous: float = 0.0
    friction_coulomb: float = 0.0


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
        star_trackers: Star Tracker configs.
        rate_gyros: Rate Gyros configs.
    """

    magnetometers: List[MagnetometerConfig]
    fine_sun_sensors: List[FineSunSensorConfig]
    coarse_sun_sensor: CoarseSunSensorConfig
    earth_horizon_sensor: EarthHorizonConfig
    star_trackers: List[StarTrackerConfig]
    rate_gyros: List[RateGyroConfig]


@dataclass
class ActuatorSuite:
    """All actuators equipped on the satellite.

    Attributes:
        reaction_wheels: Reaction wheel configurations.
        magnetorquers: Magnetorquer configurations.
    """

    reaction_wheels: List[ReactionWheelConfig]
    magnetorquers: List[MagnetorquerConfig]

# =============================================================================
# Satellite Config
# =============================================================================

@dataclass(frozen=True)
class SatelliteConfig:
    """Physical parameters of one satellite, body frame, about the COM.

    ``inertia_full`` is the assembled-satellite inertia with the wheels treated
    as locked (the CMO/CAD tensor). 
    The reduced inertia actually used in the gyrostat equations, ``inertia = inertia_full - W·J_w·Wᵀ``,
    and its inverse are derived from it and cached here so ``integrate``(in dynamics.py) never recomputes them.
    """

    name: str

    # mass properties (about COM, body frame)
    mass: float                  # kg
    inertia_full: np.ndarray     # (3,3) kg·m², wheels locked
    com_offset: np.ndarray       # (3,) m
    cop_offset: np.ndarray       # (3,) m, center of pressure

    # reaction-wheel coupling geometry
    wheel_axes: np.ndarray       # (3,4) W, spin-axis unit vectors as columns
    wheel_inertia: np.ndarray    # (4,) kg·m², per-wheel axial inertia

    # disturbance-model parameters
    dimensions: np.ndarray       # (3,) m, box side lengths (projected area)
    drag_coeff: float            # Cd
    reflectivity: float          # Cr
    residual_dipole: np.ndarray  # (3,) A·m²

    # derived / cached (not constructor arguments)
    inertia: np.ndarray = field(init=False)
    inertia_inv: np.ndarray = field(init=False)
    wheel_inertia_mat: np.ndarray = field(init=False)
    wheel_inertia_inv: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        Jw = np.diag(self.wheel_inertia)              # (4,4)
        W = self.wheel_axes                           # (3,4)
        reduced = self.inertia_full - W @ Jw @ W.T
        object.__setattr__(self, "wheel_inertia_mat", Jw)
        object.__setattr__(self, "wheel_inertia_inv", np.diag(1.0 / self.wheel_inertia))
        object.__setattr__(self, "inertia", reduced)
        object.__setattr__(self, "inertia_inv", np.linalg.inv(reduced))

# =============================================================================
# Orbit Config
# =============================================================================

@dataclass(frozen=True)
class OrbitConfig:
    """Orbit definition and propagator choice (mission-agnostic).

    propagator.py translates these into Orekit objects.

    Attributes:
        epoch: UTC epoch; the t=0 reference for the simulation clock.
        altitude_km: Mean altitude above the WGS84 equatorial radius [km].
        eccentricity: Orbit eccentricity.
        inclination_deg: Inclination [deg].
        raan_deg: Right ascension of the ascending node [deg].
        arg_perigee_deg: Argument of perigee [deg].
        true_anomaly_deg: True anomaly at epoch [deg].
        ltan_hours: Local time of the ascending node [hours, 0-24]
        propagator_type: "j2" (Eckstein-Hechler, models J2 RAAN precession) or
            "keplerian" (two-body; no precession — not sun-synchronous).
    """

    epoch: datetime
    altitude_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    true_anomaly_deg: float
    ltan_hours: Optional[float] = None   # if set, RAAN is derived from this and raan_deg is ignored
    propagator_type: str = "j2"

# =============================================================================
# Simulation Config
# =============================================================================

@dataclass(frozen=True)
class SimulationConfig:
    """Simulation-run parameters.

    Attributes:
        step_s: Simulation timestep [s].
        seed: Base seed for sensor noise, gyro bias, turn-on bias.
    """

    step_s: float
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.step_s <= 0.0:
            raise ValueError(f"step_s must be positive, got {self.step_s}")

    def resolved(self) -> "SimulationConfig":
        """Copy with a concrete seed. Call once in run(); log the result."""
        if self.seed is not None:
            return self
        return replace(self, seed=int(np.random.SeedSequence().entropy) & (2**63 - 1))
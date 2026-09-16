"""ADCS configuration.

Sensor and actuator set up as dataclasses, to enable 
configurable design trough the eventsat.py file that contains
the specific instances for the relevant configuration.

Here the parameters for each sensor/actuator are defined.
"""

from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Dict, List, Optional

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

    A set of photodiodes with fixed body-frame normals. Each cell returns a
    scalar current, so a single reading is consistent with a cone of sun
    directions.

    Attributes:
        name: Human-readable identifier.
        normals: Outward unit normals of each photodiode cell in the BODY
            frame, shape (n_cells, 3). EventSat has ten cells.
        full_scale_current: Cell current [A] at normal incidence at 1 AU with
            no albedo. A scale factor at a reference condition.
        noise_std: Readout noise standard deviation (1-SIGMA) [A]. Estimation.
    """

    name: str
    normals: np.ndarray
    full_scale_current: float
    noise_std: float


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
    fov_half_angle_horizontal: float
    fov_half_angle_vertical: float
    horizon_roll_half_angle: float
    noise_std: float
    max_slew_rate: float

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
        max_speed: Angular speed at saturation [rad/s].
        wheel_inertia: Wheel inertia about its spin axis [kg·m²].
        friction_viscous: Viscous friction coefficient sigma_v
            [N·m·s/rad], the speed-proportional term of Paluszek Eq. 10.14
        friction_coulomb: Coulomb friction torque f_c [N·m], constant in
            magnitude and opposing motion (Paluszek Eq. 10.14).
        power_fit_square_param: Quadratic coefficient of the power fit
            [W·s²/rad²]. This coefficient and the folowing two are a 
            single least-squares solution over 49 points 
            digitised from the ADCS ICD p.45 steady-state trace at 16V.
        power_fit_lin_param: Linear coefficient of the same fit [W·s/rad].
        power_fit_const_param: Constant term of the same fit [W].
        mechanical_power_scale: empirical scaling factor for the physical
            torque term that is added to the zero torque fit.
        momentum_at_max_speed: Calculating the momentum of a single wheel
            at the maimum angular velocity [N*m*s].Derived as 
            wheel_inertia * max_speed; not a constructor
    """

    name: str
    spin_axis_body: np.ndarray
    max_torque: float
    max_speed: float
    wheel_inertia: float
    friction_viscous: float
    friction_coulomb: float

    # Power Model fit parameters: 
    # P_fit(omega) = 0.0836 + 5.007e-4*|omega| + 5.487e-7*omega^2
    power_fit_square_param: float
    power_fit_lin_param: float
    power_fit_const_param: float

    mechanical_power_scale: float

    # Derived:
    momentum_at_max_speed: float = field(init=False)

    def __post_init__(self) -> None:
        self.momentum_at_max_speed = self.max_speed * self.wheel_inertia


@dataclass
class MagnetorquerConfig:
    """Per-rod configuration.

    Attributes:
        name: Human-readable identifier.
        axis_body: Unit vector along the rod axis in the body frame,
            shape (3,).
        max_dipole: Maximum commandable magnetic dipole moment [A·m²].
        coil_resistance: Coil resistance of each rod [Ohm]
        magnetic_gain: Dipole per amp [A*m^2/A]       
        supply_voltage: Voltage supplied to each rod [V]     
        duty_max: Maximum fraction of the ADCS control loop during which the rod
            may be energised.
    """

    name: str
    axis_body: np.ndarray
    max_dipole: float
    coil_resistance: float
    magnetic_gain: float
    supply_voltage: float     
    duty_max: float             


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

# =============================================================================
# Mission Config
# =============================================================================

@dataclass(frozen=True)
class MissionConfig:
    """General Mission Config parameters.

    Attributes:
        type: Name of mission type being performed
        tolerance_deg: Pointing accuracy that counts as on target [deg].
        hold_time: Consecutive seconds inside the tolerance before a target
            counts as cleared. Stops a target being "cleared" by flying through
            it at rate. 
        max_steps: Steps before the episode is truncated.
    """
    type: str
    tolerance_deg: float
    hold_time: float
    max_steps: int

    @property
    def sigma(self) -> float:   # moved off AdcsEnvConfig
        return float(np.sin(np.deg2rad(self.tolerance_deg) / 2.0))

@dataclass(frozen=True)
class SlewSequenceConfig(MissionConfig):
    """Specific Mission config for a simple attitude tracking 
    mission where the satellite tracks a list of setpoints one after the other.

    Attributes:
        num_targets: Slew targets to visit in one episode; the mission is done
            once all of them have been cleared.
    """
    num_targets: int

@dataclass(frozen=True)
class TargetTrackConfig(MissionConfig):
    """Specific Mission config for target following
    mission where the satellite searches and then tracks multiple targets.

    Attributes:
        num_targets: Slew targets to visit in one episode; the mission is done
            once all of them have been cleared.
        fov_half_angle: Half-angle of the payload's conical field of view [rad].
            A target counts as visible while it lies inside this cone about the
            boresight.
        boresight_body: Unit vector along the camera boresight in the BODY
            frame, shape (3,).
        reference_body: Unit vector giving the camera's roll reference in the
            BODY frame, shape (3,); must not be parallel to boresight_body.
            Pointing the boresight somewhere fixes only two of the three
            attitude degrees of freedom, and the pointing error the reward sees
            is the full attitude error, roll included. So the remaining freedom
            has to be pinned to something physical rather than left arbitrary:
            this axis is aligned with the orbit normal.
    """
    num_targets: int
    fov_half_angle: float
    boresight_body: np.ndarray
    reference_body: np.ndarray

    
# =============================================================================
# RL environment config
# =============================================================================

@dataclass(frozen=True)
class AdcsEnvConfig:
    """Task and reward parameters for the RL wrapper around the ADCS sim.

    Separate from SimulationConfig, which describes the *numerics* of the
    simulation: this describes the *task* the agent is trained on. 

    Attributes:
        body_rate_thresh: Body rate above which the slew-rate penalty starts
            [rad/s].
        max_body_rate: Body rate to normailse all body rate observations 
        start_step: Step index the episode starts at; scales into SatState.t and
            so picks the point in the orbit.
        seed: Base seed for the episode. None draws a fresh one per reset.
        reward_weights: Reward magnitude per component, keyed by
            adcs_rewards.REWARD_COMPONENTS and EVENT_COMPONENTS. The dense
            components are already normalised by their own functions, so these
            set relative importance rather than scale.
    """

    body_rate_thresh: float
    max_body_rate: float
    start_step: int
    seed: Optional[int]
    reward_weights: Dict[str, float]
    mission: MissionConfig

    def with_overrides(self, overrides: Optional[Dict[str, Any]]) -> "AdcsEnvConfig":
        """Copy with the given fields replaced, taking them as a plain dict.

        For overrides that arrive as data -- RLlib's ``env_config``, or a sweep
        varying the task per trial. An override written literally in code wants
        ``dataclasses.replace`` instead.
        """
        if not overrides:
            return self
        if isinstance(overrides, AdcsEnvConfig):
            return overrides

        # Every field name this config accepts; Python builds this dict on any
        # dataclass.
        fields = self.__dataclass_fields__

        # Anything asked for that is not a field is a typo. Dropping it quietly
        # would train a different config than the one written down and report a
        # clean run. Nothing foreign lands here to be tolerated: RLlib's
        # EnvContext carries worker_index and the rest as attributes, not keys.
        unknown = sorted(set(overrides) - set(fields))
        if unknown:
            raise ValueError(
                f"Unknown {type(self).__name__} field(s): {', '.join(unknown)}. "
                f"Valid fields: {', '.join(sorted(fields))}."
            )

        resolved: Dict[str, Any] = {}
        for key, value in overrides.items():
            current = getattr(self, key)

            # A dict where a config object belongs: `asdict` is recursive, and
            # so is any JSON round-trip, so `mission` arrives flattened.
            # Rebuilding off the current value restores the type -- and the
            # concrete subclass, SlewSequenceConfig rather than the
            # MissionConfig the annotation names. Assigning the dict straight
            # through is checked by nothing here and fails much later, at the
            # first attribute access inside the mission.
            if is_dataclass(current) and isinstance(value, dict):
                value = replace(current, **value)

            resolved[key] = value

        return replace(self, **resolved)
    
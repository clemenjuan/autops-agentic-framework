"""ADCS sensor models.

The read functions simulate one sensor each: they take the true satellite
state and the true environment, apply that sensor's measurement model and return
the measurement. 

SensorMeasurements bundles one timestep's worth of these outputs for the
estimator.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import List, Optional
import numpy as np

from src.environment.orbital.adcs.configs import (
    CoarseSunSensorConfig,
    EarthHorizonConfig,
    FineSunSensorConfig,
    MagnetometerConfig,
    StarTrackerConfig,
    RateGyroConfig,
    SensorSuite,
)
from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.propagator import EnvironmentData
from src.environment.orbital.adcs.dynamics import dcm_eci_to_body
from src.environment.orbital.adcs.constants import EARTH_BOND_ALBEDO, R_EARTH

@dataclass
class SensorState:
    """Internal error states of the sensor suite that carry memory across steps.

    Attributes:
        gyro_bias: Rate gyro bias [rad/s], sensor frame, one roll per gyro,
            shape (n_gyros, 3).
        fss_bias: FSS bias [rad], one roll per sensor, shape(n_fss, 2) ((alpha, beta) incident angles),
        doesn't change across steps, 
    """

    gyro_bias: np.ndarray
    fss_bias: np.ndarray


def initial_sensor_state(
    sensors: SensorSuite, rng: np.random.Generator
) -> SensorState:
    """Draw the turn on sensor error states.

     Args:
        sensors: The sensor suite; sizes both bias arrays.
        rng: Supplies the turn-on draws.

    Returns:
        A SensorState with both bias arrays populated. Empty suites give
        (0, 3) and (0, 2) rather than (0,), so downstream indexing is
        uniform.
    """
    gyro_bias = np.array(
        [rng.normal(0.0, gyr.bias_initial_std, 3) for gyr in sensors.rate_gyros]
    ).reshape(len(sensors.rate_gyros), 3)
    fss_bias = np.array(
        [rng.normal(0.0, fss.bias_std, 2) for fss in sensors.fine_sun_sensors]
    ).reshape(len(sensors.fine_sun_sensors), 2)
    return SensorState(gyro_bias=gyro_bias, fss_bias=fss_bias)


def propagate_sensor_state(
    sensor_state: SensorState,
    sensors: SensorSuite,
    dt: float,
    rng: np.random.Generator,
) -> SensorState:
    """Advance the sensor error states by one timestep.
    """
    if not sensors.rate_gyros:
        return sensor_state
    rrw = np.array([g.rrw for g in sensors.rate_gyros])[:, None]
    bias = sensor_state.gyro_bias
    new_bias = bias + rrw * np.sqrt(dt) * rng.standard_normal(bias.shape)
    return replace(sensor_state, gyro_bias=new_bias)

def read_magnetometer(
    state: SatState, 
    env: EnvironmentData, 
    config: MagnetometerConfig, 
    rng: np.random.Generator
) -> np.ndarray:
    """Measured magnetic field for one magnetometer [T], shape (3,).

    Measurement model, after Alonso & Shuster (2002)

    Args:
        state: True satellite state; supplies the attitude quaternion.
        env: True environment; supplies the geomagnetic field in GCRF.
        config: This magnetometer's mounting, noise, bias and range.
        rng: Supplies the measurement noise.

    Returns:
        Measured field [T], shape (3,), sensor frame, clipped to the unit's
        measurement range
    """
    # True B field in sensor frame
    B_sensor = config.body_to_sensor @ dcm_eci_to_body(state.q_eci_body) @ env.b_field_eci 
    # Measured B field in sensor frame 
    B_meas = B_sensor + config.bias + rng.normal(0, config.noise_std, 3) # Zero mean, because we have the bias term
    B_meas = np.clip(B_meas, -config.max_field, config.max_field)
    return B_meas


def read_fine_sun_sensor(
    state: SatState, 
    env: EnvironmentData, 
    config: FineSunSensorConfig, 
    bias: np.ndarray, 
    rng: np.random.Generator
) -> Optional[np.ndarray]:
    """Measured sun direction for one fine sun sensor, unit (3,), sensor frame.

    Returns None when there is no measurement.

    The unit reports two image-plane incident angles which are converted to a
    vector. Noise and bias are applied to those angles, since that is where the
    sensor physically measures (Kim et al., "High-Accuracy Image Centroiding Algorithm
    for CMOS-Based Digital Sun Sensors", Eq. 2.)

        alpha = arctan2(y, z),   beta = arctan2(-x, z)

    ASSUMPTION 1: the axis pairing and the sign of beta follow the cited paper,
    not a CubeSpace document.

    ASSUMPTION 2: Eq. 2 is a pinhole model; the CubeSense uses a fisheye lens.
    Kannala & Brandt (2006) for the projection models a calibration would identify.

    Flat noise across the field of view is the PD's own specification - a
    single per-axis figure with no angular dependence. 
    Accuracy above the slew cutoff is not modelled (the degradation is documented
     but its shape is not).

    The slew gate uses total body rate. Blur depends on the
    component perpendicular to the sun line, thus this model is conservative.

    Args:
        state: True satellite state; supplies attitude and body rate.
        env: True environment; supplies the sun direction and eclipse flag.
        config: This unit's mounting, field of view, noise and cutoff.
        bias: This unit's (alpha, beta) offset [rad], shape (2,), from
            SensorState.fss_bias. Drawn at turn-on, constant thereafter.
        rng: Supplies the incident-angle noise.

    Returns:
        Unit sun direction in the sensor frame, shape (3,), or None.
    """
    incident_angle_noise = rng.normal(0, config.incident_angle_noise_std, 2)

    # Check for eclipse:
    if env.eclipse:
        return None

    # Sun vector in sensor frame:
    sun_vector_sen = config.body_to_sensor @ dcm_eci_to_body(state.q_eci_body) @ env.sun_vector_eci
    
    # Check if Sun is within FoV, by dot product with z_sen = [0,0,1]:
    if sun_vector_sen[2] < np.cos(config.fov_half_angle):
        return None

    # Check if slew rate is too large for a measurement (conservative):
    if np.linalg.norm(state.omega_body) > config.max_slew_rate:
        return None

    # True incident angles alpha and beta (pin hole model):
    alpha_true = np.arctan2(sun_vector_sen[1], sun_vector_sen[2])
    beta_true = np.arctan2(-sun_vector_sen[0], sun_vector_sen[2])

    # Measured incident angles alpha and beta:
    alpha_meas = alpha_true + incident_angle_noise[0] + bias[0]
    beta_meas = beta_true + incident_angle_noise[1] + bias[1]

    # Make sure to reject unphysical measurements:
    if np.abs(alpha_meas) >= np.pi/2 or np.abs(beta_meas) >= np.pi/2:
        return None

    # Measured sun vector in sensor frame 
    sun_vector_sen_meas = np.array([
        -np.sin(beta_meas) * np.cos(alpha_meas), # minus as in ref (assumption 1) 
        np.sin(alpha_meas) * np.cos(beta_meas), 
        np.cos(alpha_meas) * np.cos(beta_meas)
        ])

    sun_vector_sen_meas = sun_vector_sen_meas / np.linalg.norm(sun_vector_sen_meas)

    return sun_vector_sen_meas


def read_coarse_sun_sensor(
    state: SatState, env: EnvironmentData, config: CoarseSunSensorConfig, rng: np.random.Generator
) -> Optional[np.ndarray]:
    """Each photodiode returns a current for the coarse sun sensor array, shape (n_cells,).

    A cell's current is proportional to its projected area toward
    the source. (SLCD-61N8 datasheet, Lambertian response)

    Takes into account both light coming from the sun and light from 
    Earth's albedo. During eclipse it returns only noise.

    The view factor (F), is the fraction of a surface's hemispherical view that Earth fills:
    F = sin^2(rho) with sin(rho) = R_EARTH/r by definition, so
    F = (R_EARTH/r)^2 exactly - an identity, not a small-angle approximation.
    Without F the point-source form runs 1.15x hot, verified against numerical
    integration of a Lambertian sphere.

    a is Earth's Bond albedo, a planetary constant.

    Not modelled:
        Earth as an extended disc,
        Surface reflectivity variation,
        Mounting offsets and obscuration,
        Diode Saturation,
        Dark current,
        Moonlight,

    Args:
        state: True satellite state; supplies the attitude quaternion.
        env: True environment; supplies position, sun direction, eclipse flag.
        config: The array's normals, full-scale current and noise.
        rng: Supplies the readout noise.

    Returns:
        Per-cell current [A], shape (n_cells,). Currents, not voltages: every
        source document works in current, and the transimpedance gain that
        would convert them is EventSat electronics, unspecified. In eclipse
        the array reads noise about zero, which is a real measurement - unlike
        the fine sun sensor, where a zero vector would be a fabricated
        direction and None is returned instead.
    """
    # Sensor_noise:
    I_noise = rng.normal(0, config.noise_std, len(config.normals))

    # Check for eclipse:
    if env.eclipse:
        return I_noise  # Should the other sensors also return noise/should this return None???

    # Sun direction in body frame:
    sun_vector_body = dcm_eci_to_body(state.q_eci_body) @ env.sun_vector_eci

    # Current due to sun:
    I_sun = config.full_scale_current * np.maximum(0, config.normals @ sun_vector_body)

    # Direction of r_eci:
    r_eci_norm = np.linalg.norm(env.r_eci)
    r_eci_direction = env.r_eci / r_eci_norm

    # r_nadir:
    r_nadir_eci = -r_eci_direction
    r_nadir_body = dcm_eci_to_body(state.q_eci_body) @ r_nadir_eci

    # View factor:
    view_factor = ( R_EARTH / r_eci_norm )**2

    # Current due to Earth's albedo:
    I_alb = (
        config.full_scale_current * EARTH_BOND_ALBEDO * view_factor 
        * np.maximum(0, r_eci_direction @ env.sun_vector_eci ) # Due to brightness of land below satellite
        * np.maximum(0, config.normals @ r_nadir_body) # Due to how much light a given diode cathes
    )

    # Total current:
    I_css = I_sun + I_alb

    # Measured current:
    I_meas = I_css + I_noise
    
    return I_meas


def read_earth_horizon(
    state: SatState, env: EnvironmentData, config: EarthHorizonConfig, rng: np.random.Generator
) -> Optional[np.ndarray]:

    """Measured nadir direction for the earth horizon sensor, unit (3,), sensor frame.

        Returns None when there is no measurement: above the slew cutoff, when the
        horizon arc is rotated too far in the image, or when the limb is not in the
        field of view. pitch and roll are computed from the body-frame nadir while
        the return is sensor-frame.

        ASSUMPTIONS: 
            - Body to sensor mounting 
            - The Earth's horizon is a circle around nadir
            - The field-of-view figures are inconsistent in the ADCS PD and the EHS PD. and
            - The 90 deg diagonal value was not used, instead the diagonal of the 72x90 FoV
                was used

        Not modelled:
            Slew-dependent accuracy degradation - 'dependent on slew' is stated
                without a shape, same as the fine sun sensor (D42).
            Earth oblateness - theta_hor uses the equatorial radius, which is what
                the ADCS PD p.30 formula specifies. The polar radius would shift it
                by ~0.497 deg.

        Args:
            state: True satellite state; supplies attitude and body rate.
            env: True environment; supplies position, from which both nadir and the
                current horizon angle are derived.
            config: This unit's mounting, field of view, roll bound, noise and slew
                cutoff.
            rng: Supplies the pitch and roll noise.

        Returns:
            Unit nadir direction in the sensor frame, shape (3,), or None.
        """
    
    # Sensor noise:
    ehs_angle_noise = rng.normal(0.0, config.noise_std, 2)

    # Check if slew rate is too large for a measurement (conservative):
    if np.linalg.norm(state.omega_body) > config.max_slew_rate:
        return None

    # Projection Matrices:
    P_xy = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    P_yz = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    P_xz = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    # Direction of r_eci:
    r_eci_norm = np.linalg.norm(env.r_eci)
    r_eci_direction = env.r_eci / r_eci_norm
    
    # r_nadir:
    r_nadir_eci = -r_eci_direction
    r_nadir_body = dcm_eci_to_body(state.q_eci_body) @ r_nadir_eci
    r_nadir_body_xz = P_xz @ r_nadir_body # Projection on xz plane
    r_nadir_body_yz = P_yz @ r_nadir_body # Projection on yz plane 
    r_nadir_sen = config.body_to_sensor @ r_nadir_body
    r_nadir_sen_xy = P_xy @ r_nadir_sen
    r_nadir_sen_norm = r_nadir_sen / np.linalg.norm(r_nadir_sen)

    # Azimuth of nadir in the sensor frame nominal:
    tan_azim_nadir_nominal = config.body_to_sensor @ np.array([0, 0, 1])
    azim_nadir_nominal = np.arctan2(tan_azim_nadir_nominal[1], tan_azim_nadir_nominal[0])

    # Azimuth of nadir in the sensor frame:
    azim_nadir = np.arctan2(r_nadir_sen_xy[1], r_nadir_sen_xy[0])

    # Roll in sensor frame:
    roll_sen = (azim_nadir - azim_nadir_nominal + np.pi) % (2*np.pi) - np.pi

    # Is the roll in sensor frame along boresight too big for measurement:
    if abs(roll_sen) > config.horizon_roll_half_angle:
        return None

    # Field of view as a function of roll:
    c, s = abs(np.cos(roll_sen)), abs(np.sin(roll_sen))
    fov = np.arctan(min(
        np.tan(config.fov_half_angle_vertical) / max(c, 1e-12),
        np.tan(config.fov_half_angle_horizontal) / max(s, 1e-12),
    ))

    # Direction of boresight relative to nadir in sensor frame:
    boresight_dir = np.arccos(np.clip(r_nadir_sen_norm @ np.array([0, 0, 1]), -1, 1 ))

    # Earth horizon angle
    earth_horizon_angle = np.arcsin(R_EARTH / r_eci_norm)

    # Is Earth's horizon in the FoV:
    if abs(boresight_dir - earth_horizon_angle) > fov:
        return None

    # True pitch and roll angles:
    pitch_true = np.arctan2(-r_nadir_body_xz[0], r_nadir_body_xz[2]) # angle between naddir in body projected onto xz_body and z_body
    roll_true = np.arctan2(r_nadir_body_yz[1], r_nadir_body_yz[2]) # angle between naddir in body projected onto yz_body and z_body
    
    # Measured pitch and roll angles:
    pitch_meas = pitch_true + ehs_angle_noise[0]
    roll_meas = roll_true + ehs_angle_noise[1]

    if abs(pitch_meas) > np.pi/2 or abs(roll_meas) > np.pi/2:
        return None

    # Nadir measured:
    r_nadir_body_z = 1 / np.sqrt(np.tan(pitch_meas) **2 + np.tan(roll_meas) **2 + 1)
    r_nadir_body_y = r_nadir_body_z * np.tan(roll_meas)
    r_nadir_body_x = - r_nadir_body_z * np.tan(pitch_meas)
    r_nadir_body_meas = np.array([r_nadir_body_x, r_nadir_body_y, r_nadir_body_z])
    r_nadir_sen_meas = config.body_to_sensor @ r_nadir_body_meas

    return r_nadir_sen_meas
    

def read_star_tracker(
    state: SatState, env: EnvironmentData, config: StarTrackerConfig, rng: np.random.Generator
) -> np.ndarray:
    """Measured attitude for one star tracker: quaternion (ECI to body),
    shape (4,), scalar-first.
    """
    return np.array([1.0, 0.0, 0.0, 0.0])

# ω_meas = R_S @ ω_body + b + n_v,   n_v ~ N(0, (arw/sqrt(dt))^2),   clipped to +-max_rate
def read_rate_gyro(
    state: SatState,
    env: EnvironmentData, # don't delete
    config: RateGyroConfig,
    bias: np.ndarray,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Measured angular velocity for one rate gyro [rad/s], shape (3,),
    sensor frame.
    
    Measurement model, after Farrenkopf (1978), "Analytic Steady-State Accuracy
    Solutions for Two Common Spacecraft Attitude Estimators", JGC 1(4):

        w_meas = w_true + b + n_v,    b_dot = n_u

    Noise scaling. arw is a continuous-time coefficient [rad/sqrt(s)]. 
    Woodman (2007) Eq. 5:

        sigma_theta(t) = sigma * sqrt(dt * t)

    Requiring that to equal the sampling-independent form arw * sqrt(t) gives
    sigma = arw / sqrt(dt).

    Assumption: dt is the gyro's sampling interval. If the loop ever runs
    faster than the unit produces data, this overstates the noise.

    Bias frame. b is in the SENSOR frame, not the body frame, and is added
    after R_S.

    The bias arrives already propagated: initial_sensor_state draws it at
    turn-on and propagate_sensor_state advances the random walk in step()

    PLACEHOLDER PARAMETERS.

    Not modelled:
        Temperature-dependent bias drift - named explicitly in the CubeADCS
            C&O Manual v1.05 p.93 as a real effect for these units. Needs a
            thermal model and a temperature coefficient on the bias state.
        Scale factor and axis misalignment - unpublished, and separable from
            body_to_sensor only with calibration data.
        Quantisation - no ADC resolution figure available.
        g-sensitivity - MEMS gyros respond to linear acceleration, negligible
            in free fall except during thruster firing, which EventSat has not.

    Args:
        state: True satellite state; supplies the body-frame angular velocity.
        env: Unused. Present for signature uniformity across the read_*
            functions - a gyro measures rotation, not the environment.
        config: This gyro's mounting, noise coefficient and range.
        bias: This gyro's current bias [rad/s], shape (3,), sensor frame,
            from SensorState.gyro_bias.
        dt: Sampling interval [s]; sets the per-sample noise magnitude.
        rng: Supplies the angle-random-walk noise.

    Returns:
        Measured angular velocity [rad/s], shape (3,), sensor frame, clipped to
        the unit's measurement range.

    """
    # True w in sensor frame:
    w_sensor = config.body_to_sensor @ state.omega_body
    # Measured w in sensor frame:
    w_meas = w_sensor + bias + rng.normal(0, config.arw/np.sqrt(dt), 3)
    w_meas = np.clip(w_meas, -config.max_rate, config.max_rate)
    return w_meas


@dataclass
class SensorMeasurements:
    """All sensor outputs at one instant, produced by the read functions and
    used by the estimator.

    Attributes:
        magnetometers: Measured field per magnetometer [T], each shape (3,),
            sensor frame.
        fine_sun_sensors: Sun unit vector per fine sun sensor, each shape (3,),
            sensor frame; None when the sun is out of view.
        coarse_sun: Photodiode currents from the coarse array, shape (n_cells,).
        earth_horizon: Nadir unit vector in the sensor frame, shape (3,); zero
            when nadir is out of the field of view.
        star_trackers: Attitude quaternion (ECI to body) per star tracker, each
            shape (4,), scalar-first; empty when none are fitted.
        gyros: Measured angular velocity per rate gyro [rad/s], each shape
            (3,), sensor frame.

    Each list holds one entry per configured instance, in the same order as the
    matching SensorSuite field, absent hardware yields an empty list.
    """

    magnetometers: List[np.ndarray]
    fine_sun_sensors: List[Optional[np.ndarray]]
    coarse_sun: np.ndarray
    earth_horizon: np.ndarray
    star_trackers: List[np.ndarray]
    gyros: List[np.ndarray]    
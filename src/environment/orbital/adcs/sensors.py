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

@dataclass
class SensorState:
    """Internal error states of the sensor suite that carry memory across steps.

    Attributes:
        gyro_bias: Rate gyro bias [rad/s], sensor frame, one row per gyro,
            shape (n_gyros, 3).
        fss_bias: FSS bias [rad], one row per sensor, shape(n_fss, 2) ((alpha, beta) incident angles),
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
) -> np.ndarray:
    """Photodiode voltages for the coarse sun sensor array, shape (n_cells,).
    """
    return np.zeros(len(config.normals))


def read_earth_horizon(
    state: SatState, env: EnvironmentData, config: EarthHorizonConfig, rng: np.random.Generator
) -> np.ndarray:
    """Measured nadir direction for the earth horizon sensor, shape (3,).
    """
    return np.zeros(3)


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
        coarse_sun: Photodiode voltages from the coarse array, shape (n_cells,).
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
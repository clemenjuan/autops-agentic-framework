"""ADCS sensor models.

The read functions simulate one sensor each: they take the true satellite
state and the true environment, apply that sensor's measurement model and return
the measurement. 

SensorMeasurements bundles one timestep's worth of these outputs for the
estimator.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
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
        gyro_bias: Rate gyro bias [rad/s], body frame, one row per gyro,
            shape (n_gyros, 3).
    """

    gyro_bias: np.ndarray


def initial_sensor_state(
    sensors: SensorSuite, rng: np.random.Generator
) -> SensorState:
    """Draw the turn on sensor error states.
    """
    gyro_bias = np.array(
        [rng.normal(0.0, g.bias_initial_std, 3) for g in sensors.rate_gyros]
    ).reshape(len(sensors.rate_gyros), 3)
    return SensorState(gyro_bias=gyro_bias)


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
    return SensorState(gyro_bias=bias + rrw * np.sqrt(dt) * rng.standard_normal(bias.shape))

def read_magnetometer(
    state: SatState, env: EnvironmentData, config: MagnetometerConfig, rng: np.random.Generator
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
    B_meas = np.clip(B_meas, -config.measurement_range, config.measurement_range)
    return B_meas


def read_fine_sun_sensor(
    state: SatState, env: EnvironmentData, config: FineSunSensorConfig, rng: np.random.Generator
) -> np.ndarray:
    """Measured sun direction for one fine sun sensor, shape (3,), sensor frame.
    """
    return np.zeros(3)


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

def read_rate_gyro(
    state: SatState,
    env: EnvironmentData,
    config: RateGyroConfig,
    bias: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Measured angular velocity for one rate gyro [rad/s], shape (3,),
    sensor frame.
    """
    return np.zeros(3)


@dataclass
class SensorMeasurements:
    """All sensor outputs at one instant, produced by the read functions and
    used by the estimator.

    Attributes:
        magnetometers: Measured field per magnetometer [T], each shape (3,),
            sensor frame.
        fine_sun_sensors: Sun unit vector per fine sun sensor, each shape (3,),
            sensor frame; zero vector when the sun is out of view.
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
    fine_sun_sensors: List[np.ndarray]
    coarse_sun: np.ndarray
    earth_horizon: np.ndarray
    star_trackers: List[np.ndarray]
    gyros: List[np.ndarray]
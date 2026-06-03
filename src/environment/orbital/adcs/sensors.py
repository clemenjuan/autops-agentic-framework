"""ADCS sensor models.

The read functions simulate one sensor each: they take the true satellite
state and the true environment, apply that sensor's measurement model and return
the measurement. 

SensorMeasurements bundles one timestep's worth of these outputs for the
estimator.

All functions are currently place-holders, so the loop runs.
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
)
from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.propagator import EnvironmentData


def read_magnetometer(
    state: SatState, env: EnvironmentData, config: MagnetometerConfig
) -> np.ndarray:
    """Measured magnetic field for one magnetometer [T], shape (3,).
    """
    return np.zeros(3)


def read_fine_sun_sensor(
    state: SatState, env: EnvironmentData, config: FineSunSensorConfig
) -> np.ndarray:
    """Measured sun direction for one fine sun sensor, shape (3,), sensor frame.
    """
    return np.zeros(3)


def read_coarse_sun_sensor(
    state: SatState, env: EnvironmentData, config: CoarseSunSensorConfig
) -> np.ndarray:
    """Photodiode voltages for the coarse sun sensor array, shape (n_cells,).
    """
    return np.zeros(len(config.normals))


def read_earth_horizon(
    state: SatState, env: EnvironmentData, config: EarthHorizonConfig
) -> np.ndarray:
    """Measured nadir direction for the earth horizon sensor, shape (3,).
    """
    return np.zeros(3)


def read_star_tracker(
    state: SatState, env: EnvironmentData, config: StarTrackerConfig
) -> np.ndarray:
    """Measured attitude for one star tracker: quaternion (ECI to body),
    shape (4,), scalar-first.
    """
    return np.array([1.0, 0.0, 0.0, 0.0])


@dataclass
class SensorMeasurements:
    """All sensor outputs at one instant, produced by the read functions and
    used by the estimator.

    Each list holds one entry per configured instance, in the same order as the
    matching SensorSuite field, absent hardware yields an empty list.

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
    """

    magnetometers: List[np.ndarray]
    fine_sun_sensors: List[np.ndarray]
    coarse_sun: np.ndarray
    earth_horizon: np.ndarray
    star_trackers: List[np.ndarray]
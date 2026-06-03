"""Attitude estimator (Multiplicative Extended Kalman Filter).

Maintains the estimated satellite state ŝ - attitude, angular velocity, and gyro
bias, with an error covariance and advances it each timestep from the sensor
measurements.

All functions are currently place-holders, so the loop runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.environment.orbital.adcs.sensors import SensorMeasurements


@dataclass
class EstimatorState:
    """The estimator's running estimate of the satellite state.

    Attributes:
        q_estimate: Estimated attitude quaternion (ECI to body), shape (4,),
            scalar-first.
        omega_estimate: Estimated angular velocity in the body frame [rad/s],
            shape (3,).
        bias_estimate: Estimated gyro bias [rad/s], shape (3,).
        covariance: Error-state covariance, shape (6, 6). The 6-dim error state
            is [attitude error (3); gyro-bias error (3)].
    """

    q_estimate: np.ndarray
    omega_estimate: np.ndarray
    bias_estimate: np.ndarray
    covariance: np.ndarray


def initial_estimator_state() -> EstimatorState:
    """Return an initial estimate.
    """
    return EstimatorState(
        q_estimate=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_estimate=np.zeros(3),
        bias_estimate=np.zeros(3),
        covariance=np.eye(6),
    )


def update_estimator(
    estimator: EstimatorState,
    measurements: SensorMeasurements,
    dt: float,
) -> EstimatorState:
    """Advance the estimate by one timestep from the latest measurements.

    Args:
        estimator: The current estimate (threaded in and out each step).
        measurements: All sensor outputs for this timestep.
        dt: Timestep [s].

    Returns:
        The updated estimate.
    """
    return estimator
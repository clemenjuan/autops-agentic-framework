"""ADCS control layer.

Defines Setpoint (the target the controller tracks) and compute_control, which
turns the estimator's view of the satellite plus a setpoint into a ControlCommand
for the actuators. 

ControlCommand itself lives in actuators.py, since it is the
actuator-command contract, the controller only produces it.

All functions are currently place-holders, so the loop runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.environment.orbital.adcs.actuators import ControlCommand
from src.environment.orbital.adcs.configs import ActuatorSuite
from src.environment.orbital.adcs.estimator import EstimatorState


@dataclass
class Setpoint:
    """The target attitude and rate the controller is tracking.

        Attributes:
        target_q_eci_body: Target attitude quaternion (ECI to body), shape (4,),
            scalar-first.
        target_omega_body: Target angular velocity in the body frame [rad/s],
            shape (3,).
    """

    target_q_eci_body: np.ndarray
    target_omega_body: np.ndarray


def initial_setpoint() -> Setpoint:
    """Return a default "hold identity attitude, zero rate" setpoint."""
    return Setpoint(
        target_q_eci_body=np.array([1.0, 0.0, 0.0, 0.0]),
        target_omega_body=np.zeros(3),
    )


def compute_control(
    estimator: EstimatorState,
    setpoint: Setpoint,
    actuators: ActuatorSuite,
    dt: float,
) -> ControlCommand:
    """Compute the actuator commands for this timestep.

    Args:
        estimator: The estimated state, never the true state.
        setpoint: The attitude and rate the controller is tracking.
        actuators: The actuator suite: sets how many commands to produce and
            (later) the allocation geometry.
        dt: Timestep [s].

    Returns:
        A ControlCommand with one entry per configured actuator.
    """
    return ControlCommand(
        wheel_commands=[0.0 for _ in actuators.reaction_wheels],
        mtq_commands=[0.0 for _ in actuators.magnetorquers],
    )
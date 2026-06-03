"""Rigid-body attitude dynamics and environmental disturbances.

integrate is the only writer of the true rotational state: given the net torque
on the body, it advances attitude, angular velocity, and wheel speeds over one
timestep.

disturbance_torque returns the net environmental torque that feeds into it.

All functions are currently place-holders, so the loop runs.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.propagator import EnvironmentData


def integrate(state: SatState, total_torque: np.ndarray, dt: float) -> SatState:
    """Advance the true rotational state by one timestep.

    Args:
        state: Current true satellite state.
        total_torque: Net body-frame torque acting on the satellite [N·m],
            shape (3,) — the sum of actuator and disturbance torques.
        dt: Timestep [s].

    Returns:
        The state advanced by dt.
    """
    return replace(state, t=state.t + dt)


def disturbance_torque(state: SatState, env: EnvironmentData) -> np.ndarray:
    """Net environmental disturbance torque in the body frame [N·m], shape (3,).

    Args:
        state: Current true satellite state.
        env: Environment at this instant.

    Returns:
        The summed disturbance torque.
    """
    return np.zeros(3)
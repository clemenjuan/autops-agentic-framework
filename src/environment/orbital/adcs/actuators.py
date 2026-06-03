"""ADCS actuator models.

The apply functions simulate one actuator each: they take the true state and
environment, the actuator's config, and the command issued to it, and return the
body-frame torque it produces. 
ControlCommand is the bundle the controller emits
and simulation.step() unpacks into the per-actuator commands.

All functions are currently place-holders, so the loop runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.environment.orbital.adcs.configs import MagnetorquerConfig, ReactionWheelConfig
from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.propagator import EnvironmentData


def apply_reaction_wheel(
    state: SatState,
    env: EnvironmentData,
    config: ReactionWheelConfig,
    command: float,
) -> np.ndarray:
    """Body-frame torque produced by one reaction wheel [N·m], shape (3,).

    Args:
        state: True satellite state (provides the current wheel speed).
        env: Environment at this instant; not used by the wheel model but
            accepted so all actuators share one signature.
        config: This wheel's configuration.
        command: Commanded wheel torque along the spin axis [N·m].
    """
    return np.zeros(3)


def apply_magnetorquer(
    state: SatState,
    env: EnvironmentData,
    config: MagnetorquerConfig,
    command: float,
) -> np.ndarray:
    """Body-frame torque produced by one magnetorquer [N·m], shape (3,).

    Args:
        state: True satellite state (provides attitude for the field rotation).
        env: Environment at this instant; provides the magnetic field.
        config: This rod's configuration.
        command: Commanded dipole magnitude along the rod axis [A·m²].
    """
    return np.zeros(3)


@dataclass
class ControlCommand:
    """Actuator commands for one timestep, produced by the controller.

    Attributes:
        wheel_commands: Torque command per reaction wheel [N·m], in the order of
            ActuatorSuite.reaction_wheels.
        mtq_commands: Dipole command per magnetorquer [A·m²], in the order of
            ActuatorSuite.magnetorquers.
    """

    wheel_commands: List[float]
    mtq_commands: List[float]
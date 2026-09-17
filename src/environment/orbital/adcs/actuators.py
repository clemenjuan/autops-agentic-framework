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
from src.environment.orbital.adcs.dynamics import dcm_eci_to_body

import logging

logger = logging.getLogger(__name__)

def _bounded_motor_torque(
    config: ReactionWheelConfig,
    command: float,    
) -> float:
    
    """Helper function, checks if the commanded motor torque is within the physical
     limits of the motor [N·m], scalar.

    The actuator's torque capability bound, applied to the raw command. This
    is not the torque the flywheel ends up receiving: apply_reaction_wheel
    subtracts friction and then applies a separate momentum-rate limit, and
    returns that net value instead. 

    Args:
        config: This wheel's configuration.
        command: Commanded motor torque along the spin axis [N·m].

    Returns:
        The commanded motor torque, bounded to ±max_torque [N·m].
    """

    bounded_motor_torque = float(np.clip(command, -config.max_torque, config.max_torque))

    return bounded_motor_torque

def _friction_torque(
    speed: float,
    config: ReactionWheelConfig,   
) -> float:

    """Helper function for calculating the total torque due to friction
    folowing the friction model from Paluszek Eq. 10.14.

    No stiction included, but will think about it.

    Args:
        speed: Supplies this wheel's current speed.
        config: This wheel's configuration.
    Returns:
        The torque along ``config.spin_axis_body`` due to friction [N·m].
    """
    
    friction_torque = float(
        config.friction_coulomb * np.sign(speed) + 
        config.friction_viscous * speed
    )

    return friction_torque

def apply_reaction_wheel(
    state: SatState,
    config: ReactionWheelConfig,
    command: float,
    wheel_index: int,
    dt: float,
) -> float:

    """Motor torque actually delivered to one flywheel [N·m], scalar.
    Args:
        state: True satellite state; supplies this wheel's current speed.
        config: This wheel's configuration.
        command: Commanded motor torque along the spin axis [N·m].
        wheel_index: Index of this wheel in ``state.wheel_speeds``.
        dt: Length of the step this torque will be held over [s].
    Returns:
        The achieved motor torque along ``config.spin_axis_body`` [N·m].
    """

    torque = _bounded_motor_torque(config, command) # Torque Clamping
    speed = float(state.wheel_speeds[wheel_index])
      
    torque -= _friction_torque(speed, config)        

    momentum = config.wheel_inertia * speed

    # Saturation/Momentum limit
    lower_m = min(0.0, (-config.momentum_at_max_speed - momentum) / dt)
    upper_m = max(0.0, (config.momentum_at_max_speed - momentum) / dt)
    torque = float(np.clip(torque, lower_m, upper_m))

    return torque

def power_reaction_wheel(
    config: ReactionWheelConfig, 
    command: float, 
    wheel_speed: float,    
) -> float:

    """Command-dependent electrical power drawn by one wheel [W], scalar.

        P(Omega, tau) = [P_fit(|Omega|) - c − tau_drag(omega)*Omega/d] + max(0, tau_bounded * Omega/d)
        P_fit(Omega)  = c + b*|Omega| + a*Omega^2

    P_fit(Omega) is the empirically derived least-squares fit to 49 points
    digitised from the ADCS ICD p.45 steady-state plot. The constant 
    term is substracted back in the P(Omega, tau) equation in order to 
    exclude the idle power draw and keep the model consistent with the 
    magnetorquer power model.

    tau_drag(omega)*Omega: the friction is already included in the fit
    and in tau_bounded (the controller commands tau = tau_drag to hold speed, 
    and that command arrives as command), so it needs to be subtracted 
    in order to prevent counting it twice in P(Omega, tau).

    max(0, tau_bounded * Omega) is the mechanical work of then commanded torque.
    
    tau_bounded is the commanded motor torque after the torque clip, not the
    net torque apply_reaction_wheel returns.

    d is an empirical correction for the torque term, that is added to the fitted
    power equation. 

    ASSUMPTIONs:
      - 16 V. The ICD characterises at 8 V and 16 V; EventSat's bus
        is 12-16.2 V (CMO), so 16 V was chosen as the only curve inside the range.
      - plot-axis inertia. CubeSpace measured speed and multiplied by an
        inertia, and their documents carry two different inertia values (ICD Table 16
        gives 9.51e-6; the PD's 5.7 mNms at 6000 rpm implies 9.07e-6).

    Not modelled:
      - Copper loss. Requires the motor's resistance and torque
        constant, neither published.
      - Bus voltage. See the 16 V assumption above; the model has no voltage
        argument.
      - The ICD torque traces spike to 150-220 mA near
        zero momentum, which a steady model does not capture.
      - Temperature. 

    Regeneration. The driver dissipates rather than returning charge. 
    ICD traces go flat rather than negative

    Args:
        config: This wheel's configuration.
        command: Commanded motor torque along the spin axis [N·m].
        wheel_speed: This wheel's speed relative to the body [rad/s], signed.

    Returns:
        Command-dependent electrical power [W], always >= 0.
    """
    
    power_fit = (
        config.power_fit_const_param + 
        config.power_fit_lin_param * abs(wheel_speed) + 
        config.power_fit_square_param * wheel_speed ** 2
    )

    cmd_dependent_power = (
        power_fit - config.power_fit_const_param - 
        _friction_torque(wheel_speed, config) * wheel_speed / config.mechanical_power_scale + 
        max(0.0, _bounded_motor_torque(config, command) * wheel_speed / config.mechanical_power_scale)
    )

    return cmd_dependent_power

def _bounded_avg_dipole(
    config: MagnetorquerConfig,
    command: float,
)-> float:
    """Helper function, checks if the commanded average dipole magnitude is within 
    the physical limits of the magnetorquer, which it calculates and clips if needed.

    Args:
        config: Used for maximum dipole moment per rod and the maximum duty cycle 
        command: Commanded the step average dipole magnitude along the rod axis [A·m²].

    Returns:
        Bounded command value, within the physical capabilities of a rod
        [A*m^2]
    """
    max_avg_dipole = config.max_dipole * config.duty_max
    bounded_avg_dipole = float(
            np.clip(command, -max_avg_dipole, max_avg_dipole)
        )
    
    return bounded_avg_dipole


def apply_magnetorquer(
    state: SatState,
    env: EnvironmentData,
    config: MagnetorquerConfig,
    command: float,
) -> np.ndarray:
    """Body-frame torque produced by one magnetorquer [N·m], shape (3,).

    Args:
        state: True satellite state; supplies the attitude for the field
            rotation.
        env: Environment at this instant; supplies the geomagnetic field.
        config: This rod's configuration.
        command: Commanded dipole magnitude along the rod axis [A·m²].

    Returns:
        Body-frame torque [N·m], shape (3,).
    """

    dipole_body = _bounded_avg_dipole(config, command) * config.axis_body

    b_body = dcm_eci_to_body(state.q_eci_body) @ env.b_field_eci

    return np.cross(dipole_body, b_body)

def power_magnetorquer(
    config: MagnetorquerConfig,
    command: float,
)-> float:

    """Electrical power drawn by one magnetorquer [W], scalar.

        P(m) = R * m^2 / (K_m^2 * duty_max)

    with R the coil resistance, K_m the magnetic gain and m the bounded
    average dipole magnitude.

    ASSUMPTION - quadratic form. Power goes as m^2 only if the drive current
    is smooth. PD p.10 states rods are typically PWM-driven and names
    variable-current drivers as an alternative, but does not state which the
    CubeTorquer uses. Both named schemes give a smooth current here, since the
    coil's corner sits at 1/(2*pi*L/R) = 21 Hz, far below any normal PWM carrier.
   
    PRECONDITION - the duty treatment holds only while ``step_s`` is a
    multiple of the ADCS loop period 

    Not modelled:
      - Coil resistance vs temperature.
      - Coil current transient within the energised window. Treating current
        as rectangular ignores the rise and the decay tail.
      - Driver quiescent and switching losses. No figure published.
      - Linearity error. m = K_m·I is assumed exact; PD p.10 bounds the
        departure at <=2.5% or <=2%. The looser figure is taken.

    Args:
        config: This rod's configuration.
        command: Commanded period-averaged dipole along the rod axis [A·m²].

    Returns:
        Loop-averaged electrical power [W], always >= 0.
    """

    bounded_avg_dipole = _bounded_avg_dipole(config, command)
    mtq_power = (config.coil_resistance * bounded_avg_dipole **2 / 
                 (config.magnetic_gain **2 * config.duty_max)
    )

    return float(mtq_power)


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
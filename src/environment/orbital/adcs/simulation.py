"""Closed-loop ADCS simulation.

step() runs the full control loop once
run() drives it over a range of steps

Data flow:

    sense -> estimate -> control -> actuate -> disturb -> integrate -> reconcile

The controller sees only the estimator state.
The dynamics integrator and the propagator are the only writers of the true state.

Time is step-based. 

run() works in integer step indices (start_step, end_step); the step length
comes from SimulationConfig.step_s and is passed to the physics as dt.

Continuous time (state.t) is carried in seconds for the physics that needs it.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from src.environment.orbital.adcs.actuators import apply_magnetorquer, apply_reaction_wheel
from src.environment.orbital.adcs.configs import ActuatorSuite, SensorSuite, SatelliteConfig, OrbitConfig, SimulationConfig
from src.environment.orbital.adcs.control import Setpoint, compute_control, initial_setpoint
from src.environment.orbital.adcs.dynamics import disturbance_torque, integrate
from src.environment.orbital.adcs.estimator import (
    EstimatorState,
    initial_estimator_state,
    update_estimator,
)
from src.environment.orbital.adcs.sensors import (
    SensorMeasurements,
    read_coarse_sun_sensor,
    read_earth_horizon,
    read_fine_sun_sensor,
    read_magnetometer,
    read_star_tracker,
    read_rate_gyro,
    SensorState,
    propagate_sensor_state,
    initial_sensor_state,


)
from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.propagator import get_environment, configure

logger = logging.getLogger(__name__)


def initial_state(
    t: float = 0.0,
    n_wheels: int = 4,
) -> SatState:
    """Create the initial true state.

    Args:
        t: Initial time since the simulation epoch [s].
        n_wheels: Number of reaction wheels (sizes the wheel-speed vector).
    """

    return SatState(
        t=t,
        q_eci_body=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_body=np.zeros(3),
        wheel_speeds=np.zeros(n_wheels),
        r_eci=np.zeros(3),
        v_eci=np.zeros(3),
    )


def step(
    state: SatState,
    sensor_state: SensorState,
    estimator: EstimatorState,
    sensors: SensorSuite,
    actuators: ActuatorSuite,
    satellite: SatelliteConfig,
    setpoint: Setpoint,
    sim: SimulationConfig,
    rng: np.random.Generator,
) -> Tuple[SatState, SensorState, EstimatorState]:
    """Advance the simulation by one timestep, running the full closed loop.

    Args:
        state: Current true satellite state.
        sensor_state: Current sensor error states (gyro bias).
        estimator: Current estimator state.
        sensors: The sensor suite.
        actuators: The actuator suite.
        satellite: Satellite physical parameters (mass properties + wheel
            geometry) used by the rigid-body integrator.
        setpoint: The target the controller tracks.
        sim: Numerical simulation parameters; supplies the timestep.
        rng: Supplies sensor noise and state drift.

    Returns:
        A tuple of: new true state, new sensor state, updated estimator state.
    """
    dt = sim.step_s
    env = get_environment(state.t)

    # Sense: every sensor reads the true state and the true environment.
    measurements = SensorMeasurements(
        magnetometers=[read_magnetometer(state, env, mag, rng) for mag in sensors.magnetometers],
        fine_sun_sensors=[
            read_fine_sun_sensor(state, env, fss, rng) for fss in sensors.fine_sun_sensors
        ],
        coarse_sun=read_coarse_sun_sensor(state, env, sensors.coarse_sun_sensor, rng),
        earth_horizon=read_earth_horizon(state, env, sensors.earth_horizon_sensor, rng),
        star_trackers=[read_star_tracker(state, env, st, rng) for st in sensors.star_trackers],
        gyros=[
            read_rate_gyro(state, env, g, sensor_state.gyro_bias[i],dt, rng)
            for i, g in enumerate(sensors.rate_gyros)
        ],
    )

    # Estimate: fuse the measurements into the estimated state.
    estimator = update_estimator(estimator, measurements, dt)

    # Control
    command = compute_control(estimator, setpoint, actuators, dt)

    # Actuate: the reaction wheels return per-wheel motor torques
    wheel_torque = np.array(
        [
            apply_reaction_wheel(state, wheel, cmd, i, dt)
            for i, (wheel, cmd) in enumerate(
                zip(actuators.reaction_wheels, command.wheel_commands)
            )
        ]
    )

    body_torque = np.zeros(3)
    for rod, cmd in zip(actuators.magnetorquers, command.mtq_commands):
        body_torque = body_torque + apply_magnetorquer(state, env, rod, cmd)

    # Disturb: environmental disturbance torque adds into the body torque.
    body_torque = body_torque + disturbance_torque(state, env, satellite)

    # Integrate: the sole writer of the true rotational state advances it.
    new_state = integrate(state, body_torque, wheel_torque, satellite, dt)

    # Drift: sensor error states advance alongside the plant.
    new_sensor_state = propagate_sensor_state(sensor_state, sensors, dt, rng)

    # Reconcile: the orbital state comes from the propagator at the new time
    env_next = get_environment(new_state.t)
    new_state = replace(
        new_state, r_eci=env_next.r_eci, v_eci=env_next.v_eci
    )

    return new_state, new_sensor_state, estimator


def run(
    sensors: SensorSuite,
    actuators: ActuatorSuite,
    satellite: SatelliteConfig,
    sim: SimulationConfig,
    orbit: OrbitConfig,
    start_step: int,
    end_step: int,
    setpoint: Optional[Setpoint] = None,
) -> List[SatState]:
    """Run the closed-loop simulation over a range of steps.

    Args:
        sensors: The sensor suite.
        actuators: The actuator suite.
        satellite: Satellite physical parameters passed to the integrator.
        sim: Numerical simulation parameters; supplies the timestep.
        start_step: First step index (inclusive).
        end_step: Final step index (exclusive); the run executes
            ``end_step - start_step`` steps.
        setpoint: Target for the controller; defaults to holding identity
            attitude at zero rate.
        orbit: Orbit definition; configures the propagator. Required, there is
            no neutral default, and an unconfigured propagator yields a
            physics-free run that completes without error.

    Returns:
        The state history: the initial state followed by the state after each
        step (length ``end_step - start_step + 1``).
    """
    
    configure(orbit)
    if setpoint is None:
        setpoint = initial_setpoint()

    sim = sim.resolved()
    rng = np.random.default_rng(sim.seed)

    t0 = start_step * sim.step_s
    env0 = get_environment(t0)
    state = replace(
        initial_state(t0, len(actuators.reaction_wheels)),
        r_eci=env0.r_eci,
        v_eci=env0.v_eci,
    )

    logger.info(
        "Running ADCS simulation: steps %d..%d at %g s/step, seed %d",
        start_step,
        end_step,
        sim.step_s,
        sim.seed,
    )

    sensor_state = initial_sensor_state(sensors, rng)

    estimator = initial_estimator_state()
    history = [state]
    
    
    for _ in range(start_step, end_step):
        state, sensor_state, estimator = step(
            state, sensor_state, estimator, sensors, actuators,
            satellite, setpoint, sim, rng
        )
        history.append(state)

    return history
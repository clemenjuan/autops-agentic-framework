"""Test for the ADCS simulation skeleton.

Confirms the closed loop runs through all modules and produces a state history
of the expected shape. 

It checks no physics (every dummy function returns zeros)

Its job is to catch wiring and interface breakage as real implementations replace 
the dummy ones, so it asserts structure correctness. 
"""
from __future__ import annotations

from typing import List

import pytest

from src.environment.orbital.adcs.control import initial_setpoint
from src.environment.orbital.adcs.estimator import initial_estimator_state
from src.environment.orbital.adcs.eventsat import actuators, sensors
from src.environment.orbital.adcs.simulation import initial_state, run, step
from src.environment.orbital.adcs.state import SatState

STEP_S = 1.0
START_STEP = 0
END_STEP = 10


@pytest.fixture
def history() -> List[SatState]:
    """Run the skeleton once and share the resulting state history."""
    return run(
        sensors, actuators, step_s=STEP_S, start_step=START_STEP, end_step=END_STEP
    )


def test_eventsat_config_counts() -> None:
    """The EventSat suite has the expected instrument counts."""
    assert len(sensors.magnetometers) == 2
    assert len(sensors.fine_sun_sensors) == 2
    assert len(sensors.star_trackers) == 0
    assert len(actuators.reaction_wheels) == 4
    assert len(actuators.magnetorquers) == 3


def test_run_executes_end_to_end(history: List[SatState]) -> None:
    """run() completes and returns one state per step boundary."""
    assert len(history) == END_STEP - START_STEP + 1
    assert all(isinstance(s, SatState) for s in history)


def test_run_advances_time(history: List[SatState]) -> None:
    """Time runs from start_step * step_s to end_step * step_s."""
    assert history[0].t == START_STEP * STEP_S
    assert history[-1].t == END_STEP * STEP_S


def test_final_state_shapes(history: List[SatState]) -> None:
    """The final state has the expected vector shapes."""
    final = history[-1]
    assert final.q_eci_body.shape == (4,)
    assert final.omega_body.shape == (3,)
    assert final.wheel_speeds.shape == (len(actuators.reaction_wheels),)
    assert final.r_eci.shape == (3,)
    assert final.v_eci.shape == (3,)


def test_single_step_returns_state_and_estimator() -> None:
    """One step returns an advanced state and an estimator with a 6x6 covariance."""
    state = initial_state(0.0, len(actuators.reaction_wheels))
    estimator = initial_estimator_state()
    new_state, new_estimator = step(
        state, estimator, sensors, actuators, initial_setpoint(), STEP_S
    )
    assert isinstance(new_state, SatState)
    assert new_state.t == STEP_S
    assert new_estimator.covariance.shape == (6, 6)
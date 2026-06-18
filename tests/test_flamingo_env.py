"""
Tests for the Flamingo-lite multi-satellite scenario.
"""

from __future__ import annotations

from src.environment.scenarios.flamingo import FlamingoEnvironment


def test_flamingo_reset_creates_constellation() -> None:
    env = FlamingoEnvironment(
        config={
            "constellation_size": 3,
            "max_steps": 10,
            "scenario_params": {"targets": {"count": 4}},
        }
    )

    obs = env.reset(seed=0)

    assert len(obs.constellation_state.satellites) == 3
    assert len(env.targets) == 4
    assert obs.tasks
    assert env.get_metrics()["utility"] == 0.0


def test_flamingo_visible_observation_earns_utility() -> None:
    env = FlamingoEnvironment(
        config={
            "constellation_size": 1,
            "max_steps": 10,
            "scenario_params": {
                "visibility_period_steps": 10,
                "visibility_window_steps": 1,
                "targets": {"count": 1, "priorities": [5.0]},
            },
        }
    )
    obs = env.reset()
    task = obs.tasks[0]

    result = env.step(
        {task["satellite_id"]: {"target_id": task["target_id"]}}
    )

    assert result.rewards["utility"] == 5.0
    assert env.get_metrics()["successful_observations"] == 1.0
    assert env.get_metrics()["coverage_rate"] == 1.0


def test_flamingo_duplicate_observations_are_counted() -> None:
    env = FlamingoEnvironment(
        config={
            "constellation_size": 2,
            "max_steps": 10,
            "scenario_params": {
                "visibility_period_steps": 10,
                "visibility_window_steps": 3,
                "targets": {"count": 1, "priorities": [1.0]},
            },
        }
    )
    env.reset()

    env.step(
        {
            "flamingo_0": {"target_id": "rso_0"},
            "flamingo_1": {"target_id": "rso_0"},
        }
    )

    metrics = env.get_metrics()
    assert metrics["successful_observations"] == 1.0
    assert metrics["duplicate_observation_rate"] == 0.5


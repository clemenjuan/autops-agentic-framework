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


def _catalog(env: FlamingoEnvironment) -> list[tuple[float, int]]:
    return [(t.priority, t.phase_offset) for t in env.targets]


def _stochastic_env() -> FlamingoEnvironment:
    return FlamingoEnvironment(
        config={
            "constellation_size": 3,
            "max_steps": 10,
            "scenario_params": {
                "stochastic": True,
                "visibility_period_steps": 12,
                "targets": {"count": 8, "priorities": [5.0, 3.0, 2.0, 1.0]},
            },
        }
    )


def test_flamingo_stochastic_catalog_is_reproducible_per_seed() -> None:
    """Same seed → identical catalog (so paired-seed org comparison is fair)."""
    env = _stochastic_env()
    env.reset(seed=42)
    first = _catalog(env)
    env.reset(seed=42)
    assert _catalog(env) == first


def test_flamingo_stochastic_catalog_varies_across_seeds() -> None:
    """Different seeds draw different SSA instances (real per-episode variance)."""
    env = _stochastic_env()
    catalogs = []
    for seed in range(5):
        env.reset(seed=seed)
        catalogs.append(tuple(_catalog(env)))
    assert len(set(catalogs)) > 1


def test_flamingo_deterministic_mode_ignores_seed() -> None:
    """Without `stochastic`, the catalog is fixed regardless of seed."""
    env = FlamingoEnvironment(
        config={
            "constellation_size": 3,
            "max_steps": 10,
            "scenario_params": {
                "visibility_period_steps": 12,
                "targets": {"count": 8, "priorities": [5.0, 3.0, 2.0, 1.0]},
            },
        }
    )
    env.reset(seed=1)
    first = _catalog(env)
    env.reset(seed=99)
    assert _catalog(env) == first
    # Deterministic layout: phase_offset = index.
    assert [phase for _, phase in first] == list(range(8))


"""
Runner smoke test for Flamingo-lite.
"""

from __future__ import annotations

from pathlib import Path

from src.orchestration.config_loader import ExperimentConfig
from src.orchestration.experiment_runner import ExperimentRunner


def test_runner_executes_flamingo_smoke(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        experiment_id="flamingo_smoke",
        agent_organization="sas",
        decision_procedure="sda",
        representation="symbolic",
        representation_config={"type": "rule_based_flamingo"},
        operations_paradigm="autonomous_onboard",
        environment={
            "scenario": "flamingo",
            "constellation_size": 3,
            "timestep_seconds": 60,
            "max_steps": 4,
            "scenario_config": {
                "scenario_params": {
                    "targets": {"count": 3, "priorities": [3.0, 2.0, 1.0]},
                    "visibility_period_steps": 6,
                    "visibility_window_steps": 3,
                }
            },
        },
        num_episodes=1,
        max_steps=4,
        output_dir=str(tmp_path / "flamingo_smoke"),
    )

    results = ExperimentRunner(config=cfg).run()
    mean = results["experiment_statistics"].mean

    assert results["experiment_id"] == "flamingo_smoke"
    assert mean["utility"] > 0.0
    assert "coverage_rate" in mean
    assert "duplicate_observation_rate" in mean
    assert (tmp_path / "flamingo_smoke" / "results.json").exists()


def test_runner_executes_flamingo_autonomous_ground(tmp_path: Path) -> None:
    """The shipped baseline path: SAS + autonomous_ground + rule_based_flamingo.

    The committed ``flamingo_sas_ag_symb`` config runs under ``autonomous_ground``
    with ``pass_through_observation`` (the constellation is always in contact in
    this MVP). This guards that exact validate-and-run path, which the onboard
    smoke above does not exercise.
    """
    cfg = ExperimentConfig(
        experiment_id="flamingo_ag_smoke",
        agent_organization="sas",
        decision_procedure="sda",
        representation="symb",
        representation_config={"type": "rule_based_flamingo"},
        operations_paradigm="autonomous_ground",
        operations_paradigm_config={"pass_through_observation": True},
        environment={
            "scenario": "flamingo",
            "constellation_size": 3,
            "timestep_seconds": 60,
            "max_steps": 8,
            "scenario_config": {
                "scenario_params": {
                    "targets": {"count": 4, "priorities": [3.0, 2.0, 1.0]},
                    "visibility_period_steps": 6,
                    "visibility_window_steps": 3,
                }
            },
        },
        num_episodes=1,
        max_steps=8,
        output_dir=str(tmp_path / "flamingo_ag_smoke"),
    )

    results = ExperimentRunner(config=cfg).run()
    mean = results["experiment_statistics"].mean

    assert results["experiment_id"] == "flamingo_ag_smoke"
    assert mean["utility"] > 0.0
    assert (tmp_path / "flamingo_ag_smoke" / "results.json").exists()


def _contended_cfg(org: str, out: Path) -> ExperimentConfig:
    """A contended Flamingo config: shared visibility windows + skewed priorities.

    ``satellite_phase_shift: 0`` makes every satellite see the same RSO windows,
    so they must compete for the top-priority target each step.
    """
    return ExperimentConfig(
        experiment_id=f"flamingo_ctn_{org}",
        agent_organization=org,
        decision_procedure="sda",
        representation="symb",
        representation_config={"type": "rule_based_flamingo"},
        operations_paradigm="autonomous_ground",
        operations_paradigm_config={"pass_through_observation": True},
        environment={
            "scenario": "flamingo",
            "constellation_size": 3,
            "timestep_seconds": 60,
            "max_steps": 24,
            "scenario_config": {
                "scenario_params": {
                    "satellite_phase_shift": 0,
                    "visibility_period_steps": 6,
                    "visibility_window_steps": 3,
                    "targets": {"count": 4, "priorities": [5.0, 3.0, 2.0, 1.0]},
                }
            },
        },
        num_episodes=1,
        max_steps=24,
        output_dir=str(out),
    )


def test_imas_wastes_capacity_while_sas_deconflicts(tmp_path: Path) -> None:
    """The organisation axis must bite: under contention IMAS duplicates, SAS does not.

    Independent agents each see only their own satellite and greedily grab the
    single highest-priority visible RSO, so several land on the same target and
    the environment counts duplicates. The single global SAS planner deconflicts
    across the constellation and wastes nothing. This guards the coordination
    bottleneck the scenario exists to measure — it asserts the *direction*, not
    pinned numbers.
    """
    sas = ExperimentRunner(config=_contended_cfg("sas", tmp_path / "sas")).run()
    imas = ExperimentRunner(
        config=_contended_cfg("independent_mas", tmp_path / "imas")
    ).run()
    sas_m = sas["experiment_statistics"].mean
    imas_m = imas["experiment_statistics"].mean

    # SAS deconflicts globally → no wasted duplicate observations.
    assert sas_m["duplicate_observation_rate"] == 0.0
    # IMAS has no coordination → independent agents collide on the top RSO.
    assert imas_m["duplicate_observation_rate"] > 0.0
    # Coordination pays off in mission utility under contention.
    assert imas_m["utility"] < sas_m["utility"]


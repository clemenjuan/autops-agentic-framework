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


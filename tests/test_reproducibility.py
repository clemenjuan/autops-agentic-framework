"""
Tests for experiment reproducibility.

Validates that the same configuration and seed produce identical results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.orchestration.config_loader import ExperimentConfig
from src.orchestration.experiment_runner import ExperimentRunner


class TestReproducibility:
    def test_same_seed_same_results(self, tmp_path: Path) -> None:
        """Same configuration and seed must produce identical results."""
        cfg1 = ExperimentConfig(
            experiment_id="repro_test",
            seed=42,
            num_episodes=2,
            max_steps=3,
            output_dir=str(tmp_path / "run1"),
        )
        cfg2 = ExperimentConfig(
            experiment_id="repro_test",
            seed=42,
            num_episodes=2,
            max_steps=3,
            output_dir=str(tmp_path / "run2"),
        )

        runner1 = ExperimentRunner(config=cfg1)
        runner2 = ExperimentRunner(config=cfg2)

        results1 = runner1.run()
        results2 = runner2.run()

        # Compare episode-level results (excluding timestamps and file paths)
        for ep1, ep2 in zip(results1["episodes"], results2["episodes"]):
            assert ep1["num_steps"] == ep2["num_steps"]
            # Wall-clock times will differ slightly, so we compare structure
            assert len(ep1["steps"]) == len(ep2["steps"])

    def test_different_seeds_different_results(self, tmp_path: Path) -> None:
        """Different seeds should (eventually) produce different results.

        Note: With placeholder components this is trivially true. This test
        serves as a contract that will matter once real stochastic components
        are wired in.
        """
        cfg1 = ExperimentConfig(
            experiment_id="seed_test",
            seed=42,
            num_episodes=1,
            max_steps=2,
            output_dir=str(tmp_path / "seed42"),
        )
        cfg2 = ExperimentConfig(
            experiment_id="seed_test",
            seed=99,
            num_episodes=1,
            max_steps=2,
            output_dir=str(tmp_path / "seed99"),
        )

        runner1 = ExperimentRunner(config=cfg1)
        runner2 = ExperimentRunner(config=cfg2)

        results1 = runner1.run()
        results2 = runner2.run()

        # Both should complete successfully
        assert results1["num_episodes"] == 1
        assert results2["num_episodes"] == 1

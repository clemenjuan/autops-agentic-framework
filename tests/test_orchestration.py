"""
Tests for the Orchestration layer: config loading, experiment runner, metrics.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.memory.fixed_memory import FixedMemory
from src.orchestration.analysis import compute_pareto_frontier, compare_experiments
from src.orchestration.config_loader import (
    ExperimentConfig,
    EnvironmentConfig,
    MetricsConfig,
    load_config,
    save_config,
)
from src.orchestration.experiment_runner import ExperimentRunner


# ======================================================================
# Configuration tests
# ======================================================================


class TestExperimentConfig:
    def test_default_values(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.seed == 42
        assert cfg.agent_organization == "sas"
        assert cfg.decision_loop == "sda"
        assert cfg.representation == "symbolic"
        assert cfg.emergence_mode == "hand_designed"
        assert cfg.num_episodes == 100

    def test_invalid_organization_raises(self) -> None:
        with pytest.raises(ValueError, match="agent_organization"):
            ExperimentConfig(agent_organization="swarm")

    def test_invalid_representation_raises(self) -> None:
        with pytest.raises(ValueError, match="representation"):
            ExperimentConfig(representation="quantum")

    def test_invalid_emergence_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="emergence_mode"):
            ExperimentConfig(emergence_mode="magic")

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(ValueError, match="log_level"):
            ExperimentConfig(log_level="VERBOSE")


class TestCombinationGuardrails:
    """Degenerate (rep × loop × paradigm) triple warnings."""

    def _make_cfg(self, loop: str, ops: str, rep_type: str) -> ExperimentConfig:
        return ExperimentConfig(
            decision_loop=loop,
            operations_paradigm=ops,
            representation_config={"type": rep_type},
        )

    def test_deterministic_rep_ground_non_sda_warns(self) -> None:
        """react + conventional_ground + deterministic rep → warning."""
        with pytest.warns(UserWarning, match="deterministic representation"):
            self._make_cfg("react", "conventional_ground", "conventional_schedule_eventsat")

    def test_ooda_ground_deterministic_warns(self) -> None:
        """ooda + autonomous_ground + schedule_based → warning."""
        with pytest.warns(UserWarning, match="deterministic representation"):
            self._make_cfg("ooda", "autonomous_ground", "schedule_based_eventsat")

    def test_sda_ground_no_warning(self) -> None:
        """sda + conventional_ground + deterministic rep → no warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._make_cfg("sda", "conventional_ground", "conventional_schedule_eventsat")

    def test_sda_ah_no_warning(self) -> None:
        """sda + autonomous_hybrid + rule_based → no warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._make_cfg("sda", "autonomous_hybrid", "rule_based_eventsat")

    def test_non_deterministic_rep_ground_no_warning(self) -> None:
        """react + conventional_ground + future LLM rep → no warning."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._make_cfg("react", "conventional_ground", "llm_eventsat")

    def test_human_rep_on_autonomous_hybrid_warns(self) -> None:
        """conventional_schedule_eventsat + autonomous_hybrid → warning."""
        with pytest.warns(UserWarning, match="human cognitive constraints"):
            self._make_cfg("sda", "autonomous_hybrid", "conventional_schedule_eventsat")

    def test_warnings_non_blocking(self) -> None:
        """Degenerate config still loads successfully."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = self._make_cfg("react", "conventional_ground", "schedule_based_eventsat")
        assert cfg.decision_loop == "react"
        assert cfg.operations_paradigm == "conventional_ground"


class TestEmergenceMechanism:
    """Validation of emergence_config.mechanism cross-field constraints."""

    def _make_learned(
        self,
        representation: str,
        repr_type: str,
        mechanism: str | None,
    ) -> ExperimentConfig:
        import warnings
        emergence_config: dict = {"mode": "learned"}
        if mechanism is not None:
            emergence_config["mechanism"] = mechanism
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ExperimentConfig(
                representation=representation,
                representation_config={"type": repr_type},
                emergence_mode="learned",
                emergence_config=emergence_config,
            )

    def test_ppo_with_subsymbolic_valid(self) -> None:
        cfg = self._make_learned("subsymbolic", "subsymbolic_eventsat", "ppo")
        assert cfg.emergence_config["mechanism"] == "ppo"

    def test_prompt_optimized_with_hybrid_valid(self) -> None:
        cfg = self._make_learned("hybrid", "llm_eventsat", "prompt_optimized")
        assert cfg.emergence_config["mechanism"] == "prompt_optimized"

    def test_writable_coala_with_agentic_valid(self) -> None:
        cfg = self._make_learned("hybrid", "agentic_eventsat", "writable_coala")
        assert cfg.emergence_config["mechanism"] == "writable_coala"

    def test_invalid_mechanism_raises(self) -> None:
        with pytest.raises(ValueError, match="mechanism"):
            self._make_learned("subsymbolic", "subsymbolic_eventsat", "neural_evolution")

    def test_ppo_with_hybrid_raises(self) -> None:
        with pytest.raises(ValueError, match="mechanism.*ppo"):
            self._make_learned("hybrid", "llm_eventsat", "ppo")

    def test_prompt_optimized_with_subsymbolic_raises(self) -> None:
        with pytest.raises(ValueError, match="mechanism.*prompt_optimized"):
            self._make_learned("subsymbolic", "subsymbolic_eventsat", "prompt_optimized")

    def test_writable_coala_with_non_agentic_raises(self) -> None:
        with pytest.raises(ValueError, match="writable_coala.*agentic_eventsat"):
            self._make_learned("hybrid", "llm_eventsat", "writable_coala")

    def test_learned_hybrid_no_mechanism_warns(self) -> None:
        with pytest.warns(UserWarning, match="mechanism"):
            ExperimentConfig(
                representation="hybrid",
                representation_config={"type": "llm_eventsat"},
                emergence_mode="learned",
                emergence_config={"mode": "learned"},
            )

    def test_hand_designed_no_mechanism_no_warning(self) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ExperimentConfig(
                representation="hybrid",
                representation_config={"type": "llm_eventsat"},
                emergence_mode="hand_designed",
                emergence_config={"mode": "hand_designed"},
            )


class TestConfigLoaderSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        original = ExperimentConfig(
            experiment_id="test_round_trip",
            seed=123,
            agent_organization="decentralized_mas",
        )
        yaml_path = tmp_path / "test.yaml"
        save_config(original, yaml_path)

        loaded = load_config(yaml_path)
        assert loaded.experiment_id == "test_round_trip"
        assert loaded.seed == 123
        assert loaded.agent_organization == "decentralized_mas"

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_template_variable_resolution(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "test.yaml"
        data = {
            "experiment_id": "exp_001",
            "output_dir": "data/results/${experiment_id}",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        cfg = load_config(yaml_path)
        assert cfg.output_dir == "data/results/exp_001"


# ======================================================================
# Memory tests
# ======================================================================


class TestFixedMemory:
    def test_reset(self) -> None:
        mem = FixedMemory()
        mem.update("constellation_state", {"sat_0": "ok"})
        mem.reset()
        assert mem.query("constellation_state") == {}

    def test_update_and_query(self) -> None:
        mem = FixedMemory()
        mem.update("task_queue", [{"task": "observe"}])
        assert mem.query("task_queue") == [{"task": "observe"}]

    def test_history_sliding_window(self) -> None:
        mem = FixedMemory(config={"history_depth": 3})
        for i in range(5):
            mem.update("constellation_state", {"step": i})
        history = mem.query("history")
        # Depth 3 → only last 3 previous states (steps 1, 2, 3)
        assert len(history) == 3

    def test_unknown_key_raises(self) -> None:
        mem = FixedMemory()
        with pytest.raises(KeyError):
            mem.update("nonexistent", {})


# ======================================================================
# Analysis tests
# ======================================================================


class TestParetoFrontier:
    def test_simple_pareto(self) -> None:
        points = [
            {"utility": 10, "latency": 5},
            {"utility": 8, "latency": 3},
            {"utility": 6, "latency": 2},
            {"utility": 5, "latency": 8},
        ]
        # Maximise utility, minimise latency
        frontier = compute_pareto_frontier(
            points,
            objectives=["utility", "latency"],
            maximise=[True, False],
        )
        # Point 0 (10, 5) and point 2 (6, 2) are Pareto-optimal
        assert 0 in frontier
        assert 2 in frontier
        # Point 3 (5, 8) is dominated
        assert 3 not in frontier

    def test_empty_input(self) -> None:
        assert compute_pareto_frontier([], ["a"]) == []


class TestCompareExperiments:
    def test_ranking(self) -> None:
        stats = [
            {"experiment_id": "a", "mean": {"utility": 10.0}, "std": {"utility": 1.0}},
            {"experiment_id": "b", "mean": {"utility": 15.0}, "std": {"utility": 2.0}},
        ]
        result = compare_experiments(stats, "utility")
        assert result["ranking"][0]["experiment_id"] == "b"


# ======================================================================
# Experiment runner (smoke test)
# ======================================================================


class TestExperimentRunner:
    def test_init_from_config_object(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            experiment_id="smoke_test",
            num_episodes=1,
            max_steps=2,
            output_dir=str(tmp_path),
        )
        runner = ExperimentRunner(config=cfg)
        assert runner.config.experiment_id == "smoke_test"

    def test_init_requires_config(self) -> None:
        with pytest.raises(ValueError):
            ExperimentRunner()

    def test_run_smoke(self, tmp_path: Path) -> None:
        """Run a minimal experiment with placeholder components."""
        cfg = ExperimentConfig(
            experiment_id="smoke_test",
            num_episodes=1,
            max_steps=2,
            output_dir=str(tmp_path / "results"),
        )
        runner = ExperimentRunner(config=cfg)
        results = runner.run()

        assert results["experiment_id"] == "smoke_test"
        assert results["num_episodes"] == 1
        assert (tmp_path / "results" / "results.json").exists()
        assert (tmp_path / "results" / "config.json").exists()

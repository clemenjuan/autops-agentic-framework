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
        assert cfg.decision_procedure == "sda"
        assert cfg.representation == "symbolic"
        assert cfg.behaviour == "hand_designed"
        assert cfg.num_episodes == 100

    def test_invalid_organization_raises(self) -> None:
        with pytest.raises(ValueError, match="agent_organization"):
            ExperimentConfig(agent_organization="swarm")

    def test_invalid_representation_raises(self) -> None:
        with pytest.raises(ValueError, match="representation"):
            ExperimentConfig(representation="quantum")

    def test_invalid_emergence_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="behaviour"):
            ExperimentConfig(behaviour="magic")

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(ValueError, match="log_level"):
            ExperimentConfig(log_level="VERBOSE")


class TestCombinationGuardrails:
    """Degenerate (rep × loop × paradigm) triple warnings."""

    def _make_cfg(self, loop: str, ops: str, rep_type: str) -> ExperimentConfig:
        return ExperimentConfig(
            decision_procedure=loop,
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

    def test_non_deterministic_scheduler_ground_no_warning(self) -> None:
        """react + conventional_ground + LLM scheduler placeholder → no warning.

        The scheduler placeholders emit a schedule, so they are valid under
        ground paradigms and (being non-deterministic family stand-ins) do not
        trip the deterministic-rep warning.
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._make_cfg("react", "conventional_ground", "llm_scheduler_eventsat")

    def test_raw_llm_rep_under_ground_raises(self) -> None:
        """A non-schedule representation under a ground paradigm is now rejected."""
        with pytest.raises(ValueError, match="schedule-producing"):
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
        assert cfg.decision_procedure == "react"
        assert cfg.operations_paradigm == "conventional_ground"


class TestEmergenceMechanism:
    """Validation of behaviour_config.mechanism cross-field constraints."""

    def _make_learned(
        self,
        representation: str,
        repr_type: str,
        mechanism: str | None,
    ) -> ExperimentConfig:
        import warnings
        behaviour_config: dict = {"mode": "emergent"}
        if mechanism is not None:
            behaviour_config["mechanism"] = mechanism
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ExperimentConfig(
                representation=representation,
                representation_config={"type": repr_type},
                behaviour="emergent",
                behaviour_config=behaviour_config,
            )

    def test_ppo_with_subsymbolic_valid(self) -> None:
        cfg = self._make_learned("subsymbolic", "subsymbolic_eventsat", "ppo")
        assert cfg.behaviour_config["mechanism"] == "ppo"

    def test_prompt_optimized_with_hybrid_valid(self) -> None:
        cfg = self._make_learned("hybrid", "llm_eventsat", "prompt_optimized")
        assert cfg.behaviour_config["mechanism"] == "prompt_optimized"

    def test_writable_coala_with_agentic_valid(self) -> None:
        cfg = self._make_learned("hybrid", "agentic_eventsat", "writable_coala")
        assert cfg.behaviour_config["mechanism"] == "writable_coala"

    def test_invalid_mechanism_raises(self) -> None:
        with pytest.raises(ValueError, match="mechanism"):
            self._make_learned("subsymbolic", "subsymbolic_eventsat", "neural_evolution")

    def test_explicit_hand_designed_mechanism_accepted(self) -> None:
        """`mechanism: hand_designed` is accepted as 'no learned mechanism'."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = ExperimentConfig(
                representation="symbolic",
                representation_config={"type": "rule_based_eventsat"},
                behaviour="hand_designed",
                behaviour_config={"mode": "hand_designed", "mechanism": "hand_designed"},
            )
        assert cfg.behaviour_config["mechanism"] == "hand_designed"

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
                behaviour="emergent",
                behaviour_config={"mode": "emergent"},
            )

    def test_hand_designed_no_mechanism_no_warning(self) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ExperimentConfig(
                representation="hybrid",
                representation_config={"type": "llm_eventsat"},
                behaviour="hand_designed",
                behaviour_config={"mode": "hand_designed"},
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


class TestRunnerMemoryWiring:
    """Regression: the runner is the source of truth for the memory object.

    writable_coala (``_lec_``) configs must receive a WritableMemory, otherwise
    every CoALA memory write silently no-ops against a FixedMemory and the arm
    becomes indistinguishable from the fixed-memory baseline.
    """

    def test_writable_coala_gets_writable_memory(self, tmp_path: Path) -> None:
        from src.memory.writable_memory import WritableMemory

        cfg = ExperimentConfig(
            experiment_id="lec_mem",
            num_episodes=1,
            max_steps=2,
            output_dir=str(tmp_path),
            representation="hybrid",
            representation_config={"type": "agentic_eventsat"},
            behaviour="emergent",
            behaviour_config={"mode": "emergent", "mechanism": "writable_coala"},
        )
        runner = ExperimentRunner(config=cfg)
        mem = runner._create_memory()
        assert isinstance(mem, WritableMemory)
        assert hasattr(mem, "write_semantic_rule")

    def test_default_gets_fixed_memory(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            experiment_id="hd_mem",
            num_episodes=1,
            max_steps=2,
            output_dir=str(tmp_path),
        )
        runner = ExperimentRunner(config=cfg)
        mem = runner._create_memory()
        assert isinstance(mem, FixedMemory)


class TestDeferredOrganizationGuard:
    """Deferred MAS variants must fail early with an actionable message."""

    @pytest.mark.parametrize(
        "org", ["decentralized_mas", "independent_mas", "hybrid_mas"]
    )
    def test_deferred_org_raises_not_implemented(self, org: str, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            experiment_id=f"deferred_{org}",
            num_episodes=1,
            max_steps=2,
            output_dir=str(tmp_path),
            agent_organization=org,
        )
        runner = ExperimentRunner(config=cfg)
        with pytest.raises(NotImplementedError, match="deferred to Flamingo"):
            runner._create_organization()


class TestMatrixRestructureNaming:
    """New axis names + action_space semantics; legacy aliases during migration."""

    def test_new_field_names(self) -> None:
        cfg = ExperimentConfig(decision_procedure="ooda", behaviour="emergent",
                               representation="subsymbolic",
                               representation_config={"type": "subsymbolic_eventsat"},
                               behaviour_config={"mechanism": "ppo"})
        assert cfg.decision_procedure == "ooda"
        assert cfg.behaviour == "emergent"

    @pytest.mark.parametrize(
        "legacy_kwargs",
        [
            {"decision_loop": "react"},
            {"emergence_mode": "hand_designed"},
            {"emergence_config": {"mode": "hand_designed"}},
            {"decision_loop_config": {"x": 1}},
        ],
    )
    def test_legacy_field_names_rejected(self, legacy_kwargs: dict) -> None:
        """Old field names no longer accepted (extra='forbid')."""
        with pytest.raises(ValueError):
            ExperimentConfig(**legacy_kwargs)

    def test_action_space_valid(self) -> None:
        cfg = ExperimentConfig(representation="hybrid",
                               representation_config={"type": "agentic_eventsat",
                                                      "action_space": "agentic"})
        assert cfg.action_space == "agentic"

    def test_action_space_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="action_space"):
            ExperimentConfig(representation="hybrid",
                             representation_config={"action_space": "reflexive"})

    def test_agentic_requires_hybrid(self) -> None:
        with pytest.raises(ValueError, match="agentic.*hybrid"):
            ExperimentConfig(representation="subsymbolic",
                             representation_config={"action_space": "agentic"})

    def test_action_space_must_agree_with_type(self) -> None:
        with pytest.raises(ValueError, match="reactive.*action_space"):
            ExperimentConfig(representation="hybrid",
                             representation_config={"type": "llm_eventsat",
                                                    "action_space": "agentic"})

    def test_action_space_derived_from_type_when_absent(self) -> None:
        assert ExperimentConfig(representation="hybrid",
                                representation_config={"type": "llm_eventsat"}).action_space == "reactive"
        assert ExperimentConfig(representation="hybrid",
                                representation_config={"type": "agentic_eventsat"}).action_space == "agentic"
        assert ExperimentConfig(representation="symbolic",
                                representation_config={"type": "rule_based_eventsat"}).action_space is None

    def test_writable_coala_requires_agentic_action_space(self) -> None:
        import warnings
        # agentic type + agentic action_space + writable_coala is valid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = ExperimentConfig(representation="hybrid", behaviour="emergent",
                                   representation_config={"type": "agentic_eventsat",
                                                          "action_space": "agentic"},
                                   behaviour_config={"mechanism": "writable_coala"})
        assert cfg.behaviour_config["mechanism"] == "writable_coala"


class TestRepresentationResolution:
    """`type` is resolved from (representation, action_space, ops); explicit type overrides."""

    @pytest.mark.parametrize(
        "rep, action_space, ops, expected",
        [
            ("symbolic", None, "autonomous_hybrid", "rule_based_eventsat"),
            ("symbolic", None, "autonomous_ground", "schedule_based_eventsat"),
            ("symbolic", None, "conventional_ground", "conventional_schedule_eventsat"),
            ("subsymbolic", None, "autonomous_hybrid", "subsymbolic_eventsat"),
            ("subsymbolic", None, "autonomous_ground", "subsymbolic_scheduler_eventsat"),
            ("hybrid", "reactive", "autonomous_hybrid", "llm_eventsat"),
            ("hybrid", "reactive", "conventional_ground", "llm_scheduler_eventsat"),
            ("hybrid", "agentic", "autonomous_hybrid", "agentic_eventsat"),
            ("hybrid", "agentic", "autonomous_ground", "agentic_scheduler_eventsat"),
        ],
    )
    def test_resolution(self, rep, action_space, ops, expected) -> None:
        import warnings
        rc = {"action_space": action_space} if action_space else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = ExperimentConfig(representation=rep, operations_paradigm=ops,
                                   representation_config=rc)
        assert cfg.resolved_representation_type == expected

    def test_hybrid_without_action_space_raises(self) -> None:
        with pytest.raises(ValueError, match="action_space"):
            ExperimentConfig(representation="hybrid", operations_paradigm="autonomous_hybrid")

    def test_explicit_type_overrides(self) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = ExperimentConfig(representation="symbolic",
                                   operations_paradigm="conventional_ground",
                                   representation_config={"type": "schedule_based_eventsat"})
        assert cfg.resolved_representation_type == "schedule_based_eventsat"  # not conventional_*


class TestAutonomousOnboard:
    """autonomous_onboard paradigm: onboard-only, resolves to the per-step core."""

    @pytest.mark.parametrize("rep, expected", [
        ("symbolic", "rule_based_eventsat"),
        ("subsymbolic", "subsymbolic_eventsat"),
    ])
    def test_ao_resolves_onboard_core(self, rep, expected) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = ExperimentConfig(representation=rep, operations_paradigm="autonomous_onboard")
        assert cfg.resolved_representation_type == expected

    def test_hybrid_onboard_excluded(self) -> None:
        with pytest.raises(ValueError, match="autonomous_onboard"):
            ExperimentConfig(representation="hybrid", operations_paradigm="autonomous_onboard",
                             representation_config={"action_space": "agentic"})

    def test_paradigm_is_passthrough_onboard(self) -> None:
        from src.operations.autonomous_onboard import AutonomousOnboard
        ao = AutonomousOnboard()
        act = {"eventsat_0": {"mode": "payload_observe"}}
        assert ao.filter_observation("OBS", 0) == "OBS"          # real-time
        assert ao.can_act(0, ground_pass_active=False) is True   # acts every step
        assert ao.should_allow_inference(0, False) is True
        assert ao.can_self_recover_anomaly() is True             # onboard FDIR
        assert ao.process_action(act, 0, False) == act           # pass-through, no schedule

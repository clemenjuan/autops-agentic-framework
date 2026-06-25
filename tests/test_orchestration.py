"""
Tests for the Orchestration layer: config loading, experiment runner, metrics.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.core.memory.fixed_memory import FixedMemory
from src.core.config_loader import (
    ExperimentConfig,
    EnvironmentConfig,
    MetricsConfig,
    load_config,
    save_config,
)
from src.core.experiment_runner import ExperimentRunner


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

    def test_invalid_behaviour_mode_raises(self) -> None:
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
            environment={"scenario": "eventsat"},
        )

    @pytest.mark.parametrize("loop", ["ooda", "react"])
    def test_retired_decision_loops_rejected(self, loop: str) -> None:
        with pytest.raises(ValueError, match="decision_procedure must be 'sda'"):
            self._make_cfg(loop, "conventional_ground", "conventional_schedule_eventsat")

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

    def test_raw_llm_rep_under_ground_raises(self) -> None:
        """A non-schedule representation under a ground paradigm is now rejected."""
        with pytest.raises(ValueError, match="schedule-producing"):
            self._make_cfg("sda", "conventional_ground", "llm_eventsat")

    def test_human_rep_on_autonomous_hybrid_warns(self) -> None:
        """conventional_schedule_eventsat + autonomous_hybrid → warning."""
        with pytest.warns(UserWarning, match="human cognitive constraints"):
            self._make_cfg("sda", "autonomous_hybrid", "conventional_schedule_eventsat")

    def test_sda_ground_schedule_based_loads(self) -> None:
        cfg = self._make_cfg("sda", "conventional_ground", "schedule_based_eventsat")
        assert cfg.decision_procedure == "sda"
        assert cfg.operations_paradigm == "conventional_ground"


class TestBehaviourMechanism:
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
            environment={"constellation_size": 12},
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
        from src.core.memory.writable_memory import WritableMemory

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


class TestOrganizationInstantiation:
    """All five Kim et al. organisations instantiate and are runnable."""

    @pytest.mark.parametrize(
        "org,expected_agents",
        [
            ("sas", 1),
            ("centralized_mas", 4),  # mission_manager + 3 locals
            ("independent_mas", 3),
            ("decentralized_mas", 3),
            ("hybrid_mas", 2),       # default num_clusters = 2
        ],
    )
    def test_org_is_runnable(self, org: str, expected_agents: int, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            experiment_id=f"{org}_runnable",
            num_episodes=1,
            max_steps=2,
            output_dir=str(tmp_path),
            agent_organization=org,
            environment={"scenario": "ssa", "constellation_size": 3},
        )
        runner = ExperimentRunner(config=cfg)
        organization = runner._create_organization()
        assert len(organization.get_agents()) == expected_agents


class TestConfigSchema:
    """Current config schema and action_space semantics."""

    def test_new_field_names(self) -> None:
        cfg = ExperimentConfig(decision_procedure="sda", behaviour="emergent",
                               representation="subsymbolic",
                               representation_config={"type": "subsymbolic_eventsat"},
                               behaviour_config={"mechanism": "ppo"})
        assert cfg.decision_procedure == "sda"
        assert cfg.behaviour == "emergent"

    @pytest.mark.parametrize(
        "removed_kwargs",
        [
            {"decision_loop": "react"},
            {"emergence_mode": "hand_designed"},
            {"emergence_config": {"mode": "hand_designed"}},
            {"decision_loop_config": {"x": 1}},
        ],
    )
    def test_removed_field_names_rejected(self, removed_kwargs: dict) -> None:
        """Old field names are rejected explicitly."""
        with pytest.raises(ValueError):
            ExperimentConfig(**removed_kwargs)

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
        from src.core.operations.autonomous_onboard import AutonomousOnboard
        ao = AutonomousOnboard()
        act = {"eventsat_0": {"mode": "payload_observe"}}
        assert ao.filter_observation("OBS", 0) == "OBS"          # real-time
        assert ao.can_act(0, ground_pass_active=False) is True   # acts every step
        assert ao.should_allow_inference(0, False) is True
        assert ao.can_self_recover_anomaly() is True             # onboard FDIR
        assert ao.process_action(act, 0, False) == act           # pass-through, no schedule

    def test_self_recovery_capability_matches_paradigm(self) -> None:
        """Onboard paradigms (AO/AH) self-recover anomalies; ground paradigms
        (AG/CG) require a ground pass. This capability is the single source of
        truth the runner uses to set env.anomaly_requires_ground_pass."""
        from src.core.operations.autonomous_onboard import AutonomousOnboard
        from src.core.operations.autonomous_hybrid import AutonomousHybrid
        from src.core.operations.autonomous_ground import AutonomousGround
        from src.core.operations.conventional_ground import ConventionalGround
        assert AutonomousOnboard().can_self_recover_anomaly() is True
        assert AutonomousHybrid().can_self_recover_anomaly() is True
        assert AutonomousGround().can_self_recover_anomaly() is False
        assert ConventionalGround().can_self_recover_anomaly() is False


class TestTwoCoreResolution:
    """resolved_onboard_type + resolved_ground_planner_type per (substrate, action_space, ops)."""

    @pytest.mark.parametrize(
        "rep, action_space, ops, onboard, ground",
        [
            # AO: onboard only
            ("symbolic", None, "autonomous_onboard", "rule_based_eventsat", None),
            ("subsymbolic", None, "autonomous_onboard", "subsymbolic_eventsat", None),
            # AG/CG: ground only
            ("symbolic", None, "autonomous_ground", None, "schedule_based_eventsat"),
            ("symbolic", None, "conventional_ground", None, "conventional_schedule_eventsat"),
            ("subsymbolic", None, "autonomous_ground", None, "subsymbolic_scheduler_eventsat"),
            ("hybrid", "reactive", "autonomous_ground", None, "llm_scheduler_eventsat"),
            # AH: both; ground = AG-equivalent (algorithmic), onboard = per-step
            ("symbolic", None, "autonomous_hybrid", "rule_based_eventsat", "schedule_based_eventsat"),
            ("subsymbolic", None, "autonomous_hybrid", "subsymbolic_eventsat", "subsymbolic_scheduler_eventsat"),
            # Onboard slot follows the configured substrate and is never silently
            # substituted by RL.
            ("hybrid", "reactive", "autonomous_hybrid", "llm_eventsat", "llm_scheduler_eventsat"),
            ("hybrid", "agentic", "autonomous_hybrid", "agentic_eventsat", "agentic_scheduler_eventsat"),
        ],
    )
    def test_two_core_resolution(self, rep, action_space, ops, onboard, ground) -> None:
        import warnings
        rc = {"action_space": action_space} if action_space else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = ExperimentConfig(representation=rep, operations_paradigm=ops,
                                   representation_config=rc)
        assert cfg.resolved_onboard_type == onboard
        assert cfg.resolved_ground_planner_type == ground


class TestRepresentationVocabulary:
    """7-cell framework tokens (morphological_matrix.md §2) normalise to the
    internal substrate + action_space; HRL and pure-LLM onboard route to flagged placeholders."""

    @pytest.mark.parametrize(
        "cell, ops, expected",
        [
            ("symb", "autonomous_hybrid", "rule_based_eventsat"),
            ("rl", "autonomous_hybrid", "subsymbolic_eventsat"),
            ("hrl", "autonomous_hybrid", "hrl_onboard_eventsat"),
            ("hrl", "autonomous_ground", "hrl_scheduler_eventsat"),
            ("llm-s", "autonomous_hybrid", "llm_single_onboard_eventsat"),
            ("llm-s", "autonomous_ground", "llm_single_scheduler_eventsat"),
            ("llm-a", "autonomous_ground", "llm_agentic_scheduler_eventsat"),
            ("hllm-s", "autonomous_hybrid", "llm_eventsat"),
            ("hllm-a", "autonomous_hybrid", "agentic_eventsat"),
        ],
    )
    def test_cell_resolves(self, cell, ops, expected) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = ExperimentConfig(representation=cell, operations_paradigm=ops)
        assert cfg.representation_cell == cell
        assert cfg.resolved_representation_type == expected

    def test_placeholder_cells_flagged(self) -> None:
        import src.eventsat.placeholders  # noqa: F401  (registers cells)
        from src.core.behaviour.controller import _REPRESENTATION_REGISTRY

        # Real cores (NOT placeholders): the LLM ground schedulers — single-shot
        # llm_single_scheduler_eventsat (llm-s) / llm_scheduler_eventsat (hllm-s) and
        # agentic llm_agentic_scheduler_eventsat (llm-a) / agentic_scheduler_eventsat (hllm-a).
        for name in (
            "hrl_onboard_eventsat", "hrl_scheduler_eventsat",
            "llm_single_onboard_eventsat", "llm_agentic_onboard_eventsat",
        ):
            assert _REPRESENTATION_REGISTRY[name].is_placeholder is True

    def test_cell_matches_expanded_equivalent(self) -> None:
        """hllm-a must resolve identically to expanded hybrid+agentic."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cell = ExperimentConfig(representation="hllm-a",
                                    operations_paradigm="autonomous_hybrid")
            expanded = ExperimentConfig(representation="hybrid",
                                      operations_paradigm="autonomous_hybrid",
                                      representation_config={"action_space": "agentic"})
        assert cell.representation == "hybrid"
        assert cell.representation_config.get("action_space") == "agentic"
        assert cell.resolved_onboard_type == expanded.resolved_onboard_type == "agentic_eventsat"
        assert cell.resolved_ground_planner_type == expanded.resolved_ground_planner_type
        assert cell.onboard_uses_jetson == expanded.onboard_uses_jetson

    def test_substrate_value_still_accepted(self) -> None:
        """Substrate values keep working (representation_cell stays None)."""
        cfg = ExperimentConfig(representation="symbolic",
                               operations_paradigm="autonomous_onboard")
        assert cfg.representation == "symbolic"
        assert cfg.representation_cell is None


class TestDualCoreAH:
    """Dual-core AH: independent onboard + ground core blocks (the ah_<onboard>_<ground>
    pairs). Onboard ∈ {symb, rl, hrl}; ground ∈ the 7 cells."""

    def _cfg(self, **kw):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ExperimentConfig(operations_paradigm="autonomous_hybrid", **kw)

    @pytest.mark.parametrize(
        "onboard_rep, ground_rep, exp_onboard, exp_ground, jetson",
        [
            ("rl",   "symb",   "subsymbolic_eventsat", "schedule_based_eventsat",    True),
            ("symb", "hllm-a", "rule_based_eventsat",  "agentic_scheduler_eventsat", False),
            ("symb", "hllm-s", "rule_based_eventsat",  "llm_scheduler_eventsat",     False),
            ("hrl",  "rl",     "hrl_onboard_eventsat", "subsymbolic_scheduler_eventsat", True),
        ],
    )
    def test_dual_core_resolution(self, onboard_rep, ground_rep, exp_onboard, exp_ground, jetson) -> None:
        cfg = self._cfg(onboard={"representation": onboard_rep},
                        ground={"representation": ground_rep})
        assert cfg.resolved_onboard_type == exp_onboard
        assert cfg.resolved_ground_planner_type == exp_ground
        assert cfg.onboard_uses_jetson is jetson

    def test_per_core_configs_independent(self) -> None:
        cfg = self._cfg(
            onboard={"representation": "rl", "representation_config": {"rl_mock": True}},
            ground={"representation": "hllm-a", "representation_config": {"llm_model": "x"}},
        )
        assert cfg.onboard_representation_config == {"rl_mock": True}
        assert cfg.ground_representation_config.get("llm_model") == "x"
        assert cfg.ground_representation_config.get("action_space") == "agentic"  # injected from cell

    def test_llm_onboard_rejected(self) -> None:
        with pytest.raises(ValueError, match="onboard-feasible"):
            self._cfg(onboard={"representation": "hllm-a"}, ground={"representation": "symb"})

    def test_cores_require_autonomous_hybrid(self) -> None:
        with pytest.raises(ValueError, match="autonomous_hybrid"):
            ExperimentConfig(operations_paradigm="autonomous_ground",
                             onboard={"representation": "symb"}, ground={"representation": "symb"})

    def test_both_cores_required(self) -> None:
        with pytest.raises(ValueError, match="BOTH"):
            self._cfg(onboard={"representation": "symb"})

    def test_ppo_mechanism_valid_with_rl_onboard(self) -> None:
        cfg = self._cfg(
            behaviour="emergent",
            onboard={"representation": "rl", "representation_config": {"rl_mock": True}},
            ground={"representation": "symb"},
            behaviour_config={"mechanism": "ppo"},
        )
        assert cfg.behaviour_config["mechanism"] == "ppo"

    def test_single_rep_ah_backward_compatible(self) -> None:
        cfg = self._cfg(representation="symb")
        assert cfg.resolved_onboard_type == "rule_based_eventsat"
        assert cfg.resolved_ground_planner_type == "schedule_based_eventsat"
        assert cfg.onboard is None and cfg.ground is None

    def test_example_configs_load(self) -> None:
        from src.core.config_loader import load_config
        from pathlib import Path
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = load_config(Path("configs/experiments/eventsat_sas_ah_rl_symb.yaml"))
            b = load_config(Path("configs/experiments/eventsat_sas_ah_symb_hllm-a.yaml"))
        assert a.resolved_onboard_type == "subsymbolic_eventsat"
        assert a.resolved_ground_planner_type == "schedule_based_eventsat"
        assert b.resolved_onboard_type == "rule_based_eventsat"
        assert b.resolved_ground_planner_type == "agentic_scheduler_eventsat"


class TestAutonomousHybridArbitration:
    """Dual-slot AH: plan-default between passes, onboard override on safety modes."""

    def _ah_with_plan(self):
        from src.core.operations.autonomous_hybrid import AutonomousHybrid
        ah = AutonomousHybrid()
        ah.set_uplinked_plan(
            {"eventsat_0": {"schedule": [("payload_observe", 3), ("charging", 2)]}}
        )
        return ah

    def test_follows_plan_when_onboard_not_safety(self) -> None:
        ah = self._ah_with_plan()
        out = ah.process_action(
            {"eventsat_0": {"mode": "payload_compress"}}, step=1, ground_pass_active=False
        )
        assert out["eventsat_0"]["mode"] == "payload_observe"  # plan, not onboard
        assert ah._onboard_overrides == 0

    def test_onboard_overrides_on_safety_mode(self) -> None:
        ah = self._ah_with_plan()
        out = ah.process_action(
            {"eventsat_0": {"mode": "charging"}}, step=1, ground_pass_active=False
        )
        assert out["eventsat_0"]["mode"] == "charging"  # safety override of the plan
        assert ah._onboard_overrides == 1

    def test_onboard_wins_during_pass(self) -> None:
        ah = self._ah_with_plan()
        out = ah.process_action(
            {"eventsat_0": {"mode": "communication"}}, step=1, ground_pass_active=True
        )
        assert out["eventsat_0"]["mode"] == "communication"  # real-time during contact

    def test_no_plan_falls_back_to_onboard(self) -> None:
        from src.core.operations.autonomous_hybrid import AutonomousHybrid
        ah = AutonomousHybrid()
        out = ah.process_action(
            {"eventsat_0": {"mode": "payload_observe"}}, step=1, ground_pass_active=False
        )
        assert out["eventsat_0"]["mode"] == "payload_observe"  # no plan → onboard
        assert ah.get_metrics()["onboard_overrides"] == 0.0


class TestOnboardUsesJetson:
    """Jetson overhead applies only to Jetson-based onboard (subsymbolic/hybrid AO/AH)."""

    @pytest.mark.parametrize(
        "rep, action_space, ops, expected",
        [
            ("symbolic", None, "autonomous_onboard", False),   # rules on OBC
            ("subsymbolic", None, "autonomous_onboard", True),  # RL on Jetson
            ("symbolic", None, "autonomous_hybrid", False),     # rule_based onboard on OBC
            ("subsymbolic", None, "autonomous_hybrid", True),
            ("hybrid", "agentic", "autonomous_hybrid", True),   # subsymbolic onboard on Jetson
            ("symbolic", None, "autonomous_ground", False),     # ground → no onboard
            ("subsymbolic", None, "autonomous_ground", False),
            ("hybrid", "reactive", "conventional_ground", False),
        ],
    )
    def test_onboard_uses_jetson(self, rep, action_space, ops, expected) -> None:
        import warnings
        rc = {"action_space": action_space} if action_space else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = ExperimentConfig(representation=rep, operations_paradigm=ops,
                                   representation_config=rc)
        assert cfg.onboard_uses_jetson is expected

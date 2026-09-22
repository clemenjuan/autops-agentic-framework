"""
Tests for the Orchestration layer: config loading, experiment runner, metrics.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

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

    def test_conflicting_explicit_max_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            ExperimentConfig(max_steps=100, environment={"max_steps": 10080})

    def test_top_level_max_steps_synchronizes_environment(self) -> None:
        cfg = ExperimentConfig(max_steps=321)
        assert cfg.max_steps == 321
        assert cfg.environment.max_steps == 321

    def test_environment_max_steps_synchronizes_runner_length(self) -> None:
        cfg = ExperimentConfig(environment={"max_steps": 654})
        assert cfg.max_steps == 654
        assert cfg.environment.max_steps == 654

    def test_ssa_rl_resolves_native_controller_without_explicit_type(self) -> None:
        cfg = ExperimentConfig(
            representation="rl",
            representation_config={"rl_mock": True},
            operations_paradigm="autonomous_onboard",
            environment={"scenario": "ssa", "constellation_size": 1},
        )

        assert cfg.resolved_representation_type == "subsymbolic_ssa"
        assert cfg.resolved_onboard_type == "subsymbolic_ssa"

    @pytest.mark.parametrize("unknown", ["steps", "num_steps"])
    def test_unknown_top_level_step_fields_raise(self, unknown: str) -> None:
        with pytest.raises(ValueError, match=unknown):
            ExperimentConfig(**{unknown: 100})

    @pytest.mark.parametrize(
        ("reserved", "bad_value"),
        [
            ("max_steps", 3),
            ("step_duration_s", 30),
            ("timestep_seconds", 30),
            ("constellation_size", 2),
            ("scenario", "ssa"),
            ("seed", 99),
        ],
    )
    def test_conflicting_scenario_config_dimensions_raise(
        self, reserved: str, bad_value: object
    ) -> None:
        with pytest.raises(ValueError, match=f"scenario_config.{reserved}"):
            ExperimentConfig(
                seed=42,
                max_steps=10,
                environment={
                    "scenario": "eventsat",
                    "constellation_size": 1,
                    "timestep_seconds": 60,
                    "max_steps": 10,
                    "scenario_config": {reserved: bad_value},
                },
            )

    def test_equal_redundant_scenario_config_dimensions_pass(self) -> None:
        cfg = ExperimentConfig(
            seed=42,
            max_steps=10,
            environment={
                "scenario": "eventsat",
                "constellation_size": 1,
                "timestep_seconds": 60,
                "max_steps": 10,
                "scenario_config": {
                    "max_steps": 10,
                    "step_duration_s": 60,
                    "timestep_seconds": 60,
                    "constellation_size": 1,
                    "scenario": "eventsat",
                    "seed": 42,
                },
            },
        )

        runner = ExperimentRunner(config=cfg)
        env = runner._create_environment()
        assert env.max_steps == 10
        assert env.step_duration_s == 60

    def test_runner_passes_discount_factor_to_reward_function(self) -> None:
        cfg = ExperimentConfig(
            seed=42,
            max_steps=10,
            behaviour_config={"gamma": 0.91},
            environment={
                "scenario": "eventsat",
                "constellation_size": 1,
                "timestep_seconds": 60,
                "max_steps": 10,
                "scenario_config": {
                    "reward_config": {"pipeline_shaping": {"enabled": True}},
                },
            },
        )

        env = ExperimentRunner(config=cfg)._create_environment()

        assert env.reward_fn.pipeline_shaping_enabled is True
        assert env.reward_fn.discount_factor == pytest.approx(0.91)


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

    def test_live_rllib_starts_ray_before_environment_and_shuts_owned_runtime(
        self, monkeypatch
    ) -> None:
        events = []
        ray_started = False

        def is_initialized() -> bool:
            return ray_started

        def init(**kwargs) -> None:
            nonlocal ray_started
            assert kwargs == {
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "log_to_driver": False,
            }
            ray_started = True
            events.append("ray_init")

        def shutdown() -> None:
            nonlocal ray_started
            ray_started = False
            events.append("ray_shutdown")

        monkeypatch.setitem(
            sys.modules,
            "ray",
            SimpleNamespace(
                is_initialized=is_initialized,
                init=init,
                shutdown=shutdown,
            ),
        )
        cfg = ExperimentConfig(
            max_steps=1,
            operations_paradigm="autonomous_onboard",
            representation="rl",
            representation_config={"checkpoint_path": "unused-in-unit-test"},
            environment={"scenario": "eventsat", "max_steps": 1},
        )
        runner = ExperimentRunner(config=cfg)
        monkeypatch.setattr(
            runner,
            "_create_environment",
            lambda: events.append("environment") or None,
        )
        monkeypatch.setattr(runner, "_create_memory", lambda: None)
        monkeypatch.setattr(runner, "_create_organization", lambda: None)
        monkeypatch.setattr(runner, "_bind_communication_topology", lambda: None)
        monkeypatch.setattr(runner, "_create_decision_loops", lambda: {})
        monkeypatch.setattr(runner, "_create_operations_paradigm", lambda: None)
        monkeypatch.setattr(runner, "_create_metrics_collector", lambda: None)

        runner._initialize_components()

        assert events[:2] == ["ray_init", "environment"]
        assert runner._owns_ray_runtime is True

        runner._teardown_components()

        assert events[-1] == "ray_shutdown"
        assert runner._owns_ray_runtime is False

    def test_runner_does_not_shut_down_external_ray_runtime(
        self, monkeypatch
    ) -> None:
        calls = []
        monkeypatch.setitem(
            sys.modules,
            "ray",
            SimpleNamespace(
                is_initialized=lambda: True,
                init=lambda **kwargs: calls.append("init"),
                shutdown=lambda: calls.append("shutdown"),
            ),
        )
        cfg = ExperimentConfig(
            operations_paradigm="autonomous_onboard",
            representation="rl",
            representation_config={"checkpoint_path": "unused-in-unit-test"},
            environment={"scenario": "eventsat"},
        )
        runner = ExperimentRunner(config=cfg)

        runner._initialize_runtime_prerequisites()
        runner._teardown_components()

        assert calls == []
        assert runner._owns_ray_runtime is False

    def test_mock_rl_does_not_start_ray(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setitem(
            sys.modules,
            "ray",
            SimpleNamespace(
                is_initialized=lambda: calls.append("is_initialized") or False,
                init=lambda **kwargs: calls.append("init"),
                shutdown=lambda: calls.append("shutdown"),
            ),
        )
        cfg = ExperimentConfig(
            operations_paradigm="autonomous_onboard",
            representation="rl",
            representation_config={"rl_mock": True},
            environment={"scenario": "eventsat"},
        )
        runner = ExperimentRunner(config=cfg)

        runner._initialize_runtime_prerequisites()
        runner._teardown_components()

        assert calls == []

    @pytest.mark.parametrize("constellation_size", [1, 3])
    def test_multieventsat_eventsat_symbolic_core_fails_fast(
        self, constellation_size: int
    ) -> None:
        cfg = ExperimentConfig(
            max_steps=1,
            agent_organization="independent_mas",
            operations_paradigm="autonomous_onboard",
            representation="symbolic",
            environment={
                "scenario": "multieventsat",
                "constellation_size": constellation_size,
                "max_steps": 1,
            },
        )

        with pytest.raises(ValueError, match="rule_based_eventsat.*incompatible"):
            ExperimentRunner(config=cfg)._initialize_components()

    def test_ssa_explicit_eventsat_core_fails_fast(self) -> None:
        cfg = ExperimentConfig(
            max_steps=1,
            operations_paradigm="autonomous_onboard",
            representation="symbolic",
            representation_config={"type": "rule_based_eventsat"},
            environment={"scenario": "ssa", "constellation_size": 1, "max_steps": 1},
        )

        with pytest.raises(ValueError, match="rule_based_eventsat.*incompatible"):
            ExperimentRunner(config=cfg)._initialize_components()

    @pytest.mark.parametrize(
        ("representation", "representation_config", "expected_type"),
        [
            ("symbolic", {}, "rule_based_ssa"),
            ("rl", {"rl_mock": True}, "subsymbolic_ssa"),
        ],
    )
    def test_native_ssa_cores_pass_scenario_contract(
        self,
        representation: str,
        representation_config: dict,
        expected_type: str,
    ) -> None:
        cfg = ExperimentConfig(
            max_steps=1,
            operations_paradigm="autonomous_onboard",
            representation=representation,
            representation_config=representation_config,
            environment={"scenario": "ssa", "constellation_size": 1, "max_steps": 1},
        )
        runner = ExperimentRunner(config=cfg)

        runner._initialize_components()

        assert cfg.resolved_onboard_type == expected_type

    def test_native_step_must_cover_at_least_one_environment_satellite(self) -> None:
        cfg = ExperimentConfig(
            operations_paradigm="autonomous_onboard",
            environment={"scenario": "ssa", "constellation_size": 2},
        )
        runner = ExperimentRunner(config=cfg)
        runner._environment = SimpleNamespace(_sat_ids=["sat_0", "sat_1"])

        with pytest.raises(ValueError, match="covers none"):
            runner._validate_native_action_coverage({"eventsat_0": {"mode": "charging"}})
        runner._validate_native_action_coverage({"sat_1": {"mode": "charging"}})

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
            environment={
                "scenario": "eventsat",
                "constellation_size": 1,
                "max_steps": 2,
                "scenario_config": {"scenario_file": "configs/scenarios/eventsat.yaml"},
            },
        )
        runner = ExperimentRunner(config=cfg)
        results = runner.run()

        assert results["experiment_id"] == "smoke_test"
        assert results["num_episodes"] == 1
        assert (tmp_path / "results" / "results.json").exists()
        assert (tmp_path / "results" / "config.json").exists()


    def test_run_step_accumulates_decision_latency_after_loop_metrics(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        class FakeLoop:
            def __init__(self, reported_latency: float) -> None:
                self.reported_latency = reported_latency

            def process(self, obs, memory):
                return {"eventsat_0": {"mode": "charging"}}, memory

            def get_metrics(self):
                return {"decision_latency_s": self.reported_latency}

        class CaptureCollector:
            def __init__(self) -> None:
                self.decision_metrics = None

            def record_step(self, **kwargs) -> None:
                self.decision_metrics = dict(kwargs["decision_metrics"])

        ticks = iter([0.0, 1.0, 1.3, 2.0, 2.7, 3.0])
        monkeypatch.setattr(
            "src.core.experiment_runner.time.perf_counter", lambda: next(ticks)
        )

        runner = ExperimentRunner(
            config=ExperimentConfig(experiment_id="latency", output_dir=str(tmp_path))
        )
        runner._operations_paradigm = None
        runner._organization = None
        runner._decision_loops = {"a": FakeLoop(99.0), "b": FakeLoop(123.0)}
        runner._memory = None
        runner._ground_planner_loops = {}
        runner._environment = None
        runner._rollout_buffer = None
        runner._representation = None
        runner._metrics_collector = CaptureCollector()
        runner._world_model_trace = None
        runner._decisions_file = None

        runner._run_step(0, None)

        assert runner._metrics_collector.decision_metrics[
            "decision_latency_s"
        ] == pytest.approx(1.0)


    def test_run_step_uses_physical_contact_for_operations(
        self, tmp_path: Path
    ) -> None:
        from src.core.satellite_env import (
            ConstellationState,
            EnvironmentObservation,
            SatelliteState,
        )

        class FakeOps:
            def __init__(self) -> None:
                self.seen = []

            def filter_observation(self, observation, step):
                return observation

            def should_allow_inference(self, step, ground_pass_active):
                self.seen.append(("allow", ground_pass_active))
                return False

            def process_action(self, actions, step, ground_pass_active):
                self.seen.append(("process", ground_pass_active))
                return actions

        sat = SatelliteState(
            satellite_id="eventsat_0",
            position=[0.0, 0.0, 500.0],
            velocity=[0.0, 0.0, 0.0],
            resources={},
            status="charging",
            metadata={
                "ground_pass_active": False,
                "contact_window_active": False,
                "physical_ground_pass_active": True,
            },
        )
        obs = EnvironmentObservation(
            constellation_state=ConstellationState(
                timestep=0,
                epoch_seconds=0.0,
                satellites={"eventsat_0": sat},
                global_info={},
            ),
            tasks=[],
            events=[],
        )

        ops = FakeOps()
        runner = ExperimentRunner(
            config=ExperimentConfig(experiment_id="contact", output_dir=str(tmp_path))
        )
        runner._operations_paradigm = ops
        runner._organization = None
        runner._decision_loops = {}
        runner._memory = None
        runner._ground_planner_loops = {}
        runner._environment = None
        runner._rollout_buffer = None
        runner._representation = None
        runner._metrics_collector = None
        runner._world_model_trace = None
        runner._decisions_file = None

        runner._run_step(0, obs)

        assert ops.seen == [("allow", True), ("process", True)]




    @pytest.mark.parametrize(
        "ops",
        ["autonomous_ground", "conventional_ground", "autonomous_hybrid"],
    )
    @pytest.mark.parametrize("scenario", ["multieventsat", "ssa"])
    @pytest.mark.parametrize("constellation_size", [1, 2])
    def test_native_ground_hybrid_action_schema_fails_fast(
        self, ops: str, scenario: str, constellation_size: int, tmp_path: Path
    ) -> None:
        kwargs = {
            "experiment_id": f"native_{scenario}_{constellation_size}_{ops}",
            "num_episodes": 1,
            "max_steps": 1,
            "output_dir": str(tmp_path),
            "operations_paradigm": ops,
            "environment": {
                "scenario": scenario,
                "constellation_size": constellation_size,
                "timestep_seconds": 60,
                "max_steps": 1,
                "scenario_config": {},
            },
        }
        if scenario == "ssa":
            with pytest.raises(ValueError, match="SSA is AO-only"):
                ExperimentConfig(**kwargs)
            return

        cfg = ExperimentConfig(**kwargs)
        runner = ExperimentRunner(config=cfg)

        with pytest.raises(NotImplementedError, match="native-action SSA or MultiEventSat"):
            runner._initialize_components()


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


class TestEpisodeLifecycleIsolation:
    def test_real_stateful_components_clear_episode_counters(self) -> None:
        from src.core.decision_procedure.sda_loop import SDALoop
        from src.eventsat.agentic import AgenticEventSat
        from src.eventsat.agentic_scheduler import AgenticSchedulerEventSat
        from src.eventsat.llm import LLMEventSat
        from src.eventsat.llm_scheduler import LLMSchedulerEventSat
        from src.eventsat.schedule_symbolic import ScheduleBasedEventSat
        from src.ssa.rl import SubsymbolicSSA

        llm = LLMEventSat({"llm_mock": True})
        llm._grounding_overrides = 4
        llm._last_parse_retries = 2
        llm.reset()
        assert llm.get_metrics()["llm_grounding_overrides"] == 0.0
        assert llm.get_metrics()["llm_last_parse_retries"] == 0.0

        agentic = AgenticEventSat({"llm_mock": True})
        agentic._total_tool_calls = 5
        agentic._total_decisions = 3
        agentic._tool_call_histogram = {"predict": 5}
        agentic.reset()
        assert agentic.get_metrics()["agentic_total_tool_calls"] == 0.0
        assert agentic.get_metrics()["agentic_total_decisions"] == 0.0
        assert "agentic_tool_predict" not in agentic.get_metrics()

        scheduler = LLMSchedulerEventSat({"llm_mock": True})
        scheduler._schedule_entries = 9
        scheduler.reset()
        assert scheduler.get_metrics()["llm_schedule_entries"] == 0.0

        agentic_scheduler = AgenticSchedulerEventSat({"llm_mock": True})
        agentic_scheduler._total_tool_calls = 7
        agentic_scheduler._total_decisions = 2
        agentic_scheduler.reset()
        assert agentic_scheduler.get_metrics()["agentic_total_tool_calls"] == 0.0
        assert agentic_scheduler.get_metrics()["agentic_total_decisions"] == 0.0

        symbolic_scheduler = ScheduleBasedEventSat({})
        symbolic_scheduler._schedule_generated_this_pass = True
        symbolic_scheduler._last_pass_active = True
        symbolic_scheduler.reset()
        assert symbolic_scheduler._schedule_generated_this_pass is False
        assert symbolic_scheduler._last_pass_active is False

        ssa_rl = SubsymbolicSSA({"rl_mock": True})
        ssa_rl._grounding_overrides = 3
        ssa_rl._total_steps = 8
        ssa_rl.reset()
        assert ssa_rl.get_metrics()["rl_grounding_overrides"] == 0.0
        assert ssa_rl.get_metrics()["rl_total_steps"] == 0.0

        loop = SDALoop(config={}, representation=llm)
        loop._total_steps = 11
        loop._last_latency = 2.0
        loop._last_has_rationale = True
        loop.reset()
        assert loop.get_metrics()["total_decisions"] == 0.0
        assert loop.get_metrics()["decision_latency_s"] == 0.0
        assert loop.get_metrics()["has_rationale"] == 0.0

    def test_runner_resets_organization_both_loops_and_unique_cores(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            experiment_id="episode_lifecycle",
            seed=42,
            num_episodes=1,
            max_steps=1,
            environment={"max_steps": 1},
            output_dir=str(tmp_path),
            log_level="WARNING",
        )
        runner = ExperimentRunner(config=cfg)
        events = []

        class Environment:
            def reset(self, seed):
                events.append(("environment_seed", seed))
                return None

            def is_done(self):
                return True

        class Organization:
            directive = "dirty"

            def initialize(self, constellation_size):
                self.directive = None
                events.append(("organization_reset", constellation_size))

        class Client:
            def reset_metrics(self):
                events.append("client_metrics_reset")

        class Core:
            def __init__(self, name):
                self.name = name
                self._client = Client()

            def reset(self):
                events.append((self.name, "reset"))

            def seed(self, seed):
                events.append((self.name, "seed", seed))

        class Loop:
            def __init__(self, representation, name):
                self.representation = representation
                self.name = name

            def reset(self):
                events.append((self.name, "loop_reset"))

        class Operations:
            def reset(self):
                events.append("operations_reset")

            def get_metrics(self):
                return {}

        onboard = Core("onboard")
        ground = Core("ground")
        organization = Organization()
        runner._environment = Environment()
        runner._organization = organization
        runner._representation = onboard
        runner._representations = {"a": onboard, "duplicate": onboard}
        runner._decision_loops = {"a": Loop(onboard, "onboard")}
        runner._ground_planner_loops = {"a": Loop(ground, "ground")}
        runner._memory = None
        runner._metrics_collector = None
        runner._operations_paradigm = Operations()
        runner._run_step = lambda step, observation: {"observation": None}

        runner._run_episode(episode_id=2)

        assert ("environment_seed", 44) in events
        assert ("organization_reset", cfg.environment.constellation_size) in events
        assert organization.directive is None
        assert events.count(("onboard", "loop_reset")) == 1
        assert events.count(("ground", "loop_reset")) == 1
        assert events.count(("onboard", "reset")) == 1
        assert events.count(("ground", "reset")) == 1
        onboard_seed = next(
            event[2]
            for event in events
            if isinstance(event, tuple) and event[:2] == ("onboard", "seed")
        )
        ground_seed = next(
            event[2]
            for event in events
            if isinstance(event, tuple) and event[:2] == ("ground", "seed")
        )
        assert onboard_seed != ground_seed
        assert onboard_seed == runner._derive_component_seed(
            44, runner._core_seed_keys()[id(onboard)]
        )
        assert ground_seed == runner._derive_component_seed(
            44, runner._core_seed_keys()[id(ground)]
        )

    def test_component_seeds_are_reproducible_and_satellite_specific(self) -> None:
        sat_0 = ExperimentRunner._derive_component_seed(42, "onboard|satellite=sat_0")
        sat_1 = ExperimentRunner._derive_component_seed(42, "onboard|satellite=sat_1")

        assert sat_0 == ExperimentRunner._derive_component_seed(
            42, "onboard|satellite=sat_0"
        )
        assert sat_0 != sat_1

    def test_results_placeholder_flag_covers_ground_core(self) -> None:
        cfg = ExperimentConfig(
            operations_paradigm="autonomous_hybrid",
            onboard={"representation": "symb"},
            ground={"representation": "hrl"},
        )
        runner = ExperimentRunner(config=cfg)
        onboard = SimpleNamespace(is_placeholder=False)
        ground = SimpleNamespace(is_placeholder=True)
        runner._representation = onboard
        runner._representations = {"central_agent": onboard}
        runner._ground_planner_loops = {
            "central_agent": SimpleNamespace(representation=ground)
        }

        class Collector:
            def finalise_experiment(self, experiment_id):
                return SimpleNamespace(metadata=None)

        runner._metrics_collector = Collector()
        results = runner._compile_results([])
        metadata = results["experiment_statistics"].metadata

        assert metadata["representation_is_placeholder"] is True
        assert metadata["representation_core_placeholders"] == {
            "onboard": False,
            "ground": True,
        }
        assert metadata["representation_core_types"] == {
            "onboard": "rule_based_eventsat",
            "ground": "hrl_scheduler_eventsat",
        }


class TestResultProvenance:
    def test_real_llm_mock_run_serializes_runtime_and_source_truth(
        self, tmp_path: Path
    ) -> None:
        cfg = ExperimentConfig(
            experiment_id="llm_mock_provenance",
            seed=42,
            num_episodes=1,
            max_steps=1,
            output_dir=str(tmp_path),
            operations_paradigm="autonomous_ground",
            representation="llm-s",
            representation_config={"llm_mock": True},
            environment={"scenario": "eventsat", "max_steps": 1},
            log_level="WARNING",
        )

        results = ExperimentRunner(config=cfg).run()
        saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))

        for artifact in (results, saved):
            assert artifact["representation_is_placeholder"] is False
            assert artifact["representation_is_mock"] is True
            assert artifact["llm_mock"] is True
            assert artifact["llm_fallback_used"] is False
            assert len(artifact["llm_provenance"]) == 1
            assert artifact["llm_provenance"][0]["llm_mock"] is True

        metadata = saved["experiment_statistics"]["metadata"]
        assert metadata["representation_core_mocks"] == {
            "onboard": False,
            "ground": True,
        }
        assert len(metadata["representation_core_runtime"]["ground"]) == 1
        runtime_core = metadata["representation_core_runtime"]["ground"][0]
        assert runtime_core["is_mock"] is True
        assert runtime_core["llm_mock"] is True

        source = saved["source_provenance"]
        assert set(source) == {
            "git_revision",
            "git_dirty",
            "git_diff_sha256",
            "git_untracked_file_count",
            "uv_lock_sha256",
        }
        assert len(source["git_revision"]) == 40
        assert isinstance(source["git_dirty"], bool)
        assert isinstance(source["git_untracked_file_count"], int)
        if source["git_dirty"]:
            assert len(source["git_diff_sha256"]) == 64
        else:
            assert source["git_diff_sha256"] is None
        expected_lock_hash = hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "uv.lock").read_bytes()
        ).hexdigest()
        assert source["uv_lock_sha256"] == expected_lock_hash
        assert metadata["source_provenance"] == source

    def test_multi_core_llm_metrics_are_deduplicated_and_weighted(self) -> None:
        from src.eventsat.llm import LLMEventSat

        first = LLMEventSat({"llm_mock": True, "agent_id": "agent_a"})
        second = LLMEventSat({"llm_mock": True, "agent_id": "agent_b"})
        for _ in range(2):
            first._client.generate("system", "first")
        for _ in range(3):
            second._client.generate("system", "second")

        first._client._cache_hits = 1
        first._client._total_latency_s = 3.0
        first._client._last_latency_s = 0.4
        first._client._total_prompt_tokens = 10
        first._client._total_completion_tokens = 20
        second._client._total_latency_s = 8.0
        second._client._last_latency_s = 0.9
        second._client._total_prompt_tokens = 40
        second._client._total_completion_tokens = 50

        runner = ExperimentRunner(config=ExperimentConfig())
        runner._representation = first
        # The first core appears through three runner references but counts once.
        runner._representations = {
            "agent_a": first,
            "duplicate_reference": first,
            "agent_b": second,
        }

        metrics = runner._collect_core_llm_metrics()
        assert metrics["llm_api_calls"] == 5.0
        assert metrics["llm_cache_hits"] == 1.0
        assert metrics["llm_cache_hit_rate"] == pytest.approx(1.0 / 5.0)
        assert metrics["llm_total_latency_s"] == 11.0
        assert metrics["llm_mean_call_latency_s"] == pytest.approx(11.0 / 4.0)
        assert metrics["llm_last_latency_s"] == 0.9
        assert metrics["llm_tokens_prompt"] == 50.0
        assert metrics["llm_tokens_completion"] == 70.0

        provenance = runner._collect_core_llm_provenance()
        assert provenance["llm_mock"] is True
        assert provenance["llm_fallback_used"] is False
        assert [entry["agent_id"] for entry in provenance["llm_provenance"]] == [
            "agent_a",
            "agent_b",
        ]

    @pytest.mark.parametrize(
        ("current_invocation", "origin_invocation", "expected_fallback"),
        [
            ("fallback", "direct", True),
            ("direct", "fallback", False),
        ],
    )
    def test_fallback_aggregation_uses_current_invocation_only(
        self,
        current_invocation: str,
        origin_invocation: str,
        expected_fallback: bool,
    ) -> None:
        class Client:
            def get_provenance(self):
                return {
                    "llm_mock": False,
                    "llm_call_provenance": [
                        {
                            "provider": "openai",
                            "invocation": current_invocation,
                            "cache_origin_invocation": origin_invocation,
                            "cache_hit": True,
                        }
                    ],
                }

        runner = ExperimentRunner(config=ExperimentConfig())
        runner._representation = SimpleNamespace(_client=Client(), config={})

        provenance = runner._collect_core_llm_provenance()
        assert provenance["llm_fallback_used"] is expected_fallback
        record = provenance["llm_provenance"][0]["llm_call_provenance"][0]
        assert record["invocation"] == current_invocation
        assert record["cache_origin_invocation"] == origin_invocation


class TestPerAgentRepresentationIsolation:
    @staticmethod
    def _ssa_rl_config(org: str) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id=f"{org}_core_isolation",
            seed=73,
            num_episodes=1,
            max_steps=1,
            operations_paradigm="autonomous_onboard",
            agent_organization=org,
            representation="rl",
            representation_config={
                "rl_mock": True,
                "deterministic": False,
                "mock_uses_heuristic": False,
            },
            environment={
                "scenario": "ssa",
                "constellation_size": 3,
                "max_steps": 1,
            },
            log_level="WARNING",
        )

    @pytest.mark.parametrize(
        ("org", "expected_agents"),
        [
            ("centralized_mas", 4),
            ("decentralized_mas", 3),
            ("hybrid_mas", 2),
        ],
    )
    def test_each_logical_agent_owns_representation_and_seed(
        self, org: str, expected_agents: int
    ) -> None:
        runner = ExperimentRunner(config=self._ssa_rl_config(org))
        runner._initialize_components()
        agents = runner._organization.get_agents()
        reps = {agent: runner._representations[agent] for agent in agents}
        assert len(reps) == expected_agents
        assert len({id(rep) for rep in reps.values()}) == expected_agents
        assert all(
            rep.config.get("agent_id") == agent_id
            for agent_id, rep in reps.items()
        )
        keys = runner._core_seed_keys()
        agent_keys = {agent: keys[id(rep)] for agent, rep in reps.items()}
        assert all(
            f"agent={agent}" in key for agent, key in agent_keys.items()
        )
        assert all(
            rep.config["seed"]
            == runner._derive_component_seed(73, agent_keys[agent])
            for agent, rep in reps.items()
        )
        seeds = {
            runner._derive_component_seed(73, key)
            for key in agent_keys.values()
        }
        assert len(seeds) == expected_agents

    def test_sas_retains_single_core_path(self) -> None:
        runner = ExperimentRunner(config=self._ssa_rl_config("sas"))
        runner._initialize_components()
        rep = runner._representations["central_agent"]
        assert rep is runner._representation
        assert len(runner._core_representations()) == 1
        assert "agent_id" not in rep.config

    def test_dmas_binds_all_to_all_physical_links_and_refreshes_local_views(
        self,
    ) -> None:
        runner = ExperimentRunner(
            config=self._ssa_rl_config("decentralized_mas")
        )
        runner._initialize_components()
        expected = {
            (src, dst)
            for src in ("sat_0", "sat_1", "sat_2")
            for dst in ("sat_0", "sat_1", "sat_2")
            if src != dst
        }

        assert runner._environment._authorized_communication_links == expected
        observation = runner._environment.reset(seed=73)
        runner._organization.initialize(constellation_size=3)
        runner._bind_communication_topology()
        observation = runner._environment.get_observation()
        views = runner._organization.distribute_observation(observation)

        assert runner._environment._authorized_communication_links == expected
        assert all(
            list(view.local_state["full_observation"].constellation_state.satellites)
            == [f"sat_{idx}"]
            for idx, view in enumerate(views.values())
        )
        assert all(
            next(
                iter(
                    view.local_state["full_observation"]
                    .constellation_state.satellites.values()
                )
            ).metadata["has_isl_peer"]
            is True
            for view in views.values()
        )

    def test_reversed_construction_and_evaluation_preserve_agent_trajectories(
        self,
    ) -> None:
        def build(reverse_agents: bool):
            runner = ExperimentRunner(
                config=self._ssa_rl_config("centralized_mas")
            )
            runner._environment = runner._create_environment()
            runner._memory = runner._create_memory()
            runner._organization = runner._create_organization()
            natural_agents = runner._organization.get_agents()
            if reverse_agents:
                runner._organization.get_agents = lambda: list(
                    reversed(natural_agents)
                )
            runner._decision_loops = runner._create_decision_loops()
            observation = runner._environment.reset(seed=73)
            runner._organization.initialize(constellation_size=3)
            return (
                runner,
                runner._organization.distribute_observation(observation),
                natural_agents,
            )

        def trajectories(runner, observations, order):
            keys = runner._core_seed_keys()
            for loop in runner._decision_loops.values():
                loop.reset()
            for rep in runner._core_representations():
                rep.reset()
                rep.seed(runner._derive_component_seed(73, keys[id(rep)]))
            memory = FixedMemory()
            result = {agent: [] for agent in order}
            for _ in range(8):
                for agent_id in order:
                    action, memory = runner._decision_loops[agent_id].process(
                        observations[agent_id], memory
                    )
                    result[agent_id].append(action)
            return result

        normal, normal_obs, agents = build(False)
        reversed_runner, reversed_obs, _ = build(True)
        normal_trajectory = trajectories(normal, normal_obs, agents)
        reversed_trajectory = trajectories(
            reversed_runner, reversed_obs, list(reversed(agents))
        )
        assert normal_trajectory == reversed_trajectory


class TestOrganizationInstantiation:
    """All five Kim et al. organisations instantiate and are runnable."""

    @pytest.mark.parametrize(
        "org,expected_agents",
        [
            ("sas", 1),
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
            operations_paradigm="autonomous_onboard",
            environment={"scenario": "ssa", "constellation_size": 3},
        )
        runner = ExperimentRunner(config=cfg)
        organization = runner._create_organization()
        assert len(organization.get_agents()) == expected_agents

    def test_cmas_multisat_fails_fast_when_run(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            experiment_id="cmas_not_runnable",
            num_episodes=1,
            max_steps=2,
            output_dir=str(tmp_path),
            agent_organization="centralized_mas",
            operations_paradigm="autonomous_onboard",
            environment={"scenario": "ssa", "constellation_size": 3, "max_steps": 2},
        )
        runner = ExperimentRunner(config=cfg)

        with pytest.raises(ValueError, match="centralized_mas.*not implemented"):
            runner.run()

    def test_placeholder_representation_fails_fast_when_run(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            experiment_id="placeholder_not_runnable",
            num_episodes=1,
            max_steps=2,
            output_dir=str(tmp_path),
            representation="hrl",
            operations_paradigm="autonomous_ground",
            environment={
                "scenario": "eventsat",
                "constellation_size": 1,
                "max_steps": 2,
                "scenario_config": {"scenario_file": "configs/scenarios/eventsat.yaml"},
            },
        )
        runner = ExperimentRunner(config=cfg)

        with pytest.raises(ValueError, match="placeholder representation"):
            runner.run()


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

    def test_nested_explicit_ground_type_controls_runtime_resolution(self) -> None:
        cfg = self._cfg(
            onboard={"representation": "symb"},
            ground={
                "representation": "symb",
                "representation_config": {"type": "hrl_scheduler_eventsat"},
            },
        )
        assert cfg.resolved_ground_planner_type == "hrl_scheduler_eventsat"

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
        # A submitted plan becomes executable only after a resolved
        # communication step proves that the uplink reached the spacecraft.
        sat = SimpleNamespace(resources={}, metadata={}, status="communication")
        obs = SimpleNamespace(
            constellation_state=SimpleNamespace(satellites={"eventsat_0": sat})
        )
        ah.update_ground_knowledge(obs, step=0)
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
            {"eventsat_0": {"mode": "payload_detect"}}, step=1, ground_pass_active=True
        )
        assert out["eventsat_0"]["mode"] == "payload_detect"  # real-time during contact

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

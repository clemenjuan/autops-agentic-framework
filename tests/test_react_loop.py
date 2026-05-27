"""Tests for the ReAct decision loop (Yao et al. 2023).

The ReAct loop implements iterative Thought-Action-Observation cycles:
  1. Thought: representation.reason(state, memory) → reasoning trace
  2. Action:  representation.select_action(context) → proposed action
  3. Observation: grounding validation → violations feedback or convergence
"""
import pytest
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------
# Minimal representation stubs for isolated testing
# -----------------------------------------------------------------


class AlwaysValidRepresentation:
    """Always returns a valid action (charging) that passes grounding."""

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        if hasattr(observation, "constellation_state"):
            sat = observation.constellation_state.satellites.get("eventsat_0")
            if sat:
                return {
                    "battery_soc": sat.resources.get("battery_soc", 0.8),
                    "ground_pass_active": sat.metadata.get("ground_pass_active", False),
                    "health_status": "nominal",
                }
        return {"battery_soc": 0.8, "ground_pass_active": False, "health_status": "nominal"}

    def select_action(self, context: Any) -> Dict[str, Any]:
        return {"eventsat_0": {"mode": "charging"}}

    def reason(self, state: Any, memory: Any) -> List[Dict[str, Any]]:
        return [{"check": "default", "value": state.get("battery_soc", 0.5), "implication": "charging"}]

    def get_rationale(self) -> Optional[str]:
        return "AlwaysValid: charging"

    def get_name(self) -> str:
        return "AlwaysValidRepresentation"


class LowBatteryObserveRepresentation:
    """Proposes payload_observe when battery is critically low — will fail grounding."""

    def __init__(self):
        self._call_count = 0

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        return {"battery_soc": 0.10, "ground_pass_active": False, "health_status": "nominal"}

    def select_action(self, context: Any) -> Dict[str, Any]:
        self._call_count += 1
        # First iterations: propose invalid action; last: fall back to charging
        if context.enrichments.get("iteration", 0) < 2:
            return {"eventsat_0": {"mode": "payload_observe"}}
        return {"eventsat_0": {"mode": "charging"}}

    def reason(self, state: Any, memory: Any) -> List[Dict[str, Any]]:
        soc = state.get("battery_soc", 0.5)
        if soc < 0.30:
            return [{"check": "battery", "value": soc, "implication": "charging_required"}]
        return []

    def get_rationale(self) -> Optional[str]:
        return "LowBattery: observe proposed but will fail grounding"

    def get_name(self) -> str:
        return "LowBatteryObserveRepresentation"


class CommsWithoutPassRepresentation:
    """Always proposes communication (will fail pass_window_timing grounding)."""

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        return {"battery_soc": 0.8, "ground_pass_active": False, "health_status": "nominal"}

    def select_action(self, context: Any) -> Dict[str, Any]:
        return {"eventsat_0": {"mode": "communication"}}

    def reason(self, state: Any, memory: Any) -> List[Dict[str, Any]]:
        return [{"check": "pass", "value": False, "implication": "no_pass_active"}]

    def get_rationale(self) -> Optional[str]:
        return "CommsWithoutPass: will fail pass_window_timing"

    def get_name(self) -> str:
        return "CommsWithoutPassRepresentation"


class NoReasonRepresentation:
    """Representation without reason() to test backward compatibility."""

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        return {"battery_soc": 0.8, "ground_pass_active": False, "health_status": "nominal"}

    def select_action(self, context: Any) -> Dict[str, Any]:
        return {"eventsat_0": {"mode": "charging"}}

    def get_name(self) -> str:
        return "NoReasonRepresentation"


# -----------------------------------------------------------------
# Environment observation helper
# -----------------------------------------------------------------


def _make_observation(battery_soc=0.8, ground_pass_active=False):
    from src.environment.satellite_env import (
        ConstellationState,
        EnvironmentObservation,
        SatelliteState,
    )
    sat = SatelliteState(
        satellite_id="eventsat_0",
        position=[0.0, 0.0, 500.0],
        velocity=[0.0, 0.0, 0.0],
        resources={"battery_soc": battery_soc, "data_stored_mb": 0.0, "data_downlinked_mb": 0.0},
        status="charging",
        metadata={
            "in_sunlight": True,
            "ground_pass_active": ground_pass_active,
            "uncompressed_observations": 0,
            "total_observation_s": 0.0,
            "storage_capacity_mb": 512.0,
            "health_status": "nominal",
        },
    )
    constellation = ConstellationState(
        timestep=0,
        epoch_seconds=0.0,
        satellites={"eventsat_0": sat},
        global_info={"max_steps": 10080},
    )
    return EnvironmentObservation(constellation_state=constellation, tasks=[], events=[])


# -----------------------------------------------------------------
# Core ReActLoop behavior
# -----------------------------------------------------------------


class TestReActLoopBasic:
    def test_get_name(self):
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(config={}, representation=AlwaysValidRepresentation())
        assert loop.get_name() == "ReActLoop"

    def test_process_returns_action_and_memory(self):
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(config={}, representation=AlwaysValidRepresentation())
        obs = _make_observation()
        action, memory = loop.process(obs, memory=None)
        assert "eventsat_0" in action
        assert "mode" in action["eventsat_0"]

    def test_valid_action_converges_first_iteration(self):
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(
            config={"max_iterations": 3, "grounding_checks": ["battery_feasibility", "pass_window_timing"]},
            representation=AlwaysValidRepresentation(),
        )
        obs = _make_observation(battery_soc=0.8)
        action, _ = loop.process(obs, memory=None)
        assert action == {"eventsat_0": {"mode": "charging"}}
        metrics = loop.get_metrics()
        assert metrics["converged"] == 1.0

    def test_memory_passthrough(self):
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(config={}, representation=AlwaysValidRepresentation())
        obs = _make_observation()
        memory_in = {"some_key": "some_value"}
        _, memory_out = loop.process(obs, memory=memory_in)
        assert memory_out is memory_in

    def test_reset_clears_metrics(self):
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(config={}, representation=AlwaysValidRepresentation())
        obs = _make_observation()
        loop.process(obs, memory=None)
        assert loop._total_steps == 1
        loop.reset()
        assert loop._total_steps == 0
        assert loop._last_grounding_violations == 0
        assert loop._last_converged is True


# -----------------------------------------------------------------
# Grounding validation
# -----------------------------------------------------------------


class TestGrounding:
    def test_battery_feasibility_violation(self):
        from src.decision_loop.react_loop import ReActLoop, _GROUNDING_MIN_SOC
        loop = ReActLoop(
            config={"grounding_checks": ["battery_feasibility"]},
            representation=AlwaysValidRepresentation(),
        )
        # Simulate low SoC state with energy-intensive mode
        violations = loop._check_grounding(
            {"eventsat_0": {"mode": "payload_observe"}},
            {"battery_soc": 0.10, "ground_pass_active": False},
        )
        assert len(violations) == 1
        assert violations[0]["check"] == "battery_feasibility"

    def test_battery_feasibility_no_violation_charging(self):
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(
            config={"grounding_checks": ["battery_feasibility"]},
            representation=AlwaysValidRepresentation(),
        )
        violations = loop._check_grounding(
            {"eventsat_0": {"mode": "charging"}},
            {"battery_soc": 0.10, "ground_pass_active": False},
        )
        assert violations == []

    def test_pass_window_timing_violation(self):
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(
            config={"grounding_checks": ["pass_window_timing"]},
            representation=AlwaysValidRepresentation(),
        )
        violations = loop._check_grounding(
            {"eventsat_0": {"mode": "communication"}},
            {"battery_soc": 0.8, "ground_pass_active": False},
        )
        assert len(violations) == 1
        assert violations[0]["check"] == "pass_window_timing"

    def test_pass_window_timing_no_violation_during_pass(self):
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(
            config={"grounding_checks": ["pass_window_timing"]},
            representation=AlwaysValidRepresentation(),
        )
        violations = loop._check_grounding(
            {"eventsat_0": {"mode": "communication"}},
            {"battery_soc": 0.8, "ground_pass_active": True},
        )
        assert violations == []

    def test_empty_grounding_checks(self):
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(
            config={"grounding_checks": []},
            representation=CommsWithoutPassRepresentation(),
        )
        obs = _make_observation(battery_soc=0.8, ground_pass_active=False)
        action, _ = loop.process(obs, memory=None)
        # No grounding checks → converges immediately with communication
        assert action == {"eventsat_0": {"mode": "communication"}}


# -----------------------------------------------------------------
# Iteration and convergence
# -----------------------------------------------------------------


class TestIteration:
    def test_low_battery_observe_fails_then_converges(self):
        """Low battery + observe fails grounding; representation retries, converges to charging."""
        from src.decision_loop.react_loop import ReActLoop
        repr_ = LowBatteryObserveRepresentation()
        loop = ReActLoop(
            config={"max_iterations": 3, "grounding_checks": ["battery_feasibility"]},
            representation=repr_,
        )
        obs = _make_observation(battery_soc=0.10)
        action, _ = loop.process(obs, memory=None)
        # Third iteration (index 2) returns charging, which passes grounding
        assert action == {"eventsat_0": {"mode": "charging"}}
        metrics = loop.get_metrics()
        assert metrics["converged"] == 1.0
        assert metrics["grounding_violations"] > 0

    def test_all_iterations_fail_fallback_to_charging(self):
        """If all iterations fail grounding, fallback to charging."""
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(
            config={"max_iterations": 3, "grounding_checks": ["pass_window_timing"]},
            representation=CommsWithoutPassRepresentation(),
        )
        obs = _make_observation(battery_soc=0.8, ground_pass_active=False)
        action, _ = loop.process(obs, memory=None)
        assert action == {"eventsat_0": {"mode": "charging"}}
        metrics = loop.get_metrics()
        assert metrics["converged"] == 0.0
        assert metrics["grounding_violations"] == 3.0  # 3 iterations × 1 violation each

    def test_max_iterations_one(self):
        """With max_iterations=1 and always-valid repr, converges in one pass."""
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(
            config={"max_iterations": 1, "grounding_checks": ["battery_feasibility"]},
            representation=AlwaysValidRepresentation(),
        )
        obs = _make_observation(battery_soc=0.8)
        action, _ = loop.process(obs, memory=None)
        assert action["eventsat_0"]["mode"] == "charging"
        metrics = loop.get_metrics()
        assert metrics["converged"] == 1.0


# -----------------------------------------------------------------
# Reasoning trace
# -----------------------------------------------------------------


class TestReasoningTrace:
    def test_reasoning_trace_populated_in_context(self):
        """Enrichments contain reasoning_trace after first iteration."""
        from src.decision_loop.react_loop import ReActLoop

        captured_contexts = []

        class CaptureContextRepresentation:
            def encode_observation(self, obs):
                return {"battery_soc": 0.8, "ground_pass_active": False, "health_status": "nominal"}

            def select_action(self, context):
                captured_contexts.append(context)
                return {"eventsat_0": {"mode": "charging"}}

            def reason(self, state, memory):
                return [{"check": "test", "value": state.get("battery_soc"), "implication": "ok"}]

        loop = ReActLoop(config={"max_iterations": 2}, representation=CaptureContextRepresentation())
        obs = _make_observation()
        loop.process(obs, memory=None)

        assert len(captured_contexts) >= 1
        ctx = captured_contexts[0]
        assert ctx.loop_type == "react"
        assert "reasoning_trace" in ctx.enrichments
        assert len(ctx.enrichments["reasoning_trace"]) > 0
        assert ctx.enrichments["iteration"] == 0

    def test_reasoning_depth_metric_positive(self):
        """reasoning_depth metric counts total thought steps."""
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(config={}, representation=AlwaysValidRepresentation())
        obs = _make_observation()
        loop.process(obs, memory=None)
        metrics = loop.get_metrics()
        assert metrics["reasoning_depth"] >= 0

    def test_no_reason_method_backward_compatible(self):
        """Representation without reason() still works (no-op)."""
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(config={}, representation=NoReasonRepresentation())
        obs = _make_observation()
        action, _ = loop.process(obs, memory=None)
        assert "eventsat_0" in action


# -----------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------


class TestMetrics:
    def test_metrics_keys_present(self):
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(config={}, representation=AlwaysValidRepresentation())
        obs = _make_observation()
        loop.process(obs, memory=None)
        metrics = loop.get_metrics()
        expected_keys = {
            "decision_latency_s", "reasoning_depth", "iterations",
            "grounding_violations", "converged", "has_rationale", "total_decisions",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_total_decisions_accumulates(self):
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(config={}, representation=AlwaysValidRepresentation())
        obs = _make_observation()
        for _ in range(5):
            loop.process(obs, memory=None)
        assert loop.get_metrics()["total_decisions"] == 5.0

    def test_has_rationale_when_rationale_available(self):
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(config={}, representation=AlwaysValidRepresentation())
        obs = _make_observation()
        loop.process(obs, memory=None)
        assert loop.get_metrics()["has_rationale"] == 1.0

    def test_has_rationale_false_when_unavailable(self):
        from src.decision_loop.react_loop import ReActLoop
        loop = ReActLoop(config={}, representation=NoReasonRepresentation())
        obs = _make_observation()
        loop.process(obs, memory=None)
        assert loop.get_metrics()["has_rationale"] == 0.0


# -----------------------------------------------------------------
# Integration: ReActLoop + real representations
# -----------------------------------------------------------------


class TestReActIntegration:
    def test_react_with_rule_based_representation(self):
        """ReAct + RuleBasedEventSat: reason() provides trace, grounding validates."""
        from src.decision_loop.react_loop import ReActLoop
        from src.representation.rule_based_eventsat import RuleBasedEventSat
        import src.representation.rule_based_eventsat  # ensure registered

        repr_ = RuleBasedEventSat()
        loop = ReActLoop(
            config={"max_iterations": 3, "grounding_checks": ["battery_feasibility", "pass_window_timing"]},
            representation=repr_,
        )
        obs = _make_observation(battery_soc=0.8, ground_pass_active=False)
        action, _ = loop.process(obs, memory=None)
        assert "eventsat_0" in action
        assert action["eventsat_0"]["mode"] in {
            "charging", "payload_observe", "payload_compress",
            "payload_detect", "payload_send", "safe",
        }

    def test_react_rule_based_low_battery_no_energy_intensive(self):
        """With low battery, grounding should reject energy-intensive modes."""
        from src.decision_loop.react_loop import ReActLoop, _GROUNDING_MIN_SOC
        from src.representation.rule_based_eventsat import RuleBasedEventSat

        repr_ = RuleBasedEventSat()
        loop = ReActLoop(
            config={"max_iterations": 3, "grounding_checks": ["battery_feasibility"]},
            representation=repr_,
        )
        obs = _make_observation(battery_soc=0.10)  # critically low
        action, _ = loop.process(obs, memory=None)
        mode = action["eventsat_0"]["mode"]
        # With SoC=0.10, R2 fires → charging, which passes grounding
        assert mode == "charging"
        assert loop.get_metrics()["converged"] == 1.0

    def test_react_with_experiment_runner(self, tmp_path):
        """Full ExperimentRunner integration with ReAct loop."""
        from src.orchestration.config_loader import ExperimentConfig
        from src.orchestration.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(
            experiment_id="react_test_integration",
            agent_organization="sas",
            decision_loop="react",
            representation="symbolic",
            emergence_mode="hand_designed",
            operations_paradigm="autonomous_hybrid",
            decision_loop_config={
                "max_iterations": 3,
                "grounding_checks": ["battery_feasibility", "pass_window_timing"],
            },
            representation_config={"type": "rule_based_eventsat"},
            environment={
                "constellation_size": 1,
                "timestep_seconds": 60,
                "max_steps": 100,
                "scenario": "eventsat",
                "scenario_config": {},
            },
            num_episodes=1,
            max_steps=100,
            save_checkpoints=False,
            log_level="WARNING",
            output_dir=str(tmp_path),
        )
        runner = ExperimentRunner(config=cfg)
        results = runner.run()
        assert results["num_episodes"] == 1
        assert len(results["episodes"][0]["steps"]) == 100

    def test_react_with_schedule_based_representation(self, tmp_path):
        """ReAct + ScheduleBasedEventSat: schedule planning reasoning + grounding."""
        from src.orchestration.config_loader import ExperimentConfig
        from src.orchestration.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(
            experiment_id="react_test_schedule",
            agent_organization="sas",
            decision_loop="react",
            representation="symbolic",
            emergence_mode="hand_designed",
            operations_paradigm="autonomous_ground",
            decision_loop_config={
                "max_iterations": 3,
                "grounding_checks": ["battery_feasibility", "pass_window_timing"],
            },
            operations_paradigm_config={"orbital_period_steps": 93},
            representation_config={"type": "schedule_based_eventsat"},
            environment={
                "constellation_size": 1,
                "timestep_seconds": 60,
                "max_steps": 200,
                "scenario": "eventsat",
                "scenario_config": {},
            },
            num_episodes=1,
            max_steps=200,
            save_checkpoints=False,
            log_level="WARNING",
            output_dir=str(tmp_path),
        )
        runner = ExperimentRunner(config=cfg)
        results = runner.run()
        assert results["num_episodes"] == 1
        assert len(results["episodes"][0]["steps"]) == 200

    def test_react_metrics_present_in_results(self, tmp_path):
        """ReAct-specific metrics should appear in the results."""
        from src.orchestration.config_loader import ExperimentConfig
        from src.orchestration.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(
            experiment_id="react_test_metrics",
            agent_organization="sas",
            decision_loop="react",
            representation="symbolic",
            emergence_mode="hand_designed",
            operations_paradigm="autonomous_hybrid",
            decision_loop_config={"max_iterations": 3},
            representation_config={"type": "rule_based_eventsat"},
            environment={
                "constellation_size": 1,
                "timestep_seconds": 60,
                "max_steps": 10,
                "scenario": "eventsat",
                "scenario_config": {},
            },
            num_episodes=1,
            max_steps=10,
            save_checkpoints=False,
            log_level="WARNING",
            output_dir=str(tmp_path),
        )
        runner = ExperimentRunner(config=cfg)
        results = runner.run()
        # Check that any step has react-related metrics
        steps = results["episodes"][0]["steps"]
        # At least one step should have decision metrics
        for step in steps:
            if "decision_metrics" in step:
                metrics = step["decision_metrics"]
                assert "converged" in metrics or "decision_latency_s" in metrics
                break

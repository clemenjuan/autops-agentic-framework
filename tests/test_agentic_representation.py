"""
Tests for agentic hybrid representation (Phase 4c).

All tests use mock mode (no live LLM calls) to ensure CI compatibility.
"""
from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.decision_procedure.context import DecisionContext
from src.representation.agentic_tools import (
    TOOL_REGISTRY,
    VALID_MODES,
    check_battery,
    check_constraints,
    check_data_pipeline,
    check_ground_pass,
    evaluate_plan,
    execute_tool,
    recall_history,
)
from src.representation.agentic_prompts import (
    AGENTIC_SYSTEM_PROMPT,
    format_agentic_reasoning_prompt,
    format_planning_prompt,
    format_tool_result_prompt,
)
from src.representation.agentic_eventsat import AgenticEventSat


# ======================================================================
# Helpers
# ======================================================================

def _make_state(**overrides: Any) -> Dict[str, Any]:
    """Create a minimal satellite state dict with sensible defaults."""
    state = {
        "battery_soc": 0.7,
        "current_mode": "charging",
        "in_sunlight": True,
        "ground_pass_active": False,
        "data_stored_mb": 10.0,
        "obc_data_mb": 0.0,
        "jetson_raw_mb": 0.0,
        "jetson_compressed_mb": 0.0,
        "storage_capacity_mb": 512.0,
        "uncompressed_observations": 0,
        "compression_progress": 0,
        "total_observation_s": 0.0,
        "health_status": "nominal",
        "undetected_observations": 0,
        "daily_downlink_budget_mb": 27.0,
    }
    state.update(overrides)
    return state


def _make_context(state: Dict[str, Any] | None = None, **kwargs: Any) -> DecisionContext:
    """Create a DecisionContext with defaults."""
    return DecisionContext(
        state=state or _make_state(),
        loop_type=kwargs.get("loop_type", "sda"),
        memory=kwargs.get("memory", None),
        enrichments=kwargs.get("enrichments", {}),
        loop_metadata=kwargs.get("loop_metadata", {}),
    )


def _mock_config(**overrides: Any) -> Dict[str, Any]:
    """Create a config dict with mock mode enabled."""
    config = {"llm_mock": True}
    config.update(overrides)
    return config


# ======================================================================
# Tool Tests
# ======================================================================

class TestCheckBattery(unittest.TestCase):
    """Test check_battery tool."""

    def test_nominal_soc(self):
        result = check_battery(state=_make_state(battery_soc=0.72))
        self.assertEqual(result["soc"], 0.72)
        self.assertEqual(result["charging_assessment"], "good")
        self.assertFalse(result["below_preferred"])
        self.assertFalse(result["below_hard_limit"])
        self.assertIn("charging", result["feasible_modes"])

    def test_low_soc(self):
        result = check_battery(state=_make_state(battery_soc=0.25))
        self.assertEqual(result["charging_assessment"], "low")
        self.assertTrue(result["below_preferred"])
        self.assertFalse(result["below_hard_limit"])

    def test_critical_soc(self):
        result = check_battery(state=_make_state(battery_soc=0.15))
        self.assertEqual(result["charging_assessment"], "critical")
        self.assertTrue(result["below_hard_limit"])
        # Only charging and safe should be feasible
        for mode in result["feasible_modes"]:
            self.assertIn(mode, ["charging", "safe"])

    def test_sunlight_affects_charging_rate(self):
        result_sun = check_battery(state=_make_state(in_sunlight=True))
        result_eclipse = check_battery(state=_make_state(in_sunlight=False))
        self.assertEqual(result_sun["charging_rate"], "nominal")
        self.assertEqual(result_eclipse["charging_rate"], "none (eclipse)")

    def test_anomaly_restricts_to_safe(self):
        result = check_battery(state=_make_state(health_status="thermal_anomaly"))
        self.assertEqual(result["feasible_modes"], ["safe"])


class TestCheckGroundPass(unittest.TestCase):
    """Test check_ground_pass tool."""

    def test_pass_active_with_data(self):
        result = check_ground_pass(state=_make_state(
            ground_pass_active=True, obc_data_mb=5.0,
        ))
        self.assertTrue(result["active"])
        self.assertTrue(result["data_ready_for_downlink"])
        self.assertEqual(result["recommendation"], "communicate")

    def test_pass_active_no_data(self):
        result = check_ground_pass(state=_make_state(
            ground_pass_active=True, obc_data_mb=0.0,
        ))
        self.assertTrue(result["active"])
        self.assertFalse(result["data_ready_for_downlink"])
        self.assertIn("no data", result["recommendation"])

    def test_pass_inactive(self):
        result = check_ground_pass(state=_make_state(ground_pass_active=False))
        self.assertFalse(result["active"])
        self.assertIn("not active", result["recommendation"])

    def test_time_to_next_from_metadata(self):
        result = check_ground_pass(state=_make_state(time_to_next_pass=47))
        self.assertIn("47", result["time_to_next"])


class TestCheckDataPipeline(unittest.TestCase):
    """Test check_data_pipeline tool."""

    def test_compression_needed(self):
        result = check_data_pipeline(state=_make_state(uncompressed_observations=3))
        self.assertEqual(result["bottleneck"], "compression_needed")
        self.assertEqual(result["uncompressed"], 3)

    def test_detection_needed(self):
        result = check_data_pipeline(state=_make_state(
            uncompressed_observations=0, undetected_observations=2,
        ))
        self.assertEqual(result["bottleneck"], "detection_needed")

    def test_send_needed(self):
        result = check_data_pipeline(state=_make_state(
            uncompressed_observations=0, undetected_observations=0,
            jetson_compressed_mb=5.0,
        ))
        self.assertEqual(result["bottleneck"], "send_to_obc_needed")

    def test_downlink_needed(self):
        result = check_data_pipeline(state=_make_state(
            uncompressed_observations=0, undetected_observations=0,
            jetson_compressed_mb=0.0, obc_data_mb=10.0,
        ))
        self.assertEqual(result["bottleneck"], "downlink_needed")

    def test_empty_pipeline(self):
        result = check_data_pipeline(state=_make_state())
        self.assertEqual(result["bottleneck"], "none")
        self.assertIn("empty", result["pipeline_summary"])


class TestCheckConstraints(unittest.TestCase):
    """Test check_constraints tool."""

    def test_feasible_mode(self):
        result = check_constraints(
            state=_make_state(battery_soc=0.7),
            proposed_mode="payload_observe",
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(len(result["violations"]), 0)

    def test_communication_without_pass(self):
        result = check_constraints(
            state=_make_state(ground_pass_active=False),
            proposed_mode="communication",
        )
        self.assertFalse(result["feasible"])
        self.assertTrue(any(v["constraint"] == "ground_pass" for v in result["violations"]))

    def test_critical_battery_non_charging(self):
        result = check_constraints(
            state=_make_state(battery_soc=0.15),
            proposed_mode="payload_observe",
        )
        self.assertFalse(result["feasible"])
        self.assertTrue(any(v["constraint"] == "battery_critical" for v in result["violations"]))

    def test_anomaly_non_safe(self):
        result = check_constraints(
            state=_make_state(health_status="power_anomaly"),
            proposed_mode="charging",
        )
        self.assertFalse(result["feasible"])
        self.assertTrue(any(v["constraint"] == "anomaly" for v in result["violations"]))

    def test_invalid_mode(self):
        result = check_constraints(state=_make_state(), proposed_mode="fly_to_mars")
        self.assertFalse(result["feasible"])
        self.assertTrue(any(v["constraint"] == "invalid_mode" for v in result["violations"]))

    def test_low_battery_warning(self):
        result = check_constraints(
            state=_make_state(battery_soc=0.30),
            proposed_mode="payload_observe",
        )
        # Low but above hard limit → feasible with warning
        self.assertTrue(result["feasible"])
        self.assertTrue(len(result["warnings"]) > 0)


class TestRecallHistory(unittest.TestCase):
    """Test recall_history tool."""

    def test_no_memory(self):
        result = recall_history(state=_make_state(), memory=None)
        self.assertEqual(result["last_modes"], [])
        self.assertEqual(result["history_depth"], 0)
        self.assertEqual(result["battery_trend"], "insufficient_data")

    def test_with_memory(self):
        memory = MagicMock()
        memory.query.return_value = [
            {"satellites": {"eventsat_0": {"status": "charging", "resources": {"battery_soc": 0.4}}}},
            {"satellites": {"eventsat_0": {"status": "charging", "resources": {"battery_soc": 0.5}}}},
            {"satellites": {"eventsat_0": {"status": "payload_observe", "resources": {"battery_soc": 0.6}}}},
        ]
        result = recall_history(state=_make_state(), memory=memory, n=5)
        self.assertEqual(result["last_modes"], ["charging", "charging", "payload_observe"])
        self.assertEqual(result["mode_counts"]["charging"], 2)
        self.assertEqual(result["battery_trend"], "rising")

    def test_empty_history(self):
        memory = MagicMock()
        memory.query.return_value = []
        result = recall_history(state=_make_state(), memory=memory)
        self.assertEqual(result["history_depth"], 0)


class TestEvaluatePlan(unittest.TestCase):
    """Test evaluate_plan tool."""

    def test_charging_high_soc(self):
        result = evaluate_plan(state=_make_state(battery_soc=0.8), proposed_mode="charging")
        self.assertLess(result["estimated_utility"], 0.5)

    def test_communication_pass_active_data(self):
        result = evaluate_plan(
            state=_make_state(ground_pass_active=True, obc_data_mb=10.0),
            proposed_mode="communication",
        )
        self.assertGreaterEqual(result["estimated_utility"], 0.8)
        self.assertEqual(result["recommendation"], "proceed")

    def test_communication_no_pass(self):
        result = evaluate_plan(
            state=_make_state(ground_pass_active=False),
            proposed_mode="communication",
        )
        self.assertEqual(result["estimated_utility"], 0.0)
        self.assertTrue(len(result["risk_factors"]) > 0)

    def test_compress_with_data(self):
        result = evaluate_plan(
            state=_make_state(uncompressed_observations=3),
            proposed_mode="payload_compress",
        )
        self.assertGreaterEqual(result["estimated_utility"], 0.5)

    def test_compress_no_data(self):
        result = evaluate_plan(
            state=_make_state(uncompressed_observations=0),
            proposed_mode="payload_compress",
        )
        self.assertLess(result["estimated_utility"], 0.3)


class TestToolRegistry(unittest.TestCase):
    """Test tool registry completeness."""

    def test_all_six_tools_registered(self):
        expected = {
            "check_battery", "check_ground_pass", "check_data_pipeline",
            "check_constraints", "recall_history", "evaluate_plan",
        }
        self.assertEqual(set(TOOL_REGISTRY.keys()), expected)

    def test_execute_unknown_tool(self):
        result = execute_tool("nonexistent", {}, _make_state())
        self.assertIn("error", result)
        self.assertIn("available", result)

    def test_execute_known_tool(self):
        result = execute_tool("check_battery", {}, _make_state())
        self.assertIn("soc", result)


# ======================================================================
# Prompt Tests
# ======================================================================

class TestAgenticPrompts(unittest.TestCase):
    """Test agentic prompt templates."""

    def test_system_prompt_contains_all_tools(self):
        for tool_name in TOOL_REGISTRY:
            self.assertIn(tool_name, AGENTIC_SYSTEM_PROMPT)

    def test_system_prompt_contains_all_modes(self):
        for mode in VALID_MODES:
            self.assertIn(mode, AGENTIC_SYSTEM_PROMPT)

    def test_system_prompt_contains_protocol(self):
        self.assertIn("Plan-Tool-Reflect-Decide", AGENTIC_SYSTEM_PROMPT)
        self.assertIn("tool_call", AGENTIC_SYSTEM_PROMPT)
        self.assertIn("decision", AGENTIC_SYSTEM_PROMPT)

    def test_format_planning_prompt_includes_state(self):
        state = _make_state(battery_soc=0.42)
        prompt = format_planning_prompt(state)
        self.assertIn("0.42", prompt)
        self.assertIn("Battery SoC", prompt)

    def test_format_planning_prompt_empty_state(self):
        prompt = format_planning_prompt({})
        self.assertIn("No satellite state", prompt)

    def test_format_planning_prompt_with_enrichments(self):
        state = _make_state()
        enrichments = {"situation_class": "low_power", "urgency": 0.8}
        prompt = format_planning_prompt(state, enrichments)
        self.assertIn("low_power", prompt)
        self.assertIn("0.80", prompt)

    def test_format_tool_result_prompt_includes_result(self):
        result = {"soc": 0.72, "feasible_modes": ["charging"]}
        prompt = format_tool_result_prompt("check_battery", result, [])
        self.assertIn("check_battery", prompt)
        self.assertIn("0.72", prompt)

    def test_format_agentic_reasoning_prompt(self):
        state = _make_state(battery_soc=0.55)
        prompt = format_agentic_reasoning_prompt(state)
        self.assertIn("0.55", prompt)
        self.assertIn("check", prompt)


# ======================================================================
# AgenticEventSat Representation Tests
# ======================================================================

class TestAgenticEventSat(unittest.TestCase):
    """Test agentic representation core functionality."""

    def setUp(self):
        self.repr = AgenticEventSat(_mock_config())

    def test_registration(self):
        from src.behaviour.controller import BehaviourController
        self.assertIn("agentic_eventsat", BehaviourController.list_registered())

    def test_encode_observation_empty(self):
        result = self.repr.encode_observation(object())
        self.assertEqual(result, {})

    def test_encode_observation_with_constellation(self):
        obs = _make_mock_observation()
        result = self.repr.encode_observation(obs)
        self.assertIn("battery_soc", result)
        self.assertIn("current_mode", result)
        self.assertIn("health_status", result)
        self.assertEqual(result["battery_soc"], 0.65)

    def test_select_action_mock_returns_valid_mode(self):
        ctx = _make_context()
        action = self.repr.select_action(ctx)
        self.assertIn("eventsat_0", action)
        mode = action["eventsat_0"]["mode"]
        self.assertIn(mode, VALID_MODES)

    def test_select_action_anomaly_forces_safe(self):
        ctx = _make_context(state=_make_state(health_status="power_anomaly"))
        action = self.repr.select_action(ctx)
        self.assertEqual(action["eventsat_0"]["mode"], "safe")

    def test_select_action_empty_state_defaults_charging(self):
        ctx = _make_context(state={})
        action = self.repr.select_action(ctx)
        self.assertEqual(action["eventsat_0"]["mode"], "charging")

    def test_select_action_low_soc_forces_charging(self):
        ctx = _make_context(state=_make_state(battery_soc=0.15))
        action = self.repr.select_action(ctx)
        self.assertEqual(action["eventsat_0"]["mode"], "charging")

    def test_select_action_no_pass_blocks_comms(self):
        """Mock fallback returns communication when pass+data, but grounding blocks it."""
        ctx = _make_context(state=_make_state(
            ground_pass_active=False, obc_data_mb=10.0, battery_soc=0.5,
        ))
        action = self.repr.select_action(ctx)
        # Without pass, communication should not be the result
        self.assertNotEqual(action["eventsat_0"]["mode"], "communication")

    def test_get_rationale_after_action(self):
        ctx = _make_context()
        self.repr.select_action(ctx)
        rationale = self.repr.get_rationale()
        self.assertIsNotNone(rationale)
        self.assertIsInstance(rationale, str)
        self.assertTrue(len(rationale) > 0)

    def test_get_metrics_includes_agentic_fields(self):
        ctx = _make_context()
        self.repr.select_action(ctx)
        metrics = self.repr.get_metrics()
        self.assertIn("agentic_total_tool_calls", metrics)
        self.assertIn("agentic_avg_steps_per_decision", metrics)
        self.assertIn("agentic_grounding_overrides", metrics)
        self.assertIn("agentic_total_decisions", metrics)

    def test_get_metrics_includes_llm_fields(self):
        metrics = self.repr.get_metrics()
        self.assertIn("llm_api_calls", metrics)
        self.assertIn("llm_cache_hits", metrics)

    def test_get_name(self):
        self.assertEqual(self.repr.get_name(), "AgenticEventSat")

    def test_max_agentic_steps_config(self):
        repr2 = AgenticEventSat(_mock_config(max_agentic_steps=5))
        self.assertEqual(repr2._max_agentic_steps, 5)

    def test_mock_mode_no_tool_calls(self):
        """In mock mode, no tools should be invoked."""
        ctx = _make_context()
        self.repr.select_action(ctx)
        metrics = self.repr.get_metrics()
        self.assertEqual(metrics["agentic_total_tool_calls"], 0.0)

    def test_reason_returns_list(self):
        state = _make_state()
        result = self.repr.reason(state, None)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        # Each entry should have check, value, implication
        for entry in result:
            self.assertIn("check", entry)
            self.assertIn("value", entry)
            self.assertIn("implication", entry)

    def test_reason_empty_state(self):
        result = self.repr.reason({}, None)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["check"], "state")

    def test_symbolic_fallback_low_battery(self):
        mode = self.repr._symbolic_fallback(_make_state(battery_soc=0.20))
        self.assertEqual(mode, "charging")

    def test_symbolic_fallback_pass_with_data(self):
        mode = self.repr._symbolic_fallback(
            _make_state(battery_soc=0.7, ground_pass_active=True, obc_data_mb=5.0)
        )
        self.assertEqual(mode, "communication")


# ======================================================================
# Symbolic Grounding Tests
# ======================================================================

class TestAgenticGrounding(unittest.TestCase):
    """Test symbolic grounding constraints (same rules as llm_eventsat)."""

    def setUp(self):
        self.repr = AgenticEventSat(_mock_config())

    def test_grounding_no_pass_blocks_communication(self):
        mode = self.repr._apply_grounding(
            "communication", _make_state(ground_pass_active=False)
        )
        self.assertEqual(mode, "charging")

    def test_grounding_allows_communication_during_pass(self):
        mode = self.repr._apply_grounding(
            "communication", _make_state(ground_pass_active=True)
        )
        self.assertEqual(mode, "communication")

    def test_grounding_very_low_soc_forces_charging(self):
        mode = self.repr._apply_grounding(
            "payload_observe", _make_state(battery_soc=0.15)
        )
        self.assertEqual(mode, "charging")

    def test_grounding_allows_mode_with_good_soc(self):
        mode = self.repr._apply_grounding(
            "payload_observe", _make_state(battery_soc=0.7)
        )
        self.assertEqual(mode, "payload_observe")

    def test_grounding_override_counter_increments(self):
        initial = self.repr._grounding_overrides
        self.repr._apply_grounding("communication", _make_state(ground_pass_active=False))
        self.assertEqual(self.repr._grounding_overrides, initial + 1)


# ======================================================================
# Decision Loop Integration Tests
# ======================================================================

class TestAgenticWithLoops(unittest.TestCase):
    """Test agentic representation with all 3 decision loop types."""

    def setUp(self):
        self.repr = AgenticEventSat(_mock_config())

    def test_with_sda_context(self):
        ctx = _make_context(loop_type="sda")
        action = self.repr.select_action(ctx)
        self.assertIn("eventsat_0", action)
        self.assertIn(action["eventsat_0"]["mode"], VALID_MODES)

    def test_with_ooda_context(self):
        ctx = _make_context(
            loop_type="ooda",
            enrichments={"situation_class": "low_power", "urgency": 0.7},
        )
        action = self.repr.select_action(ctx)
        self.assertIn("eventsat_0", action)
        self.assertIn(action["eventsat_0"]["mode"], VALID_MODES)

    def test_with_react_context(self):
        ctx = _make_context(
            loop_type="react",
            enrichments={
                "reasoning_trace": [{"check": "battery", "value": 0.7, "implication": "ok"}],
                "iteration": 1,
                "grounding_violations": [],
            },
        )
        action = self.repr.select_action(ctx)
        self.assertIn("eventsat_0", action)
        self.assertIn(action["eventsat_0"]["mode"], VALID_MODES)

    def test_with_react_reason_called(self):
        state = _make_state()
        result = self.repr.reason(state, None)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) >= 3)  # battery + pass + pipeline

    def test_with_ag_ops_paradigm_config(self):
        """Test that agentic representation works with AG ops config."""
        repr_ag = AgenticEventSat(_mock_config())
        ctx = _make_context()
        action = repr_ag.select_action(ctx)
        self.assertIn("eventsat_0", action)

    def test_with_cg_ops_paradigm_config(self):
        """Test that agentic representation works with CG ops config."""
        repr_cg = AgenticEventSat(_mock_config())
        ctx = _make_context()
        action = repr_cg.select_action(ctx)
        self.assertIn("eventsat_0", action)


# ======================================================================
# Emergence Controller Tests
# ======================================================================

class TestAgenticEmergence(unittest.TestCase):
    """Test emergence controller integration."""

    def test_emergence_controller_creates_agentic(self):
        from src.behaviour.controller import BehaviourController
        controller = BehaviourController(config=_mock_config())
        repr_obj = controller.get_representation("agentic_eventsat")
        self.assertIsInstance(repr_obj, AgenticEventSat)

    def test_emergence_controller_lists_agentic(self):
        from src.behaviour.controller import BehaviourController
        registered = BehaviourController.list_registered()
        self.assertIn("agentic_eventsat", registered)


# ======================================================================
# Agentic Loop with Patched LLM Tests
# ======================================================================

class TestAgenticLoopWithPatchedLLM(unittest.TestCase):
    """Test the full agentic loop by patching LLMClient.generate."""

    def test_immediate_decision(self):
        """LLM decides on first call without tool use."""
        repr_obj = AgenticEventSat({"llm_mock": False, "llm_provider": "auto"})
        response = json.dumps({
            "decision": {"mode": "charging", "rationale": "Battery low, need to charge."},
        })
        with patch.object(repr_obj._client, "generate", return_value=response):
            repr_obj._client.mock_mode = False
            ctx = _make_context()
            action = repr_obj.select_action(ctx)
            self.assertEqual(action["eventsat_0"]["mode"], "charging")
            # Only 1 LLM call needed
            self.assertEqual(repr_obj._client.generate.call_count, 1)

    def test_plan_then_tool_then_decide(self):
        """LLM plans → tool call → reflects with decision."""
        repr_obj = AgenticEventSat({"llm_mock": False, "llm_provider": "auto"})

        plan_response = json.dumps({
            "plan": "Need to check battery before deciding.",
            "tool_call": {"name": "check_battery", "args": {}},
        })
        reflect_response = json.dumps({
            "reflection": "Battery is good. Observe.",
            "decision": {"mode": "payload_observe", "rationale": "SoC sufficient for observation."},
        })

        call_count = [0]
        def mock_generate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return plan_response
            return reflect_response

        with patch.object(repr_obj._client, "generate", side_effect=mock_generate):
            repr_obj._client.mock_mode = False
            ctx = _make_context()
            action = repr_obj.select_action(ctx)
            self.assertEqual(action["eventsat_0"]["mode"], "payload_observe")
            self.assertEqual(call_count[0], 2)
            # Check tool was called
            metrics = repr_obj.get_metrics()
            self.assertEqual(metrics["agentic_total_tool_calls"], 1.0)

    def test_budget_exhaustion_triggers_forced_decide(self):
        """Tool requests until budget runs out → one decision-only call closes the loop."""
        repr_obj = AgenticEventSat({
            "llm_mock": False, "llm_provider": "auto", "max_agentic_steps": 2,
        })

        plan_response = json.dumps({
            "plan": "Check battery.",
            "tool_call": {"name": "check_battery", "args": {}},
        })
        reflect_response = json.dumps({
            "reflection": "Need more info.",
            "tool_call": {"name": "check_data_pipeline", "args": {}},
        })
        forced_response = json.dumps({
            "decision": {"mode": "charging", "rationale": "Battery first."},
        })

        call_count = [0]
        def mock_generate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return plan_response
            if call_count[0] == 2:
                return reflect_response
            return forced_response

        with patch.object(repr_obj._client, "generate", side_effect=mock_generate):
            repr_obj._client.mock_mode = False
            ctx = _make_context()
            action = repr_obj.select_action(ctx)
            self.assertEqual(action["eventsat_0"]["mode"], "charging")
            # plan + 1 reflect (budget 2) + 1 forced decide
            self.assertEqual(call_count[0], 3)
            self.assertIn("Decide:", repr_obj.get_rationale())

    def test_no_decision_after_forced_decide_fails_episode(self):
        """Even the forced decide returns no decision → substrate-integrity RuntimeError."""
        repr_obj = AgenticEventSat({
            "llm_mock": False, "llm_provider": "auto", "max_agentic_steps": 2,
        })

        tool_only = json.dumps({
            "reflection": "Still exploring.",
            "tool_call": {"name": "check_battery", "args": {}},
        })

        with patch.object(repr_obj._client, "generate", return_value=tool_only):
            repr_obj._client.mock_mode = False
            ctx = _make_context()
            with self.assertRaises(RuntimeError) as cm:
                repr_obj.select_action(ctx)
            self.assertIn("integrity violation", str(cm.exception))

    def test_malformed_response_fails_episode(self):
        """LLM returns garbage on every call (incl. forced decide) → RuntimeError, no fallback."""
        repr_obj = AgenticEventSat({"llm_mock": False, "llm_provider": "auto"})

        with patch.object(repr_obj._client, "generate", return_value="not json at all"):
            repr_obj._client.mock_mode = False
            ctx = _make_context()
            with self.assertRaises(RuntimeError) as cm:
                repr_obj.select_action(ctx)
            self.assertIn("integrity violation", str(cm.exception))

    def test_grounding_applied_after_llm_decision(self):
        """LLM says 'communication' but no pass → grounding overrides."""
        repr_obj = AgenticEventSat({"llm_mock": False, "llm_provider": "auto"})
        response = json.dumps({
            "decision": {"mode": "communication", "rationale": "Downlink data."},
        })
        with patch.object(repr_obj._client, "generate", return_value=response):
            repr_obj._client.mock_mode = False
            ctx = _make_context(state=_make_state(ground_pass_active=False))
            action = repr_obj.select_action(ctx)
            # Grounding should override to charging
            self.assertEqual(action["eventsat_0"]["mode"], "charging")

    def test_rationale_includes_chain(self):
        """Rationale should reflect the reasoning chain."""
        repr_obj = AgenticEventSat({"llm_mock": False, "llm_provider": "auto"})

        plan_response = json.dumps({
            "plan": "Check constraints first.",
            "tool_call": {"name": "check_constraints", "args": {"proposed_mode": "charging"}},
        })
        reflect_response = json.dumps({
            "reflection": "Charging is feasible.",
            "decision": {"mode": "charging", "rationale": "Safe choice."},
        })

        responses = iter([plan_response, reflect_response])
        with patch.object(repr_obj._client, "generate", side_effect=lambda *a, **k: next(responses)):
            repr_obj._client.mock_mode = False
            ctx = _make_context()
            repr_obj.select_action(ctx)
            rationale = repr_obj.get_rationale()
            self.assertIn("Agentic", rationale)
            self.assertIn("Plan", rationale)
            self.assertIn("Tool", rationale)


# ======================================================================
# Mock observation helper
# ======================================================================

def _make_mock_observation(soc: float = 0.65, mode: str = "charging") -> Any:
    """Create a mock observation with constellation_state."""
    class MockSat:
        def __init__(self):
            self.status = mode
            self.resources = {"battery_soc": soc, "data_stored_mb": 5.0, "obc_data_mb": 2.0}
            self.metadata = {
                "in_sunlight": True,
                "ground_pass_active": False,
                "jetson_raw_mb": 0.0,
                "jetson_compressed_mb": 0.0,
                "storage_capacity_mb": 512.0,
                "uncompressed_observations": 0,
                "compression_progress": 0,
                "total_observation_s": 100.0,
                "health_status": "nominal",
                "undetected_observations": 0,
                "daily_downlink_budget_mb": 27.0,
            }

    class MockConstellation:
        def __init__(self):
            self.satellites = {"eventsat_0": MockSat()}

    class MockObs:
        def __init__(self):
            self.constellation_state = MockConstellation()

    return MockObs()


if __name__ == "__main__":
    unittest.main()

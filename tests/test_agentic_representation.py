"""
Tests for agentic hybrid representation (Phase 4c).

All tests use mock mode (no live LLM calls) to ensure CI compatibility.
"""
from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.core.decision_procedure.context import DecisionContext
from src.eventsat.agentic_tools import (
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
from src.eventsat.agentic_prompts import (
    AGENTIC_SYSTEM_PROMPT,
    format_agentic_reasoning_prompt,
    format_planning_prompt,
    format_tool_result_prompt,
)
from src.eventsat.agentic import AgenticEventSat


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
        "jetson_capacity_mb": 249036.8,
        "observation_size_mb": 9.41,
        "compression_ratio": 5.11,
        "detection_metadata_mb": 0.01,
        "jetson_to_obc_rate_kbps": 8000.0,
        "downlink_rate_kbps": 50.0,
        "step_duration_s": 60.0,
        "uncompressed_observations": 0,
        "compression_progress": 0,
        "total_observation_s": 0.0,
        "health_status": "nominal",
        "undetected_observations": 0,
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

    def test_critical_battery_charging_request_matches_environment_safe_resolution(self):
        state = _make_state(battery_soc=0.25, battery_min_soc=0.25)

        charging = check_constraints(state=state, proposed_mode="charging")
        safe = check_constraints(state=state, proposed_mode="safe")

        self.assertFalse(charging["feasible"])
        self.assertTrue(any(
            v["constraint"] == "battery_critical"
            for v in charging["violations"]
        ))
        self.assertTrue(safe["feasible"])

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

    def test_observation_below_environment_mode_floor_is_infeasible(self):
        result = check_constraints(
            state=_make_state(battery_soc=0.30),
            proposed_mode="payload_observe",
        )
        self.assertFalse(result["feasible"])
        self.assertTrue(
            any(v["constraint"] == "mode_battery" for v in result["violations"])
        )

    def test_low_battery_warning_for_mode_still_permitted_by_environment(self):
        result = check_constraints(
            state=_make_state(
                battery_soc=0.30,
                uncompressed_observations=1,
                jetson_raw_mb=9.41,
            ),
            proposed_mode="payload_compress",
        )
        self.assertTrue(result["feasible"])
        self.assertTrue(len(result["warnings"]) > 0)

    def test_projected_shared_jetson_overflow_is_infeasible(self):
        result = check_constraints(
            state=_make_state(
                jetson_capacity_mb=10.0,
                jetson_raw_mb=5.0,
                jetson_compressed_mb=0.0,
            ),
            proposed_mode="payload_observe",
        )
        self.assertFalse(result["feasible"])
        self.assertTrue(
            any(v["constraint"] == "jetson_capacity" for v in result["violations"])
        )

    def test_partial_detection_defers_obc_admission_until_completion(self):
        partial = check_constraints(
            state=_make_state(
                undetected_observations=1,
                detection_progress=0,
                detection_steps=5,
                obc_data_mb=512.0,
            ),
            proposed_mode="payload_detect",
        )
        completing = check_constraints(
            state=_make_state(
                undetected_observations=1,
                detection_progress=4,
                detection_steps=5,
                obc_data_mb=512.0,
            ),
            proposed_mode="payload_detect",
        )

        self.assertTrue(partial["feasible"])
        self.assertTrue(partial["productive_this_step"])
        self.assertFalse(completing["feasible"])
        self.assertTrue(any(
            v["constraint"] == "obc_capacity"
            for v in completing["violations"]
        ))

    def test_communication_contact_must_outlast_adcs_settling(self):
        result = check_constraints(
            state=_make_state(
                ground_pass_active=True,
                contact_window_seconds=30.0,
                remaining_pass_duration_s=30.0,
                obc_data_mb=1.0,
                settling_time_steps=2,
                transition_steps_remaining=0,
                previous_mode="charging",
                attitude_maneuver_modes=["communication", "payload_observe"],
            ),
            proposed_mode="communication",
        )

        self.assertFalse(result["feasible"])
        self.assertFalse(result["productive_this_step"])
        self.assertEqual(result["resolved_mode_this_step"], "charging")
        self.assertEqual(result["transition_steps_required"], 2)
        self.assertTrue(any(
            v["constraint"] == "contact_too_short"
            for v in result["violations"]
        ))


    def _slew_state(self, **overrides):
        return _make_state(
            settling_time_steps=2,
            transition_steps_remaining=1,
            transition_target_mode="payload_observe",
            previous_mode="charging",
            attitude_maneuver_modes=["communication", "payload_observe"],
            **overrides,
        )

    def test_command_during_slew_is_ignored_not_a_violation(self):
        for mode in ("communication", "safe", "payload_observe"):
            result = check_constraints(
                state=self._slew_state(battery_soc=0.35), proposed_mode=mode
            )
            self.assertTrue(result["command_ignored"])
            self.assertTrue(result["feasible"])
            self.assertEqual(result["violations"], [])
            self.assertFalse(result["productive_this_step"])
            self.assertEqual(result["resolved_mode_this_step"], "charging")
            self.assertEqual(result["transition_target_mode"], "payload_observe")

    def test_forced_safe_preempts_slew(self):
        result = check_constraints(
            state=self._slew_state(battery_soc=0.15), proposed_mode="safe"
        )
        self.assertFalse(result["command_ignored"])
        self.assertEqual(result["resolved_mode_this_step"], "safe")
        self.assertEqual(result["transition_steps_required"], 0)

    def test_evaluate_plan_treats_commands_during_slew_as_equivalent(self):
        results = {
            mode: evaluate_plan(state=self._slew_state(), proposed_mode=mode)
            for mode in ("charging", "payload_observe", "communication")
        }
        self.assertEqual(
            {r["recommendation"] for r in results.values()}, {"proceed"}
        )
        self.assertTrue(all(r["command_ignored"] for r in results.values()))
        self.assertEqual({r["estimated_utility"] for r in results.values()}, {0.0})


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

    def test_compression_value_is_derived_from_next_pass_deliverability(self):
        result = evaluate_plan(
            state=_make_state(
                uncompressed_observations=1,
                jetson_raw_mb=9.41,
                compression_progress=1,
                compression_time_factor=2,
                time_to_next_pass=2,
                achievable_downlink_mb=1.0,
            ),
            proposed_mode="payload_compress",
        )

        self.assertEqual(result["estimated_utility"], 1.0)
        self.assertEqual(result["recommendation"], "proceed")
        self.assertEqual(result["estimated_deliverable_mb"], 1.0)
        self.assertAlmostEqual(result["pipeline_progress_mb"], 9.41 / 5.11, places=6)
        self.assertEqual(result["processing_progress_steps"], 0)

    def test_compress_no_data(self):
        result = evaluate_plan(
            state=_make_state(uncompressed_observations=0),
            proposed_mode="payload_compress",
        )
        self.assertLess(result["estimated_utility"], 0.3)
        self.assertEqual(result["recommendation"], "not_proceed")

    def test_rejected_observation_has_no_utility_and_is_not_recommended(self):
        result = evaluate_plan(
            state=_make_state(
                jetson_capacity_mb=10.0,
                jetson_raw_mb=5.0,
                jetson_compressed_mb=0.0,
            ),
            proposed_mode="payload_observe",
        )
        self.assertFalse(result["feasible"])
        self.assertEqual(result["estimated_utility"], 0.0)
        self.assertEqual(result["estimated_deliverable_mb"], 0.0)
        self.assertEqual(result["pipeline_progress_mb"], 0.0)
        self.assertEqual(result["recommendation"], "not_proceed")

    def test_observation_value_requires_enough_steps_to_reach_next_pass(self):
        common = {
            "achievable_downlink_mb": 1.0,
            "compression_time_factor": 2,
        }
        deliverable = evaluate_plan(
            state=_make_state(**common, time_to_next_pass=4),
            proposed_mode="payload_observe",
        )
        too_late = evaluate_plan(
            state=_make_state(**common, time_to_next_pass=2),
            proposed_mode="payload_observe",
        )

        self.assertEqual(deliverable["estimated_deliverable_mb"], 1.0)
        self.assertEqual(deliverable["estimated_utility"], 1.0)
        self.assertEqual(too_late["estimated_deliverable_mb"], 0.0)
        self.assertEqual(too_late["estimated_utility"], 0.0)

    def test_adcs_settling_never_receives_same_step_delivery_credit(self):
        state = _make_state(
            ground_pass_active=True,
            contact_window_seconds=60.0,
            remaining_pass_duration_s=180.0,
            obc_data_mb=1.0,
            settling_time_steps=2,
            transition_steps_remaining=0,
            previous_mode="charging",
            attitude_maneuver_modes=["communication", "payload_observe"],
        )

        result = evaluate_plan(state=state, proposed_mode="communication")

        self.assertTrue(result["feasible"])
        self.assertFalse(result["productive_this_step"])
        self.assertEqual(result["resolved_mode_this_step"], "charging")
        self.assertEqual(result["estimated_immediate_delivery_mb"], 0.0)
        self.assertEqual(result["estimated_deliverable_mb"], 0.0)
        self.assertEqual(result["estimated_utility"], 0.0)
        self.assertEqual(result["recommendation"], "proceed")

    def test_named_safe_or_charging_modes_have_no_fixed_utility_bonus(self):
        charging = evaluate_plan(
            state=_make_state(battery_soc=0.25), proposed_mode="charging"
        )
        safe = evaluate_plan(
            state=_make_state(health_status="power_anomaly"),
            proposed_mode="safe",
        )

        self.assertEqual(charging["estimated_utility"], 0.0)
        self.assertEqual(safe["estimated_utility"], 0.0)
        self.assertEqual(charging["recommendation"], "proceed")
        self.assertEqual(safe["recommendation"], "proceed")


class TestToolRegistry(unittest.TestCase):
    """Test tool registry completeness."""

    def test_agentic_tools_registered(self):
        # Only the what-if / memory tools are agentic tools. The echo tools
        # (check_battery / check_ground_pass / check_data_pipeline) are folded
        # into the planning prompt and are no longer registered.
        expected = {"check_constraints", "recall_history", "evaluate_plan"}
        self.assertEqual(set(TOOL_REGISTRY.keys()), expected)

    def test_execute_unknown_tool(self):
        result = execute_tool("nonexistent", {}, _make_state())
        self.assertIn("error", result)
        self.assertIn("available", result)

    def test_execute_known_tool(self):
        result = execute_tool(
            "check_constraints", {"proposed_mode": "charging"}, _make_state()
        )
        self.assertIn("feasible", result)


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
        from src.core.behaviour.controller import BehaviourController
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

    def test_reason_helper(self):
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
# Behaviour Factory Tests
# ======================================================================

class TestAgenticBehaviourFactory(unittest.TestCase):
    """Test behaviour factory integration."""

    def test_behaviour_factory_creates_agentic(self):
        from src.core.behaviour.controller import BehaviourController
        controller = BehaviourController(config=_mock_config())
        repr_obj = controller.get_representation("agentic_eventsat")
        self.assertIsInstance(repr_obj, AgenticEventSat)

    def test_behaviour_factory_lists_agentic(self):
        from src.core.behaviour.controller import BehaviourController
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
            "plan": "Validate observe before deciding.",
            "tool_call": {"name": "check_constraints", "args": {"proposed_mode": "payload_observe"}},
        })
        reflect_response = json.dumps({
            "reflection": "Observe is feasible.",
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
            "plan": "Validate charging.",
            "tool_call": {"name": "check_constraints", "args": {"proposed_mode": "charging"}},
        })
        reflect_response = json.dumps({
            "reflection": "Need more info.",
            "tool_call": {"name": "evaluate_plan", "args": {"proposed_mode": "charging"}},
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
            "tool_call": {"name": "check_constraints", "args": {"proposed_mode": "charging"}},
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

"""
Tests for LLM hybrid representation (Phase 4a).

All tests use mock mode (no live LLM calls) to ensure CI compatibility.
"""
from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from src.decision_loop.context import DecisionContext
from src.representation.llm_client import LLMClient
from src.representation.llm_eventsat import LLMEventSat, VALID_MODES
from src.representation.llm_prompts import (
    SYSTEM_PROMPT,
    format_reasoning_prompt,
    format_state_prompt,
)


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
# LLM Client Tests
# ======================================================================

class TestLLMClient(unittest.TestCase):
    """Test LLM client mock mode and caching."""

    def test_mock_mode_returns_valid_json(self):
        client = LLMClient({"llm_mock": True})
        response = client.generate("system", "user prompt")
        parsed = json.loads(response)
        self.assertIn("mode", parsed)
        self.assertIn("rationale", parsed)

    def test_mock_mode_tracks_metrics(self):
        client = LLMClient({"llm_mock": True})
        client.generate("system", "prompt 1")
        client.generate("system", "prompt 2")
        metrics = client.get_metrics()
        self.assertEqual(metrics["llm_api_calls"], 2.0)
        self.assertEqual(metrics["llm_last_latency_s"], 0.001)

    def test_cache_key_deterministic(self):
        client = LLMClient({"llm_mock": True})
        k1 = client._cache_key("sys", "user", 0.0)
        k2 = client._cache_key("sys", "user", 0.0)
        self.assertEqual(k1, k2)

    def test_cache_key_varies_with_prompt(self):
        client = LLMClient({"llm_mock": True})
        k1 = client._cache_key("sys", "prompt A", 0.0)
        k2 = client._cache_key("sys", "prompt B", 0.0)
        self.assertNotEqual(k1, k2)

    def test_cache_key_varies_with_model(self):
        c1 = LLMClient({"llm_mock": True, "llm_model": "model_a"})
        c2 = LLMClient({"llm_mock": True, "llm_model": "model_b"})
        k1 = c1._cache_key("sys", "prompt", 0.0)
        k2 = c2._cache_key("sys", "prompt", 0.0)
        self.assertNotEqual(k1, k2)


# ======================================================================
# Prompt Template Tests
# ======================================================================

class TestPromptTemplates(unittest.TestCase):
    """Test prompt formatting functions."""

    def test_system_prompt_contains_modes(self):
        for mode in VALID_MODES:
            self.assertIn(mode, SYSTEM_PROMPT)

    def test_system_prompt_requests_json(self):
        self.assertIn("JSON", SYSTEM_PROMPT)

    def test_format_state_empty(self):
        prompt = format_state_prompt({})
        self.assertIn("safest mode", prompt)

    def test_format_state_includes_soc(self):
        state = _make_state(battery_soc=0.42)
        prompt = format_state_prompt(state)
        self.assertIn("0.42", prompt)

    def test_format_state_includes_pass_status(self):
        state = _make_state(ground_pass_active=True)
        prompt = format_state_prompt(state)
        self.assertIn("YES", prompt)

    def test_format_state_with_ooda_enrichments(self):
        state = _make_state()
        enrichments = {
            "situation_class": "nominal_sunlight",
            "urgency": 0.3,
            "battery_trending_down": False,
        }
        prompt = format_state_prompt(state, enrichments)
        self.assertIn("SITUATION ASSESSMENT", prompt)
        self.assertIn("nominal_sunlight", prompt)
        self.assertIn("0.30", prompt)
        self.assertIn("stable/rising", prompt)

    def test_format_state_with_react_enrichments(self):
        state = _make_state()
        enrichments = {
            "reasoning_steps": [
                {"check": "battery", "value": 0.7, "implication": "ok"},
                {"check": "pass", "value": False, "implication": "no_pass"},
            ]
        }
        prompt = format_state_prompt(state, enrichments)
        self.assertIn("Prior reasoning steps: 2", prompt)

    def test_format_reasoning_prompt_empty_state(self):
        prompt = format_reasoning_prompt({}, None)
        self.assertIn("safe mode", prompt)

    def test_format_reasoning_prompt_includes_state(self):
        state = _make_state(battery_soc=0.55, health_status="nominal")
        prompt = format_reasoning_prompt(state, None)
        self.assertIn("SoC=0.55", prompt)
        self.assertIn("JSON", prompt)


# ======================================================================
# LLM EventSat Representation Tests
# ======================================================================

class TestLLMEventSat(unittest.TestCase):
    """Test the LLM hybrid representation (mock mode)."""

    def setUp(self):
        self.rep = LLMEventSat(_mock_config())

    def test_registration(self):
        from src.emergence.controller import _REPRESENTATION_REGISTRY
        self.assertIn("llm_eventsat", _REPRESENTATION_REGISTRY)

    def test_encode_observation_returns_dict(self):
        # Test with empty observation
        result = self.rep.encode_observation(object())
        self.assertEqual(result, {})

    def test_encode_observation_with_constellation(self):
        """Test encode_observation with a mock observation object."""
        sat = MagicMock()
        sat.status = "charging"
        sat.resources = {"battery_soc": 0.65, "data_stored_mb": 15.0}
        sat.metadata = {
            "in_sunlight": True,
            "ground_pass_active": False,
            "obc_data_mb": 2.0,
            "jetson_raw_mb": 0.0,
            "jetson_compressed_mb": 0.0,
            "storage_capacity_mb": 512.0,
            "uncompressed_observations": 1,
            "compression_progress": 0,
            "total_observation_s": 60.0,
            "health_status": "nominal",
            "undetected_observations": 0,
            "daily_downlink_budget_mb": 27.0,
        }
        obs = MagicMock()
        obs.constellation_state.satellites = {"eventsat_0": sat}

        result = self.rep.encode_observation(obs)
        self.assertEqual(result["battery_soc"], 0.65)
        self.assertEqual(result["obc_data_mb"], 2.0)
        self.assertTrue(result["in_sunlight"])

    def test_select_action_mock_returns_valid_mode(self):
        ctx = _make_context()
        action = self.rep.select_action(ctx)
        self.assertIn("eventsat_0", action)
        mode = action["eventsat_0"]["mode"]
        self.assertIn(mode, VALID_MODES)

    def test_select_action_anomaly_forces_safe(self):
        state = _make_state(health_status="anomaly_power")
        ctx = _make_context(state)
        action = self.rep.select_action(ctx)
        self.assertEqual(action["eventsat_0"]["mode"], "safe")

    def test_select_action_empty_state_defaults_charging(self):
        ctx = _make_context(state={})
        action = self.rep.select_action(ctx)
        self.assertEqual(action["eventsat_0"]["mode"], "charging")

    def test_get_rationale_after_action(self):
        ctx = _make_context()
        self.rep.select_action(ctx)
        rationale = self.rep.get_rationale()
        self.assertIsNotNone(rationale)
        self.assertIsInstance(rationale, str)

    def test_get_metrics_includes_llm_fields(self):
        ctx = _make_context()
        self.rep.select_action(ctx)
        metrics = self.rep.get_metrics()
        self.assertIn("llm_api_calls", metrics)
        self.assertIn("llm_cache_hit_rate", metrics)
        self.assertIn("llm_grounding_overrides", metrics)

    def test_get_name(self):
        self.assertEqual(self.rep.get_name(), "LLMEventSat")

    def test_reason_returns_list(self):
        state = _make_state()
        result = self.rep.reason(state, None)
        self.assertIsInstance(result, list)

    def test_reason_empty_state(self):
        result = self.rep.reason({}, None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["check"], "state")


# ======================================================================
# Symbolic Grounding Tests
# ======================================================================

class TestSymbolicGrounding(unittest.TestCase):
    """Test symbolic grounding constraints on LLM output."""

    def setUp(self):
        self.rep = LLMEventSat(_mock_config())

    def test_grounding_no_pass_blocks_communication(self):
        """Communication mode should be overridden when no ground pass is active."""
        state = _make_state(ground_pass_active=False)
        result = self.rep._apply_grounding("communication", state)
        self.assertEqual(result, "charging")

    def test_grounding_allows_communication_during_pass(self):
        state = _make_state(ground_pass_active=True)
        result = self.rep._apply_grounding("communication", state)
        self.assertEqual(result, "communication")

    def test_grounding_very_low_soc_forces_charging(self):
        state = _make_state(battery_soc=0.15)
        result = self.rep._apply_grounding("payload_observe", state)
        self.assertEqual(result, "charging")

    def test_grounding_allows_mode_with_good_soc(self):
        state = _make_state(battery_soc=0.60)
        result = self.rep._apply_grounding("payload_observe", state)
        self.assertEqual(result, "payload_observe")

    def test_symbolic_fallback_low_battery(self):
        state = _make_state(battery_soc=0.25)
        result = self.rep._symbolic_fallback(state)
        self.assertEqual(result, "charging")

    def test_symbolic_fallback_pass_with_data(self):
        state = _make_state(ground_pass_active=True, obc_data_mb=5.0)
        result = self.rep._symbolic_fallback(state)
        self.assertEqual(result, "communication")

    def test_symbolic_fallback_default_charging(self):
        state = _make_state()
        result = self.rep._symbolic_fallback(state)
        self.assertEqual(result, "charging")


# ======================================================================
# LLM Response Parsing Tests
# ======================================================================

class TestResponseParsing(unittest.TestCase):
    """Test JSON parsing of LLM responses."""

    def setUp(self):
        self.rep = LLMEventSat(_mock_config())

    def test_parse_clean_json(self):
        raw = '{"mode": "charging", "rationale": "low battery"}'
        parsed = self.rep._parse_response(raw)
        self.assertEqual(parsed["mode"], "charging")

    def test_parse_json_with_markdown_fences(self):
        raw = '```json\n{"mode": "charging", "rationale": "test"}\n```'
        parsed = self.rep._parse_response(raw)
        self.assertEqual(parsed["mode"], "charging")

    def test_parse_json_with_whitespace(self):
        raw = '  \n  {"mode": "payload_observe", "rationale": "clear sky"}  \n  '
        parsed = self.rep._parse_response(raw)
        self.assertEqual(parsed["mode"], "payload_observe")

    def test_parse_invalid_json_raises(self):
        with self.assertRaises(json.JSONDecodeError):
            self.rep._parse_response("not valid json at all")


# ======================================================================
# LLM with Realistic Responses (patched)
# ======================================================================

class TestLLMWithPatchedResponses(unittest.TestCase):
    """Test LLM representation with controlled (patched) LLM responses."""

    def _make_rep_with_response(self, response_json: Dict[str, Any]) -> LLMEventSat:
        """Create a representation with a patched LLM client."""
        rep = LLMEventSat(_mock_config())
        rep._client.generate = MagicMock(return_value=json.dumps(response_json))
        rep._client._total_calls = 0  # Reset mock counter
        return rep

    def test_llm_selects_observe(self):
        rep = self._make_rep_with_response(
            {"mode": "payload_observe", "rationale": "Good battery and clear sky."}
        )
        ctx = _make_context(_make_state(battery_soc=0.8))
        action = rep.select_action(ctx)
        self.assertEqual(action["eventsat_0"]["mode"], "payload_observe")

    def test_llm_selects_communication(self):
        rep = self._make_rep_with_response(
            {"mode": "communication", "rationale": "Ground pass with data ready."}
        )
        ctx = _make_context(_make_state(ground_pass_active=True, obc_data_mb=5.0))
        action = rep.select_action(ctx)
        self.assertEqual(action["eventsat_0"]["mode"], "communication")

    def test_llm_invalid_mode_retries(self):
        rep = LLMEventSat(_mock_config())
        # First call returns invalid, second returns valid
        rep._client.generate = MagicMock(side_effect=[
            json.dumps({"mode": "invalid_mode", "rationale": "oops"}),
            json.dumps({"mode": "charging", "rationale": "fixed"}),
        ])
        ctx = _make_context()
        action = rep.select_action(ctx)
        self.assertEqual(action["eventsat_0"]["mode"], "charging")
        self.assertEqual(rep._last_parse_retries, 1)

    def test_llm_all_retries_fail_uses_fallback(self):
        rep = LLMEventSat(_mock_config())
        rep._client.generate = MagicMock(side_effect=RuntimeError("LLM down"))
        ctx = _make_context(_make_state(battery_soc=0.7))
        action = rep.select_action(ctx)
        # Fallback should return a valid mode
        self.assertIn(action["eventsat_0"]["mode"], VALID_MODES)
        self.assertGreater(rep._grounding_overrides, 0)

    def test_grounding_overrides_communication_without_pass(self):
        rep = self._make_rep_with_response(
            {"mode": "communication", "rationale": "LLM wants to communicate."}
        )
        ctx = _make_context(_make_state(ground_pass_active=False))
        action = rep.select_action(ctx)
        # Grounding should override: no pass → can't communicate
        self.assertEqual(action["eventsat_0"]["mode"], "charging")

    def test_grounding_overrides_on_critical_battery(self):
        rep = self._make_rep_with_response(
            {"mode": "payload_observe", "rationale": "Want to observe."}
        )
        ctx = _make_context(_make_state(battery_soc=0.15))
        action = rep.select_action(ctx)
        # Grounding should override: SoC < 0.20 → forced charging
        self.assertEqual(action["eventsat_0"]["mode"], "charging")


# ======================================================================
# Loop Integration Tests
# ======================================================================

class TestLoopIntegration(unittest.TestCase):
    """Test LLM representation works with all decision loop types."""

    def test_sda_context(self):
        rep = LLMEventSat(_mock_config())
        ctx = _make_context(loop_type="sda")
        action = rep.select_action(ctx)
        self.assertIn(action["eventsat_0"]["mode"], VALID_MODES)

    def test_ooda_context_with_enrichments(self):
        rep = LLMEventSat(_mock_config())
        ctx = _make_context(
            loop_type="ooda",
            enrichments={
                "situation_class": "nominal_sunlight",
                "urgency": 0.3,
                "battery_trending_down": False,
            },
        )
        action = rep.select_action(ctx)
        self.assertIn(action["eventsat_0"]["mode"], VALID_MODES)

    def test_react_context_with_reasoning(self):
        rep = LLMEventSat(_mock_config())
        ctx = _make_context(
            loop_type="react",
            enrichments={
                "reasoning_steps": [
                    {"check": "battery", "value": 0.7, "implication": "ok"},
                ],
            },
        )
        action = rep.select_action(ctx)
        self.assertIn(action["eventsat_0"]["mode"], VALID_MODES)


# ======================================================================
# Config Validation Tests
# ======================================================================

class TestConfigValidation(unittest.TestCase):
    """Test experiment config validation for hybrid representation."""

    def test_hybrid_representation_accepted(self):
        from src.orchestration.config_loader import ExperimentConfig
        config = ExperimentConfig(
            representation="hybrid",
            representation_config={"type": "llm_eventsat", "llm_mock": True},
        )
        self.assertEqual(config.representation, "hybrid")

    def test_hybrid_with_all_loops(self):
        from src.orchestration.config_loader import ExperimentConfig
        for loop in ["sda", "ooda", "react"]:
            config = ExperimentConfig(
                representation="hybrid",
                decision_loop=loop,
                representation_config={"type": "llm_eventsat", "llm_mock": True},
            )
            self.assertEqual(config.decision_loop, loop)

    def test_hybrid_with_all_ops_paradigms(self):
        from src.orchestration.config_loader import ExperimentConfig
        for ops in ["autonomous_hybrid", "autonomous_ground", "conventional_ground"]:
            config = ExperimentConfig(
                representation="hybrid",
                operations_paradigm=ops,
                representation_config={"type": "llm_eventsat", "llm_mock": True},
            )
            self.assertEqual(config.operations_paradigm, ops)


if __name__ == "__main__":
    unittest.main()

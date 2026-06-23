"""Tests for the real agentic LLM ground schedulers (hllm-a / llm-a), mocked — no live LLM.

These exercise plumbing and invariants only (per project test policy: no outcome
pinning). The agentic Plan-Tool-Reflect-Decide loop is validated for its decision
plumbing, schedule contract, symbolic-grounding toggle, and substrate integrity.
"""
from __future__ import annotations

import warnings

import pytest

from src.decision_procedure.context import DecisionContext
from src.behaviour.controller import BehaviourController, _REPRESENTATION_REGISTRY
from src.orchestration.config_loader import load_config
from src.representation.agentic_scheduler_eventsat import (
    AgenticSchedulerEventSat,
    LLMAgenticSchedulerEventSat,
)

_SCHEDULABLE = {"charging", "payload_observe", "payload_compress", "payload_detect",
                "payload_send", "safe"}


def _fresh_pass_state():
    return {
        "battery_soc": 0.7, "current_mode": "communication", "in_sunlight": True,
        "ground_pass_active": True, "obc_data_mb": 5.0, "jetson_raw_mb": 9.41,
        "jetson_compressed_mb": 0.0, "uncompressed_observations": 1,
        "undetected_observations": 0, "staleness_steps": 1, "estimated_gap_steps": 40,
        "daily_downlink_budget_mb": 27.0,
    }


# ----------------------------------------------------------------------
# Identity / registration — these cells are REAL, not placeholders
# ----------------------------------------------------------------------

def test_both_cells_are_real_not_placeholder() -> None:
    assert AgenticSchedulerEventSat.is_placeholder is False
    assert LLMAgenticSchedulerEventSat.is_placeholder is False


def test_registered_to_real_classes() -> None:
    import src.representation.agentic_scheduler_eventsat  # noqa: F401  (trigger @register)
    assert _REPRESENTATION_REGISTRY["agentic_scheduler_eventsat"] is AgenticSchedulerEventSat
    assert _REPRESENTATION_REGISTRY["llm_agentic_scheduler_eventsat"] is LLMAgenticSchedulerEventSat
    assert _REPRESENTATION_REGISTRY["agentic_scheduler_eventsat"].is_placeholder is False
    assert _REPRESENTATION_REGISTRY["llm_agentic_scheduler_eventsat"].is_placeholder is False


def test_controller_builds_real_agentic_schedulers() -> None:
    import src.representation.agentic_scheduler_eventsat  # noqa: F401
    controller = BehaviourController(config={})
    for name, cls in (
        ("agentic_scheduler_eventsat", AgenticSchedulerEventSat),
        ("llm_agentic_scheduler_eventsat", LLMAgenticSchedulerEventSat),
    ):
        rep = controller.get_representation(repr_type=name, repr_config={"llm_mock": True})
        assert isinstance(rep, cls)


def test_ah_hllm_a_and_ag_llm_a_resolve_to_real_schedulers() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ah = load_config("configs/experiments/eventsat_sas_ah_symb_hllm-a.yaml")
        ag = load_config("configs/experiments/eventsat_sas_ag_llm-a.yaml")
    assert ah.resolved_ground_planner_type == "agentic_scheduler_eventsat"
    assert ag.resolved_ground_planner_type == "llm_agentic_scheduler_eventsat"


# ----------------------------------------------------------------------
# Schedule generation via the agentic loop (mock)
# ----------------------------------------------------------------------

def test_hllm_a_generates_valid_schedule_clamped_to_gap() -> None:
    rep = AgenticSchedulerEventSat({"llm_mock": True})
    action = rep.select_action(DecisionContext(state=_fresh_pass_state(), loop_type="sda"))
    sat = action["eventsat_0"]
    assert sat["mode"] == "communication"
    sched = sat["schedule"]
    assert sched and all(s > 0 for _, s in sched)
    assert all(m in _SCHEDULABLE for m, _ in sched)
    # Hybrid grounding pads/clamps to the 40-step gap.
    assert sum(s for _, s in sched) == 40


def test_llm_a_is_ungrounded_no_pad() -> None:
    rep = LLMAgenticSchedulerEventSat({"llm_mock": True})
    assert rep._symbolic_grounding is False
    action = rep.select_action(DecisionContext(state=_fresh_pass_state(), loop_type="sda"))
    sched = action["eventsat_0"]["schedule"]
    # mock schedule = observe3 + compress4 + charging10 = 17 steps; ungrounded → NOT
    # padded to the 40-step gap (hllm-a would pad to 40).
    assert sum(s for _, s in sched) == 17


def test_between_passes_charges() -> None:
    rep = AgenticSchedulerEventSat({"llm_mock": True})
    state = _fresh_pass_state(); state["ground_pass_active"] = False
    action = rep.select_action(DecisionContext(state=state, loop_type="sda"))
    assert action["eventsat_0"]["mode"] == "charging"
    assert "schedule" not in action["eventsat_0"]


def test_stale_telemetry_communicates_first() -> None:
    rep = AgenticSchedulerEventSat({"llm_mock": True})
    state = _fresh_pass_state(); state["staleness_steps"] = 99
    action = rep.select_action(DecisionContext(state=state, loop_type="sda"))
    assert action["eventsat_0"]["mode"] == "communication"
    assert "schedule" not in action["eventsat_0"]


def test_agentic_metrics_exposed() -> None:
    rep = AgenticSchedulerEventSat({"llm_mock": True})
    rep.select_action(DecisionContext(state=_fresh_pass_state(), loop_type="sda"))
    m = rep.get_metrics()
    assert "agentic_total_decisions" in m
    assert "agentic_avg_steps_per_decision" in m
    assert "llm_schedule_entries" in m  # inherited from the single-shot scheduler
    assert m["agentic_total_decisions"] >= 1.0


# ----------------------------------------------------------------------
# Symbolic-grounding toggle (the hllm-a ↔ llm-a ablation), reusing _validate_schedule
# ----------------------------------------------------------------------

def test_hllm_a_safety_shield_vetoes_critical_state() -> None:
    rep = AgenticSchedulerEventSat({"llm_mock": True, "settling_time_steps": 2})
    low_batt = _fresh_pass_state(); low_batt.update({"battery_soc": 0.1, "obc_data_mb": 0.0})
    sched = rep._validate_schedule([["payload_observe", 12]], gap_steps=88, state=low_batt)
    assert sched is not None and sched[0][0] == "charging"


def test_llm_a_does_not_apply_safety_shield() -> None:
    rep = LLMAgenticSchedulerEventSat({"llm_mock": True, "settling_time_steps": 2})
    low_batt = _fresh_pass_state(); low_batt.update({"battery_soc": 0.1, "obc_data_mb": 0.0})
    sched = rep._validate_schedule([["payload_observe", 12]], gap_steps=88, state=low_batt)
    # Ungrounded: the LLM's block passes through (no veto); env enforces at execution.
    assert sched is not None and sched[0] == ("payload_observe", 12)


# ----------------------------------------------------------------------
# Decision extraction (protocol form vs flattened form)
# ----------------------------------------------------------------------

def test_extract_schedule_accepts_nested_and_flat() -> None:
    nested = {"decision": {"schedule": [["charging", 5]], "rationale": "r"}}
    flat = {"schedule": [["charging", 5]], "rationale": "r"}
    plan = {"plan": "thinking", "tool_call": {"name": "check_battery", "args": {}}}
    assert AgenticSchedulerEventSat._extract_schedule(nested)[0] == [["charging", 5]]
    assert AgenticSchedulerEventSat._extract_schedule(flat)[0] == [["charging", 5]]
    assert AgenticSchedulerEventSat._extract_schedule(plan)[0] is None


# ----------------------------------------------------------------------
# Substrate integrity (no symbolic substitution on failure)
# ----------------------------------------------------------------------

def test_substrate_integrity_raises_on_no_valid_schedule(monkeypatch) -> None:
    rep = AgenticSchedulerEventSat({"llm_mock": True})
    rep.MAX_RETRIES = 0
    # The loop always returns an all-invalid schedule → validation yields nothing.
    monkeypatch.setattr(
        rep._client, "generate",
        lambda **kw: '{"decision": {"schedule": [["not_a_mode", 5]], "rationale": "x"}}',
    )
    with pytest.raises(RuntimeError, match="integrity"):
        rep.select_action(DecisionContext(state=_fresh_pass_state(), loop_type="sda"))

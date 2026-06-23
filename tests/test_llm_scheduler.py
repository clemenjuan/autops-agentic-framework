"""Tests for the real single-shot LLM ground scheduler (hllm-s), mocked — no live LLM."""
from __future__ import annotations

import warnings

import pytest

from src.core.behaviour.controller import BehaviourController
from src.core.decision_procedure.context import DecisionContext
from src.core.config_loader import ExperimentConfig, load_config
from src.eventsat.llm_scheduler import LLMSchedulerEventSat
from src.eventsat.llm_prompts import format_schedule_prompt


def _fresh_pass_state():
    return {
        "battery_soc": 0.7, "current_mode": "communication", "in_sunlight": True,
        "ground_pass_active": True, "obc_data_mb": 5.0, "jetson_raw_mb": 9.41,
        "jetson_compressed_mb": 0.0, "uncompressed_observations": 1,
        "undetected_observations": 0, "staleness_steps": 1, "estimated_gap_steps": 40,
        "daily_downlink_budget_mb": 27.0,
    }


def test_schedule_prompt_reports_downlink_without_observation_hard_cap() -> None:
    state = _fresh_pass_state()
    state.update({
        "achievable_downlink_mb": 1.89,
        "obc_data_mb": 271.52,
        "jetson_compressed_mb": 0.0,
        "uncompressed_observations": 0,
    })
    prompt = format_schedule_prompt(state, gap_steps=88)
    assert "OBC ready for downlink: 271.52 / 4096 MB" in prompt
    assert "Downlink achievable at next pass: 1.89 MB" in prompt
    assert "observing more than this just fills storage you cannot deliver" in prompt
    assert "Already queued for future downlink" not in prompt
    assert "Remaining capacity for NEW SCIENCE observations" not in prompt
    assert "payload_observe command budget" not in prompt
    assert "HARD RULE" not in prompt

def test_schedule_prompt_explicitly_reports_obc_capacity_for_full_storage_context() -> None:
    state = _fresh_pass_state()
    state.update({
        "achievable_downlink_mb": 2.35,
        "obc_data_mb": 519.99,
    })
    prompt = format_schedule_prompt(state, gap_steps=757)

    assert "OBC ready for downlink: 519.99 / 4096 MB" in prompt



def test_hllm_safety_shield_vetoes_critical_states_not_observation_volume() -> None:
    """Hybrid grounding is SAFETY, not behaviour: it must NOT cap how much the LLM
    chooses to observe, but it MUST veto operational blocks in a critical battery or
    storage state (→ charging, the env's safe fallback)."""
    rep = LLMSchedulerEventSat({"settling_time_steps": 2})

    # Safe state: a 12-step observe block survives intact (no downlink-budget clamp).
    safe = _fresh_pass_state()
    safe.update({"battery_soc": 0.7, "obc_data_mb": 0.0, "achievable_downlink_mb": 2.4})
    sched = rep._validate_schedule(
        [["payload_observe", 12], ["payload_compress", 20]], gap_steps=88, state=safe)
    assert sched is not None and sched[0] == ("payload_observe", 12)

    # Battery-critical: operational block below the operations SoC floor → charging.
    low_batt = _fresh_pass_state()
    low_batt.update({"battery_soc": 0.1, "obc_data_mb": 0.0})
    sched = rep._validate_schedule([["payload_observe", 12]], gap_steps=88, state=low_batt)
    assert sched is not None and sched[0][0] == "charging"

    # Memory-critical: OBC storage critically full → observe vetoed → charging.
    full_obc = _fresh_pass_state()
    full_obc.update({"battery_soc": 0.7, "obc_data_mb": 1.0e9})
    sched = rep._validate_schedule([["payload_observe", 12]], gap_steps=88, state=full_obc)
    assert sched is not None and sched[0][0] == "charging"


def test_llm_single_scheduler_does_not_symbolically_cap_observe_duration() -> None:
    from src.eventsat.llm_scheduler import LLMSingleSchedulerEventSat
    rep = LLMSingleSchedulerEventSat({"settling_time_steps": 2})
    state = _fresh_pass_state()
    state.update({
        "achievable_downlink_mb": 2.4,
        "obc_data_mb": 0.0,
        "jetson_compressed_mb": 0.0,
        "uncompressed_observations": 0,
    })
    schedule = rep._validate_schedule(
        [["payload_observe", 12], ["payload_compress", 20]],
        gap_steps=88,
        state=state,
    )
    assert schedule is not None
    assert schedule[0] == ("payload_observe", 12)


def test_ag_hllm_s_resolves_to_real_scheduler() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = load_config("configs/experiments/eventsat_sas_ag_hllm-s.yaml")
    assert cfg.resolved_ground_planner_type == "llm_scheduler_eventsat"
    assert LLMSchedulerEventSat.is_placeholder is False


def test_registered_real_not_placeholder() -> None:
    import src.eventsat.llm_scheduler  # noqa: F401
    from src.core.behaviour.controller import _REPRESENTATION_REGISTRY
    cls = _REPRESENTATION_REGISTRY["llm_scheduler_eventsat"]
    assert cls is LLMSchedulerEventSat
    assert cls.is_placeholder is False


def test_generates_valid_schedule_clamped_to_gap() -> None:
    rep = LLMSchedulerEventSat({"llm_mock": True})  # mock → fixed schedule, padded/clamped
    action = rep.select_action(DecisionContext(state=_fresh_pass_state(), loop_type="sda"))
    sat = action["eventsat_0"]
    assert sat["mode"] == "communication"
    sched = sat["schedule"]
    assert sched and all(s > 0 for _, s in sched)
    assert all(m in {"charging", "payload_observe", "payload_compress", "payload_detect",
                      "payload_send", "safe"} for m, _ in sched)
    assert sum(s for _, s in sched) == 40  # clamped/padded to estimated_gap_steps


def test_between_passes_charges() -> None:
    rep = LLMSchedulerEventSat({"llm_mock": True})
    state = _fresh_pass_state(); state["ground_pass_active"] = False
    action = rep.select_action(DecisionContext(state=state, loop_type="sda"))
    assert action["eventsat_0"]["mode"] == "charging"
    assert "schedule" not in action["eventsat_0"]


def test_stale_telemetry_communicates_first() -> None:
    rep = LLMSchedulerEventSat({"llm_mock": True})
    state = _fresh_pass_state(); state["staleness_steps"] = 99
    action = rep.select_action(DecisionContext(state=state, loop_type="sda"))
    assert action["eventsat_0"]["mode"] == "communication"
    assert "schedule" not in action["eventsat_0"]


def test_client_mean_latency_and_reset() -> None:
    """M-07 plumbing: mean per-live-call latency + per-episode counter reset."""
    from src.core.llm_client import LLMClient
    c = LLMClient({"llm_mock": True})
    c._total_calls = 5
    c._cache_hits = 2
    c._total_latency_s = 9.0
    m = c.get_metrics()
    assert m["llm_mean_call_latency_s"] == 3.0  # 9.0 / (5 - 2) live calls
    c.reset_metrics()
    assert c.get_metrics()["llm_api_calls"] == 0.0
    assert c.get_metrics()["llm_total_latency_s"] == 0.0


def test_llm_single_scheduler_real_and_ungrounded() -> None:
    """llm-s ground core is real (not placeholder) and applies NO symbolic grounding:
    the LLM schedule passes through without clamp/pad (unlike hllm-s)."""
    from src.eventsat.llm_scheduler import LLMSingleSchedulerEventSat
    rep = LLMSingleSchedulerEventSat({"llm_mock": True})
    assert rep.is_placeholder is False
    assert rep._symbolic_grounding is False
    action = rep.select_action(DecisionContext(state=_fresh_pass_state(), loop_type="sda"))
    sched = action["eventsat_0"]["schedule"]
    # mock = observe3 + compress4 + charging10 = 17 steps; ungrounded → NOT padded to the
    # 40-step gap (hllm-s would pad to 40).
    assert sum(s for _, s in sched) == 17


def test_m07_per_decision_cycle_and_ground_latency() -> None:
    """M-07 averages over decision cycles (steps where inference ran), not all steps;
    AH ground-planner latency is captured separately."""
    from src.core.metrics_collector import StepMetrics
    from src.eventsat.metrics import EventSatMetricsCollector
    col = EventSatMetricsCollector(config={})
    steps = []
    # 3 steps: two ran inference (1.0 s, 3.0 s; one with a 2.0 s ground call), one skipped.
    for i, (lat, allowed, gp) in enumerate([(1.0, 1, 0.0), (3.0, 1, 2.0), (0.0, 0, 0.0)]):
        steps.append(StepMetrics(
            timestep=i, wall_clock_seconds=lat, reward=0.0,
            metrics={"decision_latency_s": lat, "inference_allowed": float(allowed),
                     "ground_decision_latency_s": gp},
        ))
    agg = col.aggregate_episode_metrics(steps).aggregated
    assert agg["mean_latency_s"] == 2.0          # (1+3)/2 decided steps — NOT /3
    assert agg["max_latency_s"] == 3.0
    assert agg["mean_ground_latency_s"] == 2.0   # one ground-planning event


def test_substrate_integrity_raises_on_no_valid_schedule(monkeypatch) -> None:
    """If the LLM never returns a valid schedule, the episode fails (no fallback)."""
    rep = LLMSchedulerEventSat({"llm_mock": True})
    rep.MAX_RETRIES = 0
    monkeypatch.setattr(rep._client, "generate", lambda **kw: '{"schedule": [["not_a_mode", 5]]}')
    with pytest.raises(RuntimeError, match="integrity"):
        rep.select_action(DecisionContext(state=_fresh_pass_state(), loop_type="sda"))

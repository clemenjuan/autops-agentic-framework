"""Tests for the real single-shot LLM ground scheduler (hllm-s), mocked — no live LLM."""
from __future__ import annotations

import warnings

import pytest

from src.behaviour.controller import BehaviourController
from src.decision_procedure.context import DecisionContext
from src.orchestration.config_loader import ExperimentConfig, load_config
from src.representation.llm_scheduler_eventsat import LLMSchedulerEventSat


def _fresh_pass_state():
    return {
        "battery_soc": 0.7, "current_mode": "communication", "in_sunlight": True,
        "ground_pass_active": True, "obc_data_mb": 5.0, "jetson_raw_mb": 9.41,
        "jetson_compressed_mb": 0.0, "uncompressed_observations": 1,
        "undetected_observations": 0, "staleness_steps": 1, "estimated_gap_steps": 40,
        "daily_downlink_budget_mb": 27.0,
    }


def test_ag_hllm_s_resolves_to_real_scheduler() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = load_config("configs/experiments/eventsat_sas_ag_hllm-s.yaml")
    assert cfg.resolved_ground_planner_type == "llm_scheduler_eventsat"
    assert LLMSchedulerEventSat.is_placeholder is False


def test_registered_real_not_placeholder() -> None:
    import src.representation.llm_scheduler_eventsat  # noqa: F401
    from src.behaviour.controller import _REPRESENTATION_REGISTRY
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
    from src.representation.llm_client import LLMClient
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
    from src.representation.llm_scheduler_eventsat import LLMSingleSchedulerEventSat
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
    from src.orchestration.metrics_collector import StepMetrics
    from src.orchestration.eventsat_metrics import EventSatMetricsCollector
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

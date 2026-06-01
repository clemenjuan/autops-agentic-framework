"""
Tests for OODA Decision Loop.

Validates Boyd's OODA loop implementation grounded in:
  - Miller et al. (2021) — OODA structure, feedback loops, gamma-distributed TPM
  - Hartmann et al. (2024) METIS — situation classification, CBR in Orient
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from src.decision_procedure.ooda_loop import (
    SITUATION_ANOMALY,
    SITUATION_BATTERY_CRITICAL,
    SITUATION_DATA_PIPELINE,
    SITUATION_ECLIPSE_CHARGING,
    SITUATION_NOMINAL,
    SITUATION_PASS_OPPORTUNITY,
    SITUATION_STORAGE_CRITICAL,
    OODALoop,
)
from src.decision_procedure.context import DecisionContext
from src.memory.fixed_memory import FixedMemory
from src.representation.base import Representation


# -- Test helpers ----------------------------------------------------------


class StubRepresentation(Representation):
    """Minimal representation for unit-testing the OODA loop."""

    def __init__(self) -> None:
        super().__init__()
        self._last_context: Optional[DecisionContext] = None
        self._rationale: Optional[str] = "stub rationale"

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        if isinstance(observation, dict):
            return dict(observation)
        return {"battery_soc": 0.5, "health_status": "nominal"}

    def select_action(self, context: DecisionContext) -> Dict[str, Any]:
        self._last_context = context
        return {"eventsat_0": {"mode": "charging"}}

    def get_rationale(self) -> Optional[str]:
        return self._rationale


def _make_loop(config: Dict[str, Any] | None = None) -> OODALoop:
    return OODALoop(config=config or {}, representation=StubRepresentation())


def _make_state(**overrides: Any) -> Dict[str, Any]:
    """Create a default encoded state dict with overrides."""
    base = {
        "battery_soc": 0.7,
        "current_mode": "charging",
        "in_sunlight": True,
        "ground_pass_active": False,
        "data_stored_mb": 10.0,
        "obc_data_mb": 5.0,
        "jetson_raw_mb": 0.0,
        "jetson_compressed_mb": 0.0,
        "storage_capacity_mb": 512.0,
        "uncompressed_observations": 0,
        "undetected_observations": 0,
        "health_status": "nominal",
        "daily_downlink_budget_mb": 27.0,
    }
    base.update(overrides)
    return base


# =========================================================================
# OBSERVE PHASE: Situation Classification (cf. METIS Monitoring Agent)
# =========================================================================


class TestSituationClassification:
    def test_anomaly_active(self) -> None:
        loop = _make_loop()
        state = _make_state(health_status="degraded")
        assert loop._classify_situation(state) == SITUATION_ANOMALY

    def test_battery_critical(self) -> None:
        loop = _make_loop()
        state = _make_state(battery_soc=0.15)
        assert loop._classify_situation(state) == SITUATION_BATTERY_CRITICAL

    def test_pass_opportunity(self) -> None:
        loop = _make_loop()
        state = _make_state(ground_pass_active=True, obc_data_mb=10.0)
        assert loop._classify_situation(state) == SITUATION_PASS_OPPORTUNITY

    def test_storage_critical(self) -> None:
        loop = _make_loop()
        state = _make_state(jetson_raw_mb=450.0, storage_capacity_mb=512.0)
        assert loop._classify_situation(state) == SITUATION_STORAGE_CRITICAL

    def test_eclipse_charging(self) -> None:
        loop = _make_loop()
        state = _make_state(in_sunlight=False, battery_soc=0.40)
        assert loop._classify_situation(state) == SITUATION_ECLIPSE_CHARGING

    def test_data_pipeline_active(self) -> None:
        loop = _make_loop()
        state = _make_state(uncompressed_observations=3)
        assert loop._classify_situation(state) == SITUATION_DATA_PIPELINE

    def test_nominal(self) -> None:
        loop = _make_loop()
        state = _make_state()
        assert loop._classify_situation(state) == SITUATION_NOMINAL

    def test_empty_state(self) -> None:
        loop = _make_loop()
        assert loop._classify_situation({}) == SITUATION_NOMINAL

    def test_priority_order_anomaly_beats_battery(self) -> None:
        """Anomaly is highest priority even with critical battery."""
        loop = _make_loop()
        state = _make_state(health_status="degraded", battery_soc=0.10)
        assert loop._classify_situation(state) == SITUATION_ANOMALY

    def test_priority_order_battery_beats_pass(self) -> None:
        """Battery critical beats pass opportunity."""
        loop = _make_loop()
        state = _make_state(
            battery_soc=0.15, ground_pass_active=True, obc_data_mb=10.0
        )
        assert loop._classify_situation(state) == SITUATION_BATTERY_CRITICAL


# =========================================================================
# ORIENT PHASE: CBR + Trend Analysis + Urgency
# =========================================================================


class TestOrientPhase:
    def test_empty_history(self) -> None:
        """Orient handles empty memory history gracefully (step 0)."""
        loop = _make_loop()
        memory = FixedMemory()
        state = _make_state()
        oriented, iters = loop._orient(state, SITUATION_NOMINAL, memory)
        assert "orient_situation_class" in oriented
        assert oriented["orient_situation_class"] == SITUATION_NOMINAL
        assert iters == 1

    def test_battery_trend_detection(self) -> None:
        """Orient detects declining battery trend from history."""
        loop = _make_loop({"orient_history_window": 5})
        memory = FixedMemory()
        # Populate history with declining SoC
        for soc in [0.8, 0.7, 0.6, 0.5, 0.4]:
            memory.update("constellation_state", {"battery_soc": soc})
        state = _make_state(battery_soc=0.35)
        oriented, _ = loop._orient(state, SITUATION_NOMINAL, memory)
        assert oriented.get("orient_battery_trending_down") is True
        assert oriented["orient_battery_trend"] < 0

    def test_battery_trend_stable(self) -> None:
        """No trend flag when battery is stable."""
        loop = _make_loop()
        memory = FixedMemory()
        for soc in [0.7, 0.7, 0.7]:
            memory.update("constellation_state", {"battery_soc": soc})
        state = _make_state(battery_soc=0.7)
        oriented, _ = loop._orient(state, SITUATION_NOMINAL, memory)
        assert oriented.get("orient_battery_trending_down") is False

    def test_cbr_retrieve_matching_class(self) -> None:
        """CBR retrieves a case with matching situation class."""
        loop = _make_loop()
        memory = FixedMemory()
        # Store some history entries with orient metadata
        for i in range(5):
            memory.update("constellation_state", {
                "battery_soc": 0.3 + i * 0.1,
                "orient_situation_class": SITUATION_NOMINAL,
                "last_action": {"eventsat_0": {"mode": "charging"}},
            })
        state = _make_state(battery_soc=0.65)
        oriented, _ = loop._orient(state, SITUATION_NOMINAL, memory)
        case = oriented.get("orient_similar_case_outcome")
        assert case is not None
        assert case["matched_class"] is True

    def test_cbr_retrieve_fallback_no_match(self) -> None:
        """CBR falls back to SoC proximity when no class matches."""
        loop = _make_loop()
        memory = FixedMemory()
        for i in range(3):
            memory.update("constellation_state", {
                "battery_soc": 0.5 + i * 0.1,
                "orient_situation_class": SITUATION_DATA_PIPELINE,
                "last_action": {"eventsat_0": {"mode": "payload_compress"}},
            })
        state = _make_state(battery_soc=0.55)
        oriented, _ = loop._orient(state, SITUATION_ANOMALY, memory)
        case = oriented.get("orient_similar_case_outcome")
        assert case is not None
        assert case["matched_class"] is False

    def test_cbr_no_memory(self) -> None:
        """CBR returns None when memory is None."""
        loop = _make_loop()
        state = _make_state()
        oriented, _ = loop._orient(state, SITUATION_NOMINAL, None)
        assert oriented.get("orient_similar_case_outcome") is None

    def test_urgency_anomaly(self) -> None:
        """Anomaly situation produces high urgency."""
        loop = _make_loop()
        state = _make_state(health_status="degraded", battery_soc=0.3)
        urgency = loop._compute_urgency(
            state, SITUATION_ANOMALY, {"orient_battery_trending_down": True}
        )
        assert urgency >= 0.9

    def test_urgency_nominal(self) -> None:
        """Nominal situation has zero base urgency."""
        loop = _make_loop()
        state = _make_state()
        urgency = loop._compute_urgency(state, SITUATION_NOMINAL, {})
        assert urgency == 0.0

    def test_urgency_battery_critical_boosted(self) -> None:
        """Battery critical + low SoC boosts urgency."""
        loop = _make_loop()
        state = _make_state(battery_soc=0.15)
        urgency = loop._compute_urgency(
            state, SITUATION_BATTERY_CRITICAL, {}
        )
        assert urgency > 0.8

    def test_competing_priorities_charge_vs_downlink(self) -> None:
        """Detect charge vs downlink competition."""
        loop = _make_loop()
        state = _make_state(
            battery_soc=0.40, ground_pass_active=True, obc_data_mb=10.0
        )
        competing = loop._detect_competing_priorities(state)
        assert "charge_vs_downlink" in competing

    def test_competing_priorities_none(self) -> None:
        """No competing priorities in nominal state."""
        loop = _make_loop()
        state = _make_state()
        competing = loop._detect_competing_priorities(state)
        assert len(competing) == 0

    def test_orient_multiple_iterations(self) -> None:
        """Orient re-iterates when competing priorities are detected."""
        loop = _make_loop({"max_orient_iterations": 3})
        memory = FixedMemory()
        # State with multiple competing priorities
        state = _make_state(
            battery_soc=0.40,
            ground_pass_active=True,
            obc_data_mb=10.0,
            uncompressed_observations=3,
        )
        oriented, iters = loop._orient(state, SITUATION_PASS_OPPORTUNITY, memory)
        # Should iterate more than once due to competing priorities
        assert iters >= 2

    def test_orient_single_iteration_unambiguous(self) -> None:
        """Orient only iterates once when unambiguous."""
        loop = _make_loop({"max_orient_iterations": 3})
        memory = FixedMemory()
        state = _make_state()  # nominal, no competing priorities
        oriented, iters = loop._orient(state, SITUATION_NOMINAL, memory)
        assert iters == 1

    def test_sunlight_transition_detection(self) -> None:
        """Orient detects sunlight transitions (sun→eclipse)."""
        loop = _make_loop()
        memory = FixedMemory()
        # Two updates: first goes to history, second is current constellation_state.
        # History will contain {in_sunlight: True}, so when current state is False
        # the trend analysis detects the transition.
        memory.update("constellation_state", {"in_sunlight": True})
        memory.update("constellation_state", {"in_sunlight": True})
        state = _make_state(in_sunlight=False)
        oriented, _ = loop._orient(state, SITUATION_ECLIPSE_CHARGING, memory)
        assert oriented.get("orient_sunlight_transition") is True
        assert oriented.get("orient_entered_eclipse") is True


# =========================================================================
# ACT PHASE: Memory Update + CBR Retain
# =========================================================================


class TestActPhase:
    def test_memory_stores_constellation_state(self) -> None:
        """Act phase stores encoded state + orient metadata in memory."""
        loop = _make_loop()
        memory = FixedMemory()
        state = _make_state()
        oriented = dict(state, orient_situation_class=SITUATION_NOMINAL)
        action = {"eventsat_0": {"mode": "charging"}}
        loop._act_and_update_memory(
            memory, state, oriented, action, SITUATION_NOMINAL
        )
        stored = memory.query("constellation_state")
        assert stored is not None
        assert stored.get("orient_situation_class") == SITUATION_NOMINAL
        assert stored.get("last_action") == action

    def test_memory_stores_orient_assessment(self) -> None:
        """Custom slot gets orient assessment for introspection."""
        loop = _make_loop()
        memory = FixedMemory()
        state = _make_state()
        oriented = dict(
            state,
            orient_situation_class=SITUATION_NOMINAL,
            orient_urgency=0.3,
        )
        action = {"eventsat_0": {"mode": "charging"}}
        loop._act_and_update_memory(
            memory, state, oriented, action, SITUATION_NOMINAL
        )
        custom = memory.query("custom")
        assert "last_orient_assessment" in custom
        assert custom["last_orient_assessment"]["orient_urgency"] == 0.3

    def test_history_populated_across_steps(self) -> None:
        """Memory history grows as constellation_state is updated each step."""
        loop = _make_loop()
        memory = FixedMemory()
        for i in range(5):
            state = _make_state(battery_soc=0.5 + i * 0.05)
            oriented = dict(state, orient_situation_class=SITUATION_NOMINAL)
            action = {"eventsat_0": {"mode": "charging"}}
            loop._act_and_update_memory(
                memory, state, oriented, action, SITUATION_NOMINAL
            )
        history = memory.query("history")
        # 5 updates: first has no previous, so 4 history entries
        assert len(history) == 4

    def test_no_memory_does_not_raise(self) -> None:
        """Act phase handles None memory gracefully."""
        loop = _make_loop()
        result = loop._act_and_update_memory(
            None, {}, {}, {}, SITUATION_NOMINAL
        )
        assert result is None


# =========================================================================
# FEEDBACK: Attention Guidance (Boyd's Orient→Observe)
# =========================================================================


class TestFeedback:
    def test_attention_guidance_set(self) -> None:
        """Attention guidance is stored for next Observe cycle."""
        loop = _make_loop()
        memory = FixedMemory()
        state = _make_state(health_status="degraded")
        oriented = dict(
            state,
            orient_situation_class=SITUATION_ANOMALY,
            orient_attention_guidance={
                "priority_monitor": ["health_status", "battery_soc"],
                "urgency_level": 0.9,
            },
        )
        action = {"eventsat_0": {"mode": "safe"}}
        loop._act_and_update_memory(
            memory, state, oriented, action, SITUATION_ANOMALY
        )
        assert loop._attention_guidance is not None
        assert "health_status" in loop._attention_guidance["priority_monitor"]

    def test_attention_guidance_for_battery_critical(self) -> None:
        """Battery critical should monitor battery_soc and in_sunlight."""
        loop = _make_loop()
        guidance = loop._generate_attention_guidance(
            SITUATION_BATTERY_CRITICAL, {}, 0.8
        )
        assert "battery_soc" in guidance["priority_monitor"]
        assert "in_sunlight" in guidance["priority_monitor"]


# =========================================================================
# METRICS (Miller et al. TPM framework)
# =========================================================================


class TestMetrics:
    def test_all_metric_keys_present(self) -> None:
        """All expected OODA metrics are returned."""
        loop = _make_loop()
        metrics = loop.get_metrics()
        expected_keys = {
            "decision_latency_s",
            "observe_latency_s",
            "orient_latency_s",
            "decide_latency_s",
            "orient_iterations",
            "orient_urgency",
            "orient_cases_retrieved",
            "total_decisions",
            "has_rationale",
            "rationale",
        }
        assert set(metrics.keys()) == expected_keys

    def test_latency_breakdown_consistent(self) -> None:
        """Total latency >= sum of phase latencies."""
        loop = _make_loop()
        memory = FixedMemory()
        obs = _make_state()
        loop.process(obs, memory)
        m = loop.get_metrics()
        phase_sum = m["observe_latency_s"] + m["orient_latency_s"] + m["decide_latency_s"]
        assert m["decision_latency_s"] >= phase_sum * 0.99  # allow tiny float error

    def test_total_decisions_increments(self) -> None:
        """total_decisions increments each step."""
        loop = _make_loop()
        memory = FixedMemory()
        obs = _make_state()
        loop.process(obs, memory)
        loop.process(obs, memory)
        assert loop.get_metrics()["total_decisions"] == 2.0

    def test_reset_clears_state(self) -> None:
        loop = _make_loop()
        memory = FixedMemory()
        loop.process(_make_state(), memory)
        loop.reset()
        m = loop.get_metrics()
        assert m["total_decisions"] == 0.0
        assert m["decision_latency_s"] == 0.0

    def test_get_name(self) -> None:
        loop = _make_loop()
        assert loop.get_name() == "OODALoop"


# =========================================================================
# FULL PROCESS (end-to-end unit test)
# =========================================================================


class TestProcess:
    def test_process_returns_action_and_memory(self) -> None:
        loop = _make_loop()
        memory = FixedMemory()
        action, updated_memory = loop.process(_make_state(), memory)
        assert "eventsat_0" in action
        assert updated_memory is memory  # same object, mutated

    def test_orient_enriches_context_passed_to_representation(self) -> None:
        """The representation receives OODA enrichments via DecisionContext."""
        repr_ = StubRepresentation()
        loop = OODALoop(config={}, representation=repr_)
        memory = FixedMemory()
        loop.process(_make_state(), memory)
        ctx = repr_._last_context
        assert ctx is not None
        assert ctx.loop_type == "ooda"
        assert "situation_class" in ctx.enrichments
        assert "urgency" in ctx.enrichments
        # State should be the raw encoded state (no orient_* keys merged)
        orient_keys = [k for k in ctx.state if k.startswith("orient_")]
        assert len(orient_keys) == 0

    def test_has_rationale_tracked(self) -> None:
        loop = _make_loop()
        memory = FixedMemory()
        loop.process(_make_state(), memory)
        assert loop.get_metrics()["has_rationale"] == 1.0


# =========================================================================
# INTEGRATION: ExperimentRunner with OODA
# =========================================================================


class TestExperimentRunnerOODAIntegration:
    def test_ooda_autonomous_hybrid(self, tmp_path) -> None:
        """OODA + rule_based + autonomous_hybrid runs end-to-end."""
        from src.orchestration.config_loader import ExperimentConfig
        from src.orchestration.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(
            experiment_id="ooda_test_ah",
            agent_organization="sas",
            decision_loop="ooda",
            representation="symbolic",
            emergence_mode="hand_designed",
            operations_paradigm="autonomous_hybrid",
            decision_loop_config={
                "orient_history_window": 10,
                "max_orient_iterations": 1,
            },
            representation_config={"type": "rule_based_eventsat"},
            environment={
                "constellation_size": 1,
                "timestep_seconds": 60,
                "max_steps": 50,
                "scenario": "eventsat",
                "scenario_config": {},
            },
            num_episodes=1,
            max_steps=50,
            save_checkpoints=False,
            log_level="WARNING",
            output_dir=str(tmp_path),
        )
        runner = ExperimentRunner(config=cfg)
        results = runner.run()
        assert results["num_episodes"] == 1
        assert len(results["episodes"][0]["steps"]) == 50

    def test_ooda_conventional_ground(self, tmp_path) -> None:
        """OODA + schedule_based + conventional_ground runs end-to-end."""
        from src.orchestration.config_loader import ExperimentConfig
        from src.orchestration.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(
            experiment_id="ooda_test_cg",
            agent_organization="sas",
            decision_loop="ooda",
            representation="symbolic",
            emergence_mode="hand_designed",
            operations_paradigm="conventional_ground",
            decision_loop_config={
                "orient_history_window": 10,
                "max_orient_iterations": 1,
            },
            operations_paradigm_config={"orbital_period_steps": 93},
            representation_config={"type": "schedule_based_eventsat"},
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

    def test_orient_metrics_in_results(self, tmp_path) -> None:
        """OODA-specific metrics appear in collected step and episode metrics."""
        from src.orchestration.config_loader import ExperimentConfig
        from src.orchestration.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(
            experiment_id="ooda_metrics_test",
            agent_organization="sas",
            decision_loop="ooda",
            representation="symbolic",
            emergence_mode="hand_designed",
            operations_paradigm="autonomous_hybrid",
            decision_loop_config={"orient_history_window": 5},
            representation_config={"type": "rule_based_eventsat"},
            environment={
                "constellation_size": 1,
                "timestep_seconds": 60,
                "max_steps": 20,
                "scenario": "eventsat",
                "scenario_config": {},
            },
            num_episodes=1,
            max_steps=20,
            save_checkpoints=False,
            log_level="WARNING",
            output_dir=str(tmp_path),
        )
        runner = ExperimentRunner(config=cfg)
        results = runner.run()

        # Check step-level metrics via the episode_metrics object
        ep_metrics = results["episodes"][0]["episode_metrics"]
        assert ep_metrics is not None
        assert len(ep_metrics.step_metrics) == 20
        first_step = ep_metrics.step_metrics[0]
        assert "orient_latency_s" in first_step.metrics
        assert "orient_iterations" in first_step.metrics
        assert "orient_urgency" in first_step.metrics

        # Check episode-level aggregation
        assert "mean_orient_latency_s" in ep_metrics.aggregated
        assert "mean_orient_iterations" in ep_metrics.aggregated

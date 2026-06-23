"""Tests for the Operations Paradigm dimension."""

import pytest

from src.core.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteState,
)
from src.core.operations.autonomous_ground import AutonomousGround
from src.core.operations.base import GroundKnowledge, OperationsParadigm
from src.core.operations.autonomous_hybrid import AutonomousHybrid
from src.core.operations.conventional_ground import ConventionalGround


# -----------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------


def _make_observation(
    step=0,
    battery_soc=0.8,
    data_stored_mb=10.0,
    mode="charging",
    in_sunlight=True,
    ground_pass_active=False,
):
    sat = SatelliteState(
        satellite_id="eventsat_0",
        position=[0.0, 0.0, 500.0],
        velocity=[0.0, 0.0, 0.0],
        resources={
            "battery_soc": battery_soc,
            "data_stored_mb": data_stored_mb,
            "data_downlinked_mb": 0.0,
        },
        status=mode,
        metadata={
            "in_sunlight": in_sunlight,
            "ground_pass_active": ground_pass_active,
            "uncompressed_observations": 0,
            "total_observation_s": 0.0,
            "storage_capacity_mb": 512.0,
            "health_status": "nominal",
        },
    )
    constellation = ConstellationState(
        timestep=step,
        epoch_seconds=step * 60.0,
        satellites={"eventsat_0": sat},
        global_info={"max_steps": 10080},
    )
    return EnvironmentObservation(
        constellation_state=constellation, tasks=[], events=[]
    )


# -----------------------------------------------------------------
# OperationsParadigm ABC
# -----------------------------------------------------------------


class TestOperationsParadigmABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            OperationsParadigm()

    def test_ground_knowledge_defaults(self):
        gk = GroundKnowledge()
        assert gk.last_update_step == 0
        assert gk.battery_soc == 0.8
        assert gk.staleness_steps == 0
        assert gk.health_status == "nominal"


# -----------------------------------------------------------------
# AutonomousHybrid
# -----------------------------------------------------------------


class TestAutonomousHybrid:
    def test_filter_observation_is_passthrough(self):
        paradigm = AutonomousHybrid()
        obs = _make_observation(step=5, battery_soc=0.7)
        filtered = paradigm.filter_observation(obs, step=5)
        assert filtered is obs

    def test_can_always_act(self):
        paradigm = AutonomousHybrid()
        assert paradigm.can_act(step=0, ground_pass_active=False) is True
        assert paradigm.can_act(step=100, ground_pass_active=True) is True

    def test_process_action_is_passthrough(self):
        paradigm = AutonomousHybrid()
        action = {"eventsat_0": {"mode": "payload_observe"}}
        result = paradigm.process_action(action, step=0, ground_pass_active=False)
        assert result is action

    def test_get_name(self):
        paradigm = AutonomousHybrid()
        assert paradigm.get_name() == "AutonomousHybrid"

    def test_reset(self):
        paradigm = AutonomousHybrid()
        paradigm._ground_knowledge.battery_soc = 0.5
        paradigm.reset()
        assert paradigm._ground_knowledge.battery_soc == 0.8


# -----------------------------------------------------------------
# AutonomousGround
# -----------------------------------------------------------------


class TestAutonomousGround:
    def test_filter_observation_returns_stale_data(self):
        paradigm = AutonomousGround()
        # Simulate a downlink at step 10
        obs_at_10 = _make_observation(step=10, battery_soc=0.75, mode="communication")
        paradigm.update_ground_knowledge(obs_at_10, step=10)

        # At step 50, the agent should see stale data from step 10
        obs_at_50 = _make_observation(step=50, battery_soc=0.5, mode="charging")
        filtered = paradigm.filter_observation(obs_at_50, step=50)

        # Should see the stale battery_soc from step 10, not step 50
        sat = filtered.constellation_state.satellites["eventsat_0"]
        assert sat.resources["battery_soc"] == 0.75
        assert sat.metadata["staleness_steps"] == 40
        assert sat.metadata["last_update_step"] == 10

    def test_stale_obs_carries_physical_capacity(self):
        # Regression: the ground planner must see achievable_downlink_mb (Phase B)
        # in the stale view — it's rebuilt metadata and used to drop these.
        paradigm = AutonomousGround()
        obs = _make_observation(step=10, ground_pass_active=True)
        meta_in = obs.constellation_state.satellites["eventsat_0"].metadata
        meta_in["achievable_downlink_mb"] = 2.5
        meta_in["daily_downlink_budget_mb"] = 27.0
        meta_in["storage_capacity_mb"] = 4096.0
        filtered = paradigm.filter_observation(obs, step=10)
        meta = filtered.constellation_state.satellites["eventsat_0"].metadata
        assert meta["achievable_downlink_mb"] == 2.5
        assert meta["daily_downlink_budget_mb"] == 27.0
        assert meta["storage_capacity_mb"] == 4096.0   # not the old hardcoded 1 TB

    def test_can_act_only_during_pass(self):
        paradigm = AutonomousGround()
        assert paradigm.can_act(step=0, ground_pass_active=False) is False
        assert paradigm.can_act(step=0, ground_pass_active=True) is True

    def test_process_action_during_pass_passes_through_mode(self):
        paradigm = AutonomousGround()
        action = {"eventsat_0": {"mode": "communication"}}

        # During pass: mode passes through, schedule (if any) is stripped
        result = paradigm.process_action(action, step=5, ground_pass_active=True)
        assert result == {"eventsat_0": {"mode": "communication"}}

    def test_process_action_schedule_stored_during_pass(self):
        paradigm = AutonomousGround()
        schedule = [("payload_observe", 3), ("charging", 2)]
        action = {"eventsat_0": {"mode": "communication", "schedule": schedule}}

        # During pass: schedule is stored, stripped from env-facing action
        result = paradigm.process_action(action, step=5, ground_pass_active=True)
        assert result == {"eventsat_0": {"mode": "communication"}}  # schedule stripped
        assert len(paradigm._schedule) == 2

        # Between passes: schedule plays back immediately (no planning delay)
        result2 = paradigm.process_action({}, step=6, ground_pass_active=False)
        assert result2 == {"eventsat_0": {"mode": "payload_observe"}}
        result3 = paradigm.process_action({}, step=7, ground_pass_active=False)
        assert result3 == {"eventsat_0": {"mode": "payload_observe"}}
        result4 = paradigm.process_action({}, step=8, ground_pass_active=False)
        assert result4 == {"eventsat_0": {"mode": "payload_observe"}}
        result5 = paradigm.process_action({}, step=9, ground_pass_active=False)
        assert result5 == {"eventsat_0": {"mode": "charging"}}

    def test_schedule_exhaustion_falls_back_to_default(self):
        paradigm = AutonomousGround()
        schedule = [("payload_compress", 2)]
        action = {"eventsat_0": {"mode": "communication", "schedule": schedule}}
        paradigm.process_action(action, step=0, ground_pass_active=True)

        # Consume the schedule
        paradigm.process_action({}, step=1, ground_pass_active=False)
        paradigm.process_action({}, step=2, ground_pass_active=False)
        # Schedule exhausted → fallback to default (charging)
        result = paradigm.process_action({}, step=3, ground_pass_active=False)
        assert result == {"eventsat_0": {"mode": "charging"}}

    def test_process_action_default_when_no_buffer(self):
        paradigm = AutonomousGround()
        result = paradigm.process_action(
            {"eventsat_0": {"mode": "payload_observe"}},
            step=0,
            ground_pass_active=False,
        )
        assert result == {"eventsat_0": {"mode": "charging"}}

    def test_reset_clears_schedule(self):
        paradigm = AutonomousGround()
        schedule = [("payload_observe", 5)]
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": schedule}},
            step=5,
            ground_pass_active=True,
        )
        paradigm.reset()

        # After reset, schedule is cleared — falls back to default
        result = paradigm.process_action(
            {"eventsat_0": {"mode": "safe"}}, step=0, ground_pass_active=False
        )
        assert result == {"eventsat_0": {"mode": "charging"}}

    def test_new_pass_replaces_old_schedule(self):
        paradigm = AutonomousGround()
        # First pass uploads schedule A
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": [("payload_observe", 10)]}},
            step=0, ground_pass_active=True,
        )
        # Second pass uploads schedule B (different)
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": [("payload_compress", 3)]}},
            step=100, ground_pass_active=True,
        )
        # Between second pass: schedule B plays back immediately
        result = paradigm.process_action({}, step=101, ground_pass_active=False)
        assert result == {"eventsat_0": {"mode": "payload_compress"}}

    def test_update_ground_knowledge(self):
        paradigm = AutonomousGround()
        obs = _make_observation(step=20, battery_soc=0.6, mode="communication")
        paradigm.update_ground_knowledge(obs, step=20)

        gk = paradigm.get_ground_knowledge()
        assert gk.last_update_step == 20
        assert gk.battery_soc == 0.6
        assert gk.current_mode == "communication"
        assert gk.staleness_steps == 0

    def test_filter_observation_shows_real_pass_status(self):
        paradigm = AutonomousGround()
        obs_during_pass = _make_observation(step=5, ground_pass_active=True)
        filtered = paradigm.filter_observation(obs_during_pass, step=5)
        sat = filtered.constellation_state.satellites["eventsat_0"]
        assert sat.metadata["ground_pass_active"] is True

    def test_filter_observation_no_pass_status_false(self):
        paradigm = AutonomousGround()
        obs_no_pass = _make_observation(step=5, ground_pass_active=False)
        filtered = paradigm.filter_observation(obs_no_pass, step=5)
        sat = filtered.constellation_state.satellites["eventsat_0"]
        assert sat.metadata["ground_pass_active"] is False

    def test_estimated_gap_steps_in_metadata_during_pass(self):
        paradigm = AutonomousGround(config={"orbital_period_steps": 93})
        obs = _make_observation(step=10, ground_pass_active=True)
        filtered = paradigm.filter_observation(obs, step=10)
        sat = filtered.constellation_state.satellites["eventsat_0"]
        assert sat.metadata["estimated_gap_steps"] == 93

    def test_estimated_gap_steps_absent_between_passes(self):
        paradigm = AutonomousGround(config={"orbital_period_steps": 93})
        obs = _make_observation(step=10, ground_pass_active=False)
        filtered = paradigm.filter_observation(obs, step=10)
        sat = filtered.constellation_state.satellites["eventsat_0"]
        assert "estimated_gap_steps" not in sat.metadata

    def test_get_name(self):
        paradigm = AutonomousGround()
        assert paradigm.get_name() == "AutonomousGround"

    def test_custom_default_mode(self):
        paradigm = AutonomousGround(config={"default_mode": "safe"})
        result = paradigm.process_action(
            {"eventsat_0": {"mode": "charging"}},
            step=0,
            ground_pass_active=False,
        )
        assert result == {"eventsat_0": {"mode": "safe"}}


# -----------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------


class TestConfigValidation:
    def test_valid_operations_paradigm_autonomous(self):
        from src.core.config_loader import ExperimentConfig

        cfg = ExperimentConfig(operations_paradigm="autonomous_hybrid")
        assert cfg.operations_paradigm == "autonomous_hybrid"

    def test_valid_operations_paradigm_autonomous_ground(self):
        from src.core.config_loader import ExperimentConfig

        cfg = ExperimentConfig(operations_paradigm="autonomous_ground")
        assert cfg.operations_paradigm == "autonomous_ground"

    def test_valid_operations_paradigm_conventional(self):
        from src.core.config_loader import ExperimentConfig

        cfg = ExperimentConfig(operations_paradigm="conventional_ground")
        assert cfg.operations_paradigm == "conventional_ground"

    def test_invalid_operations_paradigm_raises(self):
        from src.core.config_loader import ExperimentConfig

        with pytest.raises(Exception):
            ExperimentConfig(operations_paradigm="telepathic")

    def test_default_is_autonomous_hybrid(self):
        from src.core.config_loader import ExperimentConfig

        cfg = ExperimentConfig()
        assert cfg.operations_paradigm == "autonomous_hybrid"


# -----------------------------------------------------------------
# Integration: ExperimentRunner with operations paradigm
# -----------------------------------------------------------------


class TestExperimentRunnerIntegration:
    def test_baseline_with_autonomous_hybrid(self, tmp_path):
        """The existing baseline should work identically with autonomous_hybrid."""
        from src.core.config_loader import ExperimentConfig
        from src.core.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(
            experiment_id="ops_test_onboard",
            agent_organization="sas",
            decision_procedure="sda",
            representation="symbolic",
            behaviour="hand_designed",
            operations_paradigm="autonomous_hybrid",
            representation_config={"type": "rule_based_eventsat"},
            environment={"constellation_size": 1, "timestep_seconds": 60,
                         "max_steps": 100, "scenario": "eventsat",
                         "scenario_config": {}},
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

    def test_conventional_ground_with_rule_based_rejected(self, tmp_path):
        """A non-schedule rep (rule_based) under a ground paradigm is now rejected.

        rule_based_eventsat is a per-step controller; it emits no schedule, so it
        would degrade to 'charge between passes'. The validator blocks it — use a
        schedule producer (see test_conventional_ground_with_schedule_based).
        """
        from src.core.config_loader import ExperimentConfig

        with pytest.raises(ValueError, match="schedule-producing"):
            ExperimentConfig(
                experiment_id="ops_test_ground_rule",
                agent_organization="sas",
                decision_procedure="sda",
                representation="symbolic",
                behaviour="hand_designed",
                operations_paradigm="conventional_ground",
                representation_config={"type": "rule_based_eventsat"},
                environment={"constellation_size": 1, "timestep_seconds": 60,
                             "max_steps": 100, "scenario": "eventsat",
                             "scenario_config": {}},
                num_episodes=1,
                max_steps=100,
                save_checkpoints=False,
                log_level="WARNING",
                output_dir=str(tmp_path),
            )

    def test_conventional_ground_with_schedule_based(self, tmp_path):
        """Conventional ground + schedule_based_eventsat: the intended pairing."""
        from src.core.config_loader import ExperimentConfig
        from src.core.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(
            experiment_id="ops_test_ground_schedule",
            agent_organization="sas",
            decision_procedure="sda",
            representation="symbolic",
            behaviour="hand_designed",
            operations_paradigm="conventional_ground",
            operations_paradigm_config={"orbital_period_steps": 93},
            representation_config={"type": "schedule_based_eventsat"},
            environment={"constellation_size": 1, "timestep_seconds": 60,
                         "max_steps": 200, "scenario": "eventsat",
                         "scenario_config": {}},
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

    def test_autonomous_ground_with_schedule_based(self, tmp_path):
        """Autonomous ground + schedule_based_eventsat: algorithmic no-delay pairing."""
        from src.core.config_loader import ExperimentConfig
        from src.core.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(
            experiment_id="ops_test_ag_schedule",
            agent_organization="sas",
            decision_procedure="sda",
            representation="symbolic",
            behaviour="hand_designed",
            operations_paradigm="autonomous_ground",
            operations_paradigm_config={"orbital_period_steps": 93},
            representation_config={"type": "schedule_based_eventsat"},
            environment={"constellation_size": 1, "timestep_seconds": 60,
                         "max_steps": 200, "scenario": "eventsat",
                         "scenario_config": {}},
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


# -----------------------------------------------------------------
# Inference Gating (should_allow_inference)
# -----------------------------------------------------------------


class TestInferenceGating:
    """Verify that operations paradigms correctly gate inference timing.

    Ground-based paradigms (AG/CG) should only allow inference during
    ground passes, when fresh telemetry is available (Rossi et al. 2023).
    """

    def test_ah_always_allows_inference(self):
        """AutonomousHybrid allows inference every step (onboard autonomy)."""
        ah = AutonomousHybrid()
        assert ah.should_allow_inference(0, ground_pass_active=False) is True
        assert ah.should_allow_inference(50, ground_pass_active=True) is True
        assert ah.should_allow_inference(100, ground_pass_active=False) is True

    def test_ag_allows_inference_only_during_pass(self):
        """AutonomousGround allows inference only when ground pass active."""
        ag = AutonomousGround(config={"orbital_period_steps": 93})
        assert ag.should_allow_inference(10, ground_pass_active=False) is False
        assert ag.should_allow_inference(50, ground_pass_active=True) is True
        assert ag.should_allow_inference(60, ground_pass_active=False) is False

    def test_cg_allows_inference_only_during_pass(self):
        """ConventionalGround allows inference only when ground pass active."""
        from src.core.operations.conventional_ground import ConventionalGround
        cg = ConventionalGround(config={"orbital_period_steps": 93})
        assert cg.should_allow_inference(10, ground_pass_active=False) is False
        assert cg.should_allow_inference(50, ground_pass_active=True) is True
        assert cg.should_allow_inference(60, ground_pass_active=False) is False

    def test_base_default_allows_inference(self):
        """Base class default returns True (backward compatible)."""
        ah = AutonomousHybrid()
        assert ah.should_allow_inference(0, ground_pass_active=False) is True


# -----------------------------------------------------------------
# Ground-segment gap estimate (true pass-table horizon)
# -----------------------------------------------------------------


def _make_observation_with_lookahead(
    step=10,
    ground_pass_active=True,
    time_to_next_pass=700,
    remaining_pass_duration=4,
    following_gap_steps=650,
):
    obs = _make_observation(step=step, ground_pass_active=ground_pass_active)
    sat = obs.constellation_state.satellites["eventsat_0"]
    sat.metadata["time_to_next_pass"] = time_to_next_pass
    sat.metadata["remaining_pass_duration"] = remaining_pass_duration
    sat.metadata["following_gap_steps"] = following_gap_steps
    return obs


class TestGroundSegmentGapEstimate:
    """estimated_gap_steps comes from the env pass table, not one orbit.

    Pass prediction is deterministic ground-segment capability (Sellmaier
    et al. 2022 §16.4). Pre-fix, the hardcoded one-orbit value capped every
    ground schedule at 93 steps inside 92-764-step real gaps.
    """

    def test_ag_gap_is_pass_end_to_next_pass_start(self):
        ag = AutonomousGround(config={"orbital_period_steps": 93})
        obs = _make_observation_with_lookahead(
            time_to_next_pass=700, remaining_pass_duration=4
        )
        filtered = ag.filter_observation(obs, step=10)
        sat = filtered.constellation_state.satellites["eventsat_0"]
        assert sat.metadata["estimated_gap_steps"] == 696

    def test_ag_falls_back_to_orbital_period_without_lookahead(self):
        ag = AutonomousGround(config={"orbital_period_steps": 93})
        obs = _make_observation(step=10, ground_pass_active=True)
        filtered = ag.filter_observation(obs, step=10)
        sat = filtered.constellation_state.satellites["eventsat_0"]
        assert sat.metadata["estimated_gap_steps"] == 93

    def test_cg_gap_is_the_following_gap(self):
        """CG's schedule executes one pass later, so it plans the gap after next."""
        cg = ConventionalGround(config={"orbital_period_steps": 93})
        obs = _make_observation_with_lookahead(following_gap_steps=650)
        filtered = cg.filter_observation(obs, step=10)
        sat = filtered.constellation_state.satellites["eventsat_0"]
        assert sat.metadata["estimated_gap_steps"] == 650

    def test_cg_falls_back_to_next_gap_then_orbital_period(self):
        cg = ConventionalGround(config={"orbital_period_steps": 93})
        obs = _make_observation_with_lookahead(
            time_to_next_pass=700, remaining_pass_duration=4, following_gap_steps=None
        )
        sat_meta = obs.constellation_state.satellites["eventsat_0"].metadata
        sat_meta["following_gap_steps"] = None
        filtered = cg.filter_observation(obs, step=10)
        sat = filtered.constellation_state.satellites["eventsat_0"]
        assert sat.metadata["estimated_gap_steps"] == 696

        obs_bare = _make_observation(step=10, ground_pass_active=True)
        filtered = cg.filter_observation(obs_bare, step=10)
        sat = filtered.constellation_state.satellites["eventsat_0"]
        assert sat.metadata["estimated_gap_steps"] == 93


# -----------------------------------------------------------------
# ConventionalGround link-gated uplink
# -----------------------------------------------------------------


class TestConventionalGroundLinkGating:
    """Schedule uplink requires an actually-established comm link.

    A pass too short for ADCS settling to complete never reaches
    communication mode: nothing is transferred in either direction
    (observed at step 3441 of the 2026-06-12 CG probe trace, where a
    schedule was 'uploaded' over a link that never existed).
    """

    def _action(self, schedule=None):
        sat = {"mode": "communication"}
        if schedule is not None:
            sat["schedule"] = schedule
        return {"eventsat_0": sat}

    def test_promotion_requires_link(self):
        """Pass with established comm: schedule promotes when telemetry flows."""
        cg = ConventionalGround()
        cg._planned_schedule = [["payload_observe", 3], ["charging", 90]]

        # Pass starts: schedule staged, not yet active
        cg.process_action(self._action(), step=100, ground_pass_active=True)
        assert cg._upload_candidate is not None
        assert cg._active_schedule == []

        # Link established mid-pass (runner fires this on resolved comm mode)
        cg.update_ground_knowledge(_make_observation(step=101), step=101)
        assert cg._upload_candidate is None

        # Between passes: the uplinked schedule executes
        result = cg.process_action(self._action(), step=103, ground_pass_active=False)
        assert result == {"eventsat_0": {"mode": "payload_observe"}}

    def test_no_link_returns_schedule_to_planned(self):
        """Settling-swallowed pass: nothing transfers; ops team retries next pass."""
        cg = ConventionalGround()
        planned = [["payload_observe", 3], ["charging", 90]]
        cg._planned_schedule = [list(e) for e in planned]

        # 2-step pass, comm never established (no update_ground_knowledge)
        cg.process_action(self._action(), step=100, ground_pass_active=True)
        cg.process_action(self._action(), step=101, ground_pass_active=True)

        # Pass ends: satellite has no schedule, executes default mode
        result = cg.process_action(self._action(), step=102, ground_pass_active=False)
        assert result == {"eventsat_0": {"mode": "charging"}}
        assert cg._active_schedule == []
        # The schedule is back in planned, staged for the next contact
        assert cg._planned_schedule == planned

        # Next pass with a real link: it finally uplinks and executes
        cg.process_action(self._action(), step=200, ground_pass_active=True)
        cg.update_ground_knowledge(_make_observation(step=201), step=201)
        result = cg.process_action(self._action(), step=205, ground_pass_active=False)
        assert result == {"eventsat_0": {"mode": "payload_observe"}}

"""Tests for the Operations Paradigm dimension."""

import pytest

from src.environment.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteState,
)
from src.operations.base import GroundKnowledge, OperationsParadigm
from src.operations.autonomous_hybrid import AutonomousHybrid
from src.operations.conventional_ground import ConventionalGround


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
# ConventionalGround
# -----------------------------------------------------------------


class TestConventionalGround:
    def test_filter_observation_returns_stale_data(self):
        paradigm = ConventionalGround()
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

    def test_can_act_only_during_pass(self):
        paradigm = ConventionalGround()
        assert paradigm.can_act(step=0, ground_pass_active=False) is False
        assert paradigm.can_act(step=0, ground_pass_active=True) is True

    def test_process_action_buffers_during_pass(self):
        paradigm = ConventionalGround()
        action = {"eventsat_0": {"mode": "payload_observe"}}

        # During pass: action goes through and is stored
        result = paradigm.process_action(action, step=5, ground_pass_active=True)
        assert result == action

        # After pass: last commanded action is replayed
        result2 = paradigm.process_action(
            {"eventsat_0": {"mode": "safe"}}, step=10, ground_pass_active=False
        )
        assert result2 == action  # replays the buffered action, not the new one

    def test_process_action_default_when_no_buffer(self):
        paradigm = ConventionalGround()
        result = paradigm.process_action(
            {"eventsat_0": {"mode": "payload_observe"}},
            step=0,
            ground_pass_active=False,
        )
        assert result == {"eventsat_0": {"mode": "charging"}}

    def test_reset_clears_buffer(self):
        paradigm = ConventionalGround()
        paradigm.process_action(
            {"eventsat_0": {"mode": "payload_observe"}},
            step=5,
            ground_pass_active=True,
        )
        paradigm.reset()

        # After reset, should fall back to default
        result = paradigm.process_action(
            {"eventsat_0": {"mode": "safe"}}, step=0, ground_pass_active=False
        )
        assert result == {"eventsat_0": {"mode": "charging"}}

    def test_update_ground_knowledge(self):
        paradigm = ConventionalGround()
        obs = _make_observation(step=20, battery_soc=0.6, mode="communication")
        paradigm.update_ground_knowledge(obs, step=20)

        gk = paradigm.get_ground_knowledge()
        assert gk.last_update_step == 20
        assert gk.battery_soc == 0.6
        assert gk.current_mode == "communication"
        assert gk.staleness_steps == 0

    def test_get_name(self):
        paradigm = ConventionalGround()
        assert paradigm.get_name() == "ConventionalGround"

    def test_custom_default_mode(self):
        paradigm = ConventionalGround(config={"default_mode": "safe"})
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
        from src.orchestration.config_loader import ExperimentConfig

        cfg = ExperimentConfig(operations_paradigm="autonomous_hybrid")
        assert cfg.operations_paradigm == "autonomous_hybrid"

    def test_valid_operations_paradigm_conventional(self):
        from src.orchestration.config_loader import ExperimentConfig

        cfg = ExperimentConfig(operations_paradigm="conventional_ground")
        assert cfg.operations_paradigm == "conventional_ground"

    def test_invalid_operations_paradigm_raises(self):
        from src.orchestration.config_loader import ExperimentConfig

        with pytest.raises(Exception):
            ExperimentConfig(operations_paradigm="telepathic")

    def test_default_is_autonomous_hybrid(self):
        from src.orchestration.config_loader import ExperimentConfig

        cfg = ExperimentConfig()
        assert cfg.operations_paradigm == "autonomous_hybrid"


# -----------------------------------------------------------------
# Integration: ExperimentRunner with operations paradigm
# -----------------------------------------------------------------


class TestExperimentRunnerIntegration:
    def test_baseline_with_autonomous_hybrid(self):
        """The existing baseline should work identically with autonomous_hybrid."""
        from src.orchestration.config_loader import ExperimentConfig
        from src.orchestration.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(
            experiment_id="ops_test_onboard",
            agent_organization="centralized",
            decision_loop="sda",
            representation="symbolic",
            emergence_mode="hand_designed",
            operations_paradigm="autonomous_hybrid",
            representation_config={"type": "rule_based_eventsat"},
            environment={"constellation_size": 1, "timestep_seconds": 60,
                         "max_steps": 100, "scenario": "eventsat",
                         "scenario_config": {}},
            num_episodes=1,
            max_steps=100,
            save_checkpoints=False,
            log_level="WARNING",
        )
        runner = ExperimentRunner(config=cfg)
        results = runner.run()
        assert results["num_episodes"] == 1
        assert len(results["episodes"][0]["steps"]) == 100

    def test_baseline_with_conventional_ground(self):
        """Conventional ground should run without errors (degraded performance expected)."""
        from src.orchestration.config_loader import ExperimentConfig
        from src.orchestration.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(
            experiment_id="ops_test_ground",
            agent_organization="centralized",
            decision_loop="sda",
            representation="symbolic",
            emergence_mode="hand_designed",
            operations_paradigm="conventional_ground",
            representation_config={"type": "rule_based_eventsat"},
            environment={"constellation_size": 1, "timestep_seconds": 60,
                         "max_steps": 100, "scenario": "eventsat",
                         "scenario_config": {}},
            num_episodes=1,
            max_steps=100,
            save_checkpoints=False,
            log_level="WARNING",
        )
        runner = ExperimentRunner(config=cfg)
        results = runner.run()
        assert results["num_episodes"] == 1
        assert len(results["episodes"][0]["steps"]) == 100

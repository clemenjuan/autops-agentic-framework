"""Tests for the new ConventionalGround operations paradigm.

The new ConventionalGround models realistic human flight dynamics operations
with a one-pass planning delay (Sellmaier et al. 2022, ECSS-E-ST-70C):

  - Schedule planned at pass N is uploaded at pass N+1 (one-pass delay)
  - Cold start: first pass has no prior schedule; satellite stays in default_mode
  - Two internal buffers: _active_schedule (executing) and _planned_schedule (waiting)
  - During every pass: satellite always communicates (downlinking + HK)
"""

import pytest

from src.environment.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteState,
)
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
# Basic interface
# -----------------------------------------------------------------


class TestConventionalGroundBasic:
    def test_get_name(self):
        paradigm = ConventionalGround()
        assert paradigm.get_name() == "ConventionalGround"

    def test_can_act_only_during_pass(self):
        paradigm = ConventionalGround()
        assert paradigm.can_act(step=0, ground_pass_active=False) is False
        assert paradigm.can_act(step=0, ground_pass_active=True) is True

    def test_default_mode_charging(self):
        paradigm = ConventionalGround()
        # Between passes with no schedule: falls back to default (charging)
        result = paradigm.process_action({}, step=5, ground_pass_active=False)
        assert result == {"eventsat_0": {"mode": "charging"}}

    def test_custom_default_mode(self):
        paradigm = ConventionalGround(config={"default_mode": "safe"})
        result = paradigm.process_action({}, step=5, ground_pass_active=False)
        assert result == {"eventsat_0": {"mode": "safe"}}

    def test_during_pass_always_communication(self):
        paradigm = ConventionalGround()
        # During pass: always returns communication regardless of requested mode
        result = paradigm.process_action(
            {"eventsat_0": {"mode": "payload_observe"}},
            step=5, ground_pass_active=True
        )
        assert result == {"eventsat_0": {"mode": "communication"}}

    def test_reset_clears_buffers(self):
        paradigm = ConventionalGround()
        # Give it a planned schedule
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": [("payload_observe", 5)]}},
            step=5, ground_pass_active=True
        )
        paradigm.reset()
        assert paradigm._active_schedule == []
        assert paradigm._planned_schedule is None
        assert paradigm._active_index == 0
        assert paradigm._last_pass_active is False


# -----------------------------------------------------------------
# One-pass delay (the key differentiator from AutonomousGround)
# -----------------------------------------------------------------


class TestOnPassDelay:
    def test_cold_start_no_schedule_until_second_pass(self):
        """First pass: no prior schedule to upload — satellite stays in default_mode."""
        paradigm = ConventionalGround()

        # Pass 1: send a schedule (this becomes _planned_schedule, NOT active yet)
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": [("payload_observe", 3)]}},
            step=0, ground_pass_active=True
        )

        # Between passes 1 and 2: active_schedule is still empty → default_mode
        result = paradigm.process_action({}, step=1, ground_pass_active=False)
        assert result == {"eventsat_0": {"mode": "charging"}}

    def test_planned_schedule_promoted_at_next_pass(self):
        """Schedule generated at pass N becomes active at pass N+1."""
        paradigm = ConventionalGround()

        # Pass 1: schedule stored as _planned_schedule
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": [("payload_observe", 3)]}},
            step=0, ground_pass_active=True
        )
        # Between passes: default_mode (active still empty)
        paradigm.process_action({}, step=1, ground_pass_active=False)

        # Pass 2: planned → active (upload). Also generate a new planned schedule.
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": [("payload_compress", 2)]}},
            step=100, ground_pass_active=True
        )

        # Between passes 2 and 3: NOW the pass-1 schedule plays back
        result = paradigm.process_action({}, step=101, ground_pass_active=False)
        assert result == {"eventsat_0": {"mode": "payload_observe"}}

    def test_schedule_delayed_two_passes(self):
        """Full two-pass delay chain: telemetry N-1 → schedule N-1 → upload at N."""
        paradigm = ConventionalGround(config={"orbital_period_steps": 5})

        # Pass 1: store schedule A as _planned_schedule
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": [("payload_compress", 2)]}},
            step=0, ground_pass_active=True
        )
        # Gap 1-2: default (no active schedule)
        for s in range(1, 5):
            r = paradigm.process_action({}, step=s, ground_pass_active=False)
            assert r == {"eventsat_0": {"mode": "charging"}}, f"step {s}: expected charging"

        # Pass 2: A promoted to active; store schedule B as _planned_schedule
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": [("payload_detect", 4)]}},
            step=5, ground_pass_active=True
        )
        # Gap 2-3: schedule A plays back
        r = paradigm.process_action({}, step=6, ground_pass_active=False)
        assert r == {"eventsat_0": {"mode": "payload_compress"}}
        r = paradigm.process_action({}, step=7, ground_pass_active=False)
        assert r == {"eventsat_0": {"mode": "payload_compress"}}
        # Schedule A exhausted → default
        r = paradigm.process_action({}, step=8, ground_pass_active=False)
        assert r == {"eventsat_0": {"mode": "charging"}}

        # Pass 3: B promoted to active
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": []}},
            step=10, ground_pass_active=True
        )
        # Gap 3-4: schedule B plays back
        r = paradigm.process_action({}, step=11, ground_pass_active=False)
        assert r == {"eventsat_0": {"mode": "payload_detect"}}

    def test_no_schedule_sent_at_pass_leaves_planned_as_none(self):
        """If no schedule in action, _planned_schedule stays None."""
        paradigm = ConventionalGround()
        # Pass with no schedule key
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication"}},
            step=0, ground_pass_active=True
        )
        assert paradigm._planned_schedule is None

    def test_empty_schedule_not_stored(self):
        """Empty schedule list is not stored (treated as no schedule)."""
        paradigm = ConventionalGround()
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": []}},
            step=0, ground_pass_active=True
        )
        assert paradigm._planned_schedule is None


# -----------------------------------------------------------------
# Two-buffer management
# -----------------------------------------------------------------


class TestTwoBufferManagement:
    def test_planned_replaces_active_at_pass_start(self):
        """At each pass start, _planned is promoted to _active."""
        paradigm = ConventionalGround()

        # Pass 1: plan A
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": [("charging", 10)]}},
            step=0, ground_pass_active=True
        )
        assert paradigm._planned_schedule is not None
        assert paradigm._active_schedule == []

        # Between passes (required to reset _last_pass_active)
        paradigm.process_action({}, step=1, ground_pass_active=False)

        # Pass 2: plan A → active, plan B stored
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": [("payload_observe", 5)]}},
            step=100, ground_pass_active=True
        )
        assert paradigm._active_schedule[0][0] == "charging"
        assert paradigm._planned_schedule is not None
        assert paradigm._planned_schedule[0][0] == "payload_observe"

    def test_active_schedule_exhaustion_falls_to_default(self):
        """When active schedule runs out, default_mode is used."""
        paradigm = ConventionalGround()

        # Pass 1: store 1-step schedule as planned
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": [("payload_compress", 1)]}},
            step=0, ground_pass_active=True
        )
        # Between passes (required to reset _last_pass_active)
        paradigm.process_action({}, step=1, ground_pass_active=False)
        # Pass 2: promote planned → active
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": []}},
            step=100, ground_pass_active=True
        )
        # Gap: one step of payload_compress, then default
        r1 = paradigm.process_action({}, step=101, ground_pass_active=False)
        assert r1 == {"eventsat_0": {"mode": "payload_compress"}}
        r2 = paradigm.process_action({}, step=102, ground_pass_active=False)
        assert r2 == {"eventsat_0": {"mode": "charging"}}

    def test_multi_step_pass_only_stores_once(self):
        """Schedule captured on first uplink step of pass; subsequent pass steps ignored."""
        paradigm = ConventionalGround()

        # Step 0: pass start, send schedule A
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": [("charging", 5)]}},
            step=0, ground_pass_active=True
        )
        # Step 1: still in pass, try to overwrite with schedule B (should be ignored)
        paradigm.process_action(
            {"eventsat_0": {"mode": "communication", "schedule": [("payload_observe", 3)]}},
            step=1, ground_pass_active=True
        )
        # _planned_schedule should still be schedule A (charging, 5)
        assert paradigm._planned_schedule is not None
        assert paradigm._planned_schedule[0][0] == "charging"


# -----------------------------------------------------------------
# filter_observation: stale telemetry
# -----------------------------------------------------------------


class TestConventionalGroundObservation:
    def test_filter_observation_returns_stale_state(self):
        paradigm = ConventionalGround()
        obs_at_10 = _make_observation(step=10, battery_soc=0.75, mode="communication")
        paradigm.update_ground_knowledge(obs_at_10, step=10)

        obs_at_50 = _make_observation(step=50, battery_soc=0.5, mode="charging")
        filtered = paradigm.filter_observation(obs_at_50, step=50)

        sat = filtered.constellation_state.satellites["eventsat_0"]
        assert sat.resources["battery_soc"] == 0.75
        assert sat.metadata["staleness_steps"] == 40
        assert sat.metadata["last_update_step"] == 10

    def test_estimated_gap_steps_present_during_pass(self):
        paradigm = ConventionalGround(config={"orbital_period_steps": 93})
        obs = _make_observation(step=10, ground_pass_active=True)
        filtered = paradigm.filter_observation(obs, step=10)
        sat = filtered.constellation_state.satellites["eventsat_0"]
        assert sat.metadata["estimated_gap_steps"] == 93

    def test_estimated_gap_steps_absent_between_passes(self):
        paradigm = ConventionalGround(config={"orbital_period_steps": 93})
        obs = _make_observation(step=10, ground_pass_active=False)
        filtered = paradigm.filter_observation(obs, step=10)
        sat = filtered.constellation_state.satellites["eventsat_0"]
        assert "estimated_gap_steps" not in sat.metadata


# -----------------------------------------------------------------
# Integration: ConventionalGround + ExperimentRunner
# -----------------------------------------------------------------


class TestConventionalGroundIntegration:
    def test_conventional_ground_with_conventional_schedule(self, tmp_path):
        """Human-realistic paradigm + human schedule representation: end-to-end."""
        from src.orchestration.config_loader import ExperimentConfig
        from src.orchestration.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(
            experiment_id="cg_test_conventional_schedule",
            agent_organization="sas",
            decision_loop="sda",
            representation="symbolic",
            emergence_mode="hand_designed",
            operations_paradigm="conventional_ground",
            operations_paradigm_config={"orbital_period_steps": 93},
            representation_config={"type": "conventional_schedule_eventsat"},
            environment={
                "constellation_size": 1,
                "timestep_seconds": 60,
                "max_steps": 300,
                "scenario": "eventsat",
                "scenario_config": {},
            },
            num_episodes=1,
            max_steps=300,
            save_checkpoints=False,
            log_level="WARNING",
            output_dir=str(tmp_path),
        )
        runner = ExperimentRunner(config=cfg)
        results = runner.run()
        assert results["num_episodes"] == 1
        assert len(results["episodes"][0]["steps"]) == 300

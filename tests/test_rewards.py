"""Tests for EventSat reward functions (Individual Negative, autops-rl Case 2)."""
from __future__ import annotations

import math

import pytest
from src.eventsat.rewards import EventSatRewardFunction


@pytest.fixture
def rf():
    """Default reward function with scale=1.0 for easier testing."""
    return EventSatRewardFunction({"reward_scale": 1.0})


class TestResourcePenalty:
    def test_no_penalty_healthy_resources(self, rf):
        """No penalty when battery > threshold and storage < threshold."""
        p = rf.resource_penalty(battery_soc=0.8, data_stored_mb=100.0, storage_capacity_mb=512.0)
        assert p == 0.0

    def test_low_battery_penalty(self, rf):
        """Penalty when battery below threshold."""
        p = rf.resource_penalty(battery_soc=0.15, data_stored_mb=0.0, storage_capacity_mb=512.0)
        assert p < 0.0

    def test_high_storage_penalty(self, rf):
        """Penalty when storage above threshold."""
        p = rf.resource_penalty(battery_soc=0.8, data_stored_mb=480.0, storage_capacity_mb=512.0)
        assert p < 0.0

    def test_both_bad(self, rf):
        """Combined penalty is worse than either alone."""
        p_bat = rf.resource_penalty(battery_soc=0.1, data_stored_mb=0.0, storage_capacity_mb=512.0)
        p_stor = rf.resource_penalty(battery_soc=0.8, data_stored_mb=500.0, storage_capacity_mb=512.0)
        p_both = rf.resource_penalty(battery_soc=0.1, data_stored_mb=500.0, storage_capacity_mb=512.0)
        assert p_both < p_bat
        assert p_both < p_stor


class TestActionReward:
    def test_observe_default_has_no_direct_reward(self, rf):
        r = rf.action_reward("payload_observe", {"storage_overflow": False})
        assert r == 0.0

    def test_observe_shaping_is_opt_in(self):
        shaped = EventSatRewardFunction({"reward_scale": 1.0, "observe_reward": 1.0})
        r = shaped.action_reward("payload_observe", {"storage_overflow": False})
        assert r > 0.0

    def test_observe_overflow_reduced(self):
        shaped = EventSatRewardFunction({"reward_scale": 1.0, "observe_reward": 1.0})
        r_ok = shaped.action_reward("payload_observe", {"storage_overflow": False})
        r_of = shaped.action_reward("payload_observe", {"storage_overflow": True})
        assert r_of < r_ok
        assert r_of < 0.0

    def test_compress_with_data(self, rf):
        r = rf.action_reward("payload_compress", {"had_data_to_compress": True})
        assert r == 0.0

    def test_compress_no_data(self, rf):
        r = rf.action_reward("payload_compress", {"had_data_to_compress": False})
        assert r < 0.0

    def test_comm_success(self, rf):
        r = rf.action_reward("communication", {"pass_active": True, "data_downlinked_mb": 2.0})
        assert r > 0.0

    def test_comm_capped(self, rf):
        r = rf.action_reward("communication", {"pass_active": True, "data_downlinked_mb": 100.0})
        assert r == rf.comm_reward_cap

    def test_comm_no_pass(self, rf):
        r = rf.action_reward("communication", {"pass_active": False})
        assert r < 0.0

    def test_clamped_constraint_violation_is_failed_action(self, rf):
        """A clamped invalid command must not inherit neutral charging reward."""
        r = rf.action_reward(
            "charging",
            {"constraint_violation": True},
        )
        assert r == -rf.failed_action_penalty

    def test_empty_comm_during_pass_is_penalized(self, rf):
        r = rf.action_reward(
            "communication",
            {"pass_active": True, "data_downlinked_mb": 0.0},
        )
        assert r < 0.0

    def test_charging_neutral(self, rf):
        """The default objective does not privilege activity over waiting."""
        r = rf.action_reward("charging", {})
        assert r == 0.0

    def test_safe_more_negative_than_charging(self, rf):
        r_charge = rf.action_reward("charging", {})
        r_safe = rf.action_reward("safe", {})
        assert r_safe < r_charge


class TestMissionPenalty:
    def test_no_penalty_targets_met(self, rf):
        """No mission penalty when all targets achieved."""
        p = rf.mission_penalty(
            is_final_step=True, obs_hours=2.0, downlinked_mb=240.0,
            obs_target_hours=2.0, downlink_target_mb=240.0,
            episode_steps=10080, max_mission_steps=10080,
        )
        assert p == 0.0

    def test_full_penalty_nothing_done(self, rf):
        """Full penalty at final step with nothing accomplished."""
        p = rf.mission_penalty(
            is_final_step=True, obs_hours=0.0, downlinked_mb=0.0,
            obs_target_hours=2.0, downlink_target_mb=240.0,
            episode_steps=10080, max_mission_steps=10080,
        )
        assert p == pytest.approx(-rf.mission_scale)

    def test_partial_progress_partial_penalty(self, rf):
        """Partial achievement gives partial penalty."""
        p = rf.mission_penalty(
            is_final_step=True, obs_hours=1.0, downlinked_mb=120.0,
            obs_target_hours=2.0, downlink_target_mb=240.0,
            episode_steps=10080, max_mission_steps=10080,
        )
        assert -rf.mission_scale < p < 0.0

    def test_observation_without_downlink_gets_full_delivered_only_penalty(self, rf):
        """Default mission reward tracks delivered information, not hoarded observations."""
        p = rf.mission_penalty(
            is_final_step=True, obs_hours=2.0, downlinked_mb=0.0,
            obs_target_hours=2.0, downlink_target_mb=240.0,
            episode_steps=10080, max_mission_steps=10080,
        )
        assert p == pytest.approx(-rf.mission_scale)

    def test_observation_weight_is_explicit_ablation(self):
        rf_weighted = EventSatRewardFunction({
            "reward_scale": 1.0,
            "mission_weights": {"observation": 1.0, "downlink": 1.0},
        })
        p = rf_weighted.mission_penalty(
            is_final_step=True, obs_hours=2.0, downlinked_mb=0.0,
            obs_target_hours=2.0, downlink_target_mb=240.0,
            episode_steps=10080, max_mission_steps=10080,
        )
        assert p == pytest.approx(-0.5 * rf_weighted.mission_scale)

    def test_penalty_scales_with_progress(self, rf):
        """Mid-episode penalty is smaller than end-of-episode."""
        p_early = rf.mission_penalty(
            is_final_step=False, obs_hours=0.0, downlinked_mb=0.0,
            obs_target_hours=2.0, downlink_target_mb=240.0,
            episode_steps=100, max_mission_steps=10080,
        )
        p_late = rf.mission_penalty(
            is_final_step=False, obs_hours=0.0, downlinked_mb=0.0,
            obs_target_hours=2.0, downlink_target_mb=240.0,
            episode_steps=9000, max_mission_steps=10080,
        )
        assert p_early > p_late  # Both negative, early is less negative

    def test_undelivered_observations_do_not_reduce_default_penalty(self, rf):
        """Raw collection is not reward until its product reaches the ground."""
        kwargs = dict(
            is_final_step=True,
            downlinked_mb=0.0,
            obs_target_hours=2.0,
            downlink_target_mb=240.0,
            episode_steps=10080,
            max_mission_steps=10080,
        )
        no_observations = rf.mission_penalty(obs_hours=0.0, **kwargs)
        observation_target_met = rf.mission_penalty(obs_hours=2.0, **kwargs)
        assert observation_target_met == no_observations

    def test_observation_credit_is_explicit_ablation(self):
        rf = EventSatRewardFunction(
            {
                "reward_scale": 1.0,
                "mission_observation_weight": 1.0,
                "mission_downlink_weight": 1.0,
            }
        )
        p = rf.mission_penalty(
            is_final_step=True,
            obs_hours=2.0,
            downlinked_mb=0.0,
            obs_target_hours=2.0,
            downlink_target_mb=240.0,
            episode_steps=10080,
            max_mission_steps=10080,
        )
        assert p == pytest.approx(-0.5 * rf.mission_scale)


class TestComputeTotal:
    def test_scaled_output(self):
        """Total reward is scaled by reward_scale."""
        rf_scaled = EventSatRewardFunction({"reward_scale": 0.01})
        rf_unscaled = EventSatRewardFunction({"reward_scale": 1.0})
        kwargs = dict(
            mode="payload_observe", battery_soc=0.8,
            data_stored_mb=100.0, storage_capacity_mb=512.0,
            action_info={"storage_overflow": False},
            obs_hours=1.0, downlinked_mb=100.0,
            obs_target_hours=2.0, downlink_target_mb=240.0,
            episode_step=5000, max_steps=10080, is_final_step=False,
        )
        r_scaled = rf_scaled.compute(**kwargs)
        r_unscaled = rf_unscaled.compute(**kwargs)
        assert r_scaled == pytest.approx(r_unscaled * 0.01)

    def test_pipeline_action_does_not_beat_waiting_by_construction(self):
        """Healthy successful pipeline work is neutral until delivery."""
        rf = EventSatRewardFunction({"reward_scale": 1.0})
        kwargs_base = dict(
            battery_soc=0.8, data_stored_mb=100.0, storage_capacity_mb=512.0,
            obs_hours=0.5, downlinked_mb=50.0,
            obs_target_hours=2.0, downlink_target_mb=240.0,
            episode_step=5000, max_steps=10080, is_final_step=False,
        )
        r_obs = rf.compute(mode="payload_observe", action_info={"storage_overflow": False}, **kwargs_base)
        r_chg = rf.compute(mode="charging", action_info={}, **kwargs_base)
        assert r_obs == r_chg

    def test_observe_beats_charging_when_shaping_is_enabled(self):
        """Observation shaping remains available when explicitly configured."""
        rf = EventSatRewardFunction({"reward_scale": 1.0, "observe_reward": 1.0})
        kwargs_base = dict(
            battery_soc=0.8, data_stored_mb=100.0, storage_capacity_mb=512.0,
            obs_hours=0.5, downlinked_mb=50.0,
            obs_target_hours=2.0, downlink_target_mb=240.0,
            episode_step=5000, max_steps=10080, is_final_step=False,
        )
        r_obs = rf.compute(
            mode="payload_observe",
            action_info={"storage_overflow": False},
            **kwargs_base,
        )
        r_chg = rf.compute(mode="charging", action_info={}, **kwargs_base)
        assert r_obs > r_chg


class TestIntegrationWithEnv:
    """Test that the reward function integrates with EventSatEnvironment."""

    def test_env_uses_reward_fn(self):
        from src.eventsat.env import EventSatEnvironment
        env = EventSatEnvironment({
            "scenario_config": "configs/scenarios/eventsat.yaml",
            "max_steps": 100,
        })
        assert isinstance(env.reward_fn, EventSatRewardFunction)
        obs = env.reset(seed=42)
        result = env.step({"eventsat_0": {"mode": "payload_observe"}})
        assert "total" in result.rewards

    def test_env_storage_penalty_uses_obc_occupancy(self, monkeypatch):
        """Jetson backlog must not be normalised by the smaller OBC capacity."""
        from src.eventsat.env import EventSatEnvironment

        env = EventSatEnvironment({
            "scenario_config": "configs/scenarios/eventsat.yaml",
            "max_steps": 100,
        })
        env.reset(seed=42)
        env.jetson_raw_mb = 12_000.0
        env.jetson_compressed_mb = 500.0
        env.obc_data_mb = 123.0
        captured = {}

        def capture_reward(**kwargs):
            captured.update(kwargs)
            return 0.0

        monkeypatch.setattr(env.reward_fn, "compute", capture_reward)
        env.step({"eventsat_0": {"mode": "charging"}})

        assert captured["data_stored_mb"] == pytest.approx(123.0)
        assert captured["storage_capacity_mb"] == pytest.approx(
            env.storage_capacity_mb
        )

    def test_env_merges_scenario_rewards_with_explicit_override(self):
        from src.eventsat.env import EventSatEnvironment

        env = EventSatEnvironment({
            "max_steps": 100,
            "scenario_params": {
                "rewards": {"observe_reward": 0.25},
                "power": {"battery": {"capacity_wh": 70.0}},
                "communications": {"sband": {"downlink_rate_kbps": 50}},
            },
        })
        assert env.reward_fn.observe_reward == pytest.approx(0.25)

        override = EventSatEnvironment({
            "max_steps": 100,
            "scenario_params": {
                "rewards": {"observe_reward": 0.25},
                "power": {"battery": {"capacity_wh": 70.0}},
                "communications": {"sband": {"downlink_rate_kbps": 50}},
            },
            "reward_config": {"observe_reward": 0.75},
        })
        assert override.reward_fn.observe_reward == pytest.approx(0.75)

    def test_no_free_charging_reward(self):
        """Charging should not give positive reward (bug fix)."""
        from src.eventsat.env import EventSatEnvironment
        env = EventSatEnvironment({
            "scenario_config": "configs/scenarios/eventsat.yaml",
            "max_steps": 100,
        })
        env.reset(seed=42)
        result = env.step({"eventsat_0": {"mode": "charging"}})
        assert result.rewards["total"] <= 0.0

    @staticmethod
    def _constraint_reward_env():
        from src.eventsat.env import EventSatEnvironment

        return EventSatEnvironment({
            "max_steps": 10,
            "anomaly_prob": 0.0,
            "scenario_params": {
                "rewards": {
                    "reward_scale": 1.0,
                    "resource_penalty_factor": 0.0,
                    "mission_scale": 0.0,
                    "failed_action_penalty": 0.1,
                    "safe_penalty": 0.3,
                },
                "modes": {
                    "transition_overhead": {"settling_time_s": 0.0},
                },
            },
        })

    def test_env_penalizes_operational_command_clamped_to_charging(self):
        env = self._constraint_reward_env()
        env.battery_soc = 0.35

        result = env.step({"eventsat_0": {"mode": "payload_observe"}})

        assert result.info["requested_mode"] == "payload_observe"
        assert result.info["resolved_mode"] == "charging"
        assert result.info["constraint_violation"] is True
        assert result.rewards["total"] == pytest.approx(-0.1)

    @pytest.mark.parametrize("onboard", [False, True])
    def test_out_of_contact_attempt_pays_radio_power_and_preserves_data(self, monkeypatch, onboard):
        env = self._constraint_reward_env()
        env.consumption = {
            "communication": {"eclipse_w": 33.24},
            "charging": {"eclipse_w": 4.32},
        }
        env.onboard_compute_active = onboard
        env.battery_soc = 0.8
        env.obc_data_mb = 5.0
        monkeypatch.setattr(env, "_is_ground_pass_active", lambda: False)
        monkeypatch.setattr(env, "_is_in_sunlight", lambda: False)

        attempt = env.step({"eventsat_0": {"mode": "communication"}})
        energy = (33.24 + (env.onboard_compute_w if onboard else 0)) / 60.0
        assert attempt.info["requested_mode"] == "communication"
        assert attempt.info["resolved_mode"] == "communication"
        assert attempt.info["communication_failure"] == "no_contact"
        assert attempt.info["step_downlinked_mb"] == 0.0
        assert attempt.info["gross_energy_consumed_wh"] == pytest.approx(energy)
        assert env.battery_soc == pytest.approx(0.8 - energy / env.battery_capacity_wh)
        assert env.obc_data_mb == 5.0
        assert attempt.rewards["total"] == pytest.approx(-0.1)
        assert attempt.info["constraint_violation"] is False

        charging = env.step({"eventsat_0": {"mode": "charging"}})
        assert charging.info["gross_energy_consumed_wh"] < energy
        assert charging.rewards["total"] == pytest.approx(0.0)

    def test_out_of_contact_attempt_respects_settling_and_safe_mode(self, monkeypatch):
        env = self._constraint_reward_env()
        env.settling_time_steps = 3
        env.attitude_maneuver_modes = {"communication", "payload_observe"}
        monkeypatch.setattr(env, "_is_ground_pass_active", lambda: False)
        results = [env.step({"eventsat_0": {"mode": "communication"}}) for _ in range(4)]
        assert [r.info["resolved_mode"] for r in results] == [
            "charging", "charging", "charging", "communication",
        ]
        assert all("communication_failure" not in r.info for r in results[:3])
        assert results[-1].info["communication_failure"] == "no_contact"
        env.settling_time_steps = 0
        env.battery_soc = 0.19
        protected = env.step({"eventsat_0": {"mode": "communication"}})
        assert protected.info["resolved_mode"] == "safe"
        assert "communication_failure" not in protected.info
        assert protected.rewards["total"] == pytest.approx(-0.3)

    def test_env_does_not_misclassify_forced_safe_as_constraint_failure(self):
        env = self._constraint_reward_env()
        env.battery_soc = 0.19

        result = env.step({"eventsat_0": {"mode": "payload_observe"}})

        assert result.info["resolved_mode"] == "safe"
        assert result.info["constraint_violation"] is False
        assert result.rewards["total"] == pytest.approx(-0.3)

    def test_reward_components_present(self):
        """Reward should be a finite number from structured computation."""
        from src.eventsat.env import EventSatEnvironment
        env = EventSatEnvironment({
            "scenario_config": "configs/scenarios/eventsat.yaml",
            "max_steps": 100,
        })
        env.reset(seed=42)
        for mode in ["charging", "payload_observe", "payload_compress"]:
            result = env.step({"eventsat_0": {"mode": mode}})
            total = result.rewards["total"]
            assert isinstance(total, float)
            assert math.isfinite(total)
            assert -1.0 <= total <= 1.0

    def test_overflow_rejects_observation_without_mission_credit(self):
        """A discarded product cannot increment observations or raw-data totals."""
        from src.eventsat.env import EventSatEnvironment
        env = EventSatEnvironment({
            "scenario_config": "configs/scenarios/eventsat.yaml",
            "scenario_overrides": {
                "modes": {"transition_overhead": {"settling_time_s": 0.0}},
            },
            "max_steps": 100,
        })
        env.reset(seed=42)
        # Raw and compressed products share the same physical Jetson volume.
        env.jetson_compressed_mb = (
            env.jetson_capacity_mb - env.observation_size_mb / 2.0
        )
        before_raw = env.jetson_raw_mb
        before_compressed = env.jetson_compressed_mb

        result = env.step({"eventsat_0": {"mode": "payload_observe"}})

        assert result.info["storage_overflow"] is True
        assert env.jetson_raw_mb == pytest.approx(before_raw)
        assert env.jetson_compressed_mb == pytest.approx(before_compressed)
        assert env.uncompressed_observations == 0
        assert env.total_observation_s == 0.0
        assert env.total_raw_captured_mb == 0.0
        assert result.rewards["total"] < 0.0

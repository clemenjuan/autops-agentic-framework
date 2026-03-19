"""Tests for EventSat reward functions (Individual Negative, autops-rl Case 2)."""
from __future__ import annotations

import pytest
from src.environment.rewards import EventSatRewardFunction


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
    def test_observe_positive(self, rf):
        r = rf.action_reward("payload_observe", {"storage_overflow": False})
        assert r > 0.0

    def test_observe_overflow_reduced(self, rf):
        r_ok = rf.action_reward("payload_observe", {"storage_overflow": False})
        r_of = rf.action_reward("payload_observe", {"storage_overflow": True})
        assert r_of < r_ok
        assert r_of > 0.0  # Still positive, just reduced

    def test_compress_with_data(self, rf):
        r = rf.action_reward("payload_compress", {"had_data_to_compress": True})
        assert r > 0.0

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

    def test_charging_negative(self, rf):
        """Charging (idle) gives small negative -- no free reward."""
        r = rf.action_reward("charging", {})
        assert r < 0.0

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

    def test_observe_beats_charging(self):
        """Observing with good resources gives higher reward than charging."""
        rf = EventSatRewardFunction({"reward_scale": 1.0})
        kwargs_base = dict(
            battery_soc=0.8, data_stored_mb=100.0, storage_capacity_mb=512.0,
            obs_hours=0.5, downlinked_mb=50.0,
            obs_target_hours=2.0, downlink_target_mb=240.0,
            episode_step=5000, max_steps=10080, is_final_step=False,
        )
        r_obs = rf.compute(mode="payload_observe", action_info={"storage_overflow": False}, **kwargs_base)
        r_chg = rf.compute(mode="charging", action_info={}, **kwargs_base)
        assert r_obs > r_chg


class TestIntegrationWithEnv:
    """Test that the reward function integrates with EventSatEnvironment."""

    def test_env_uses_reward_fn(self):
        from src.environment.scenarios.eventsat_env import EventSatEnvironment
        env = EventSatEnvironment({
            "scenario_config": "configs/scenarios/eventsat.yaml",
            "max_steps": 100,
        })
        assert isinstance(env.reward_fn, EventSatRewardFunction)
        obs = env.reset(seed=42)
        result = env.step({"eventsat_0": {"mode": "payload_observe"}})
        assert "total" in result.rewards

    def test_no_free_charging_reward(self):
        """Charging should not give positive reward (bug fix)."""
        from src.environment.scenarios.eventsat_env import EventSatEnvironment
        env = EventSatEnvironment({
            "scenario_config": "configs/scenarios/eventsat.yaml",
            "max_steps": 100,
        })
        env.reset(seed=42)
        result = env.step({"eventsat_0": {"mode": "charging"}})
        assert result.rewards["total"] <= 0.0

    def test_reward_components_present(self):
        """Reward should be a finite number from structured computation."""
        from src.environment.scenarios.eventsat_env import EventSatEnvironment
        env = EventSatEnvironment({
            "scenario_config": "configs/scenarios/eventsat.yaml",
            "max_steps": 100,
        })
        env.reset(seed=42)
        for mode in ["charging", "payload_observe", "payload_compress"]:
            result = env.step({"eventsat_0": {"mode": mode}})
            assert isinstance(result.rewards["total"], float)
            assert result.rewards["total"] != float("inf")

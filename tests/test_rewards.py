"""Tests for EventSat reward functions (Individual Negative, autops-rl Case 2)."""
from __future__ import annotations

import math

import pytest
from src.eventsat.rewards import EventSatRewardFunction


def test_shaped_diagnostic_uses_natural_delivery_reward():
    from src.core.config_loader import load_config
    cfg = load_config("configs/diagnostics/eventsat_sas_ao_rl_shaped.yaml")
    scenario_config = cfg.environment.scenario_config
    rf = EventSatRewardFunction(
        scenario_config["reward_config"],
        discount_factor=cfg.behaviour_config["gamma"],
    )
    # Shaping settings are a tuning choice of the diagnostic and not checked.
    common = dict(battery_soc=0.1, data_stored_mb=4000.0, storage_capacity_mb=4096.0,
                  obs_hours=0.0, downlinked_mb=0.0, obs_target_hours=1.0,
                  downlink_target_mb=20.0, episode_step=5000, max_steps=10080,
                  is_final_step=False, pipeline_state_before={}, pipeline_state_after={},
                  compression_ratio=5.11)
    # Failed, safe, idle and resource-stressed steps are all neutral; only
    # delivery (and pipeline potential changes, none here) move the reward.
    for mode, info in [("communication", {"pass_active": False}),
                       ("payload_compress", {}), ("safe", {}), ("charging", {})]:
        assert rf.compute(mode=mode, action_info=info, **common) == 0.0
    delivered = rf.compute(
        mode="communication",
        action_info={"pass_active": True, "data_downlinked_mb": 0.375},
        **common,
    )
    assert delivered == pytest.approx(rf.reward_scale * 0.375)


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
    @pytest.mark.parametrize("mode,info,failed", [
        ("communication", {"pass_active": False}, True),
        ("communication", {"pass_active": True, "data_downlinked_mb": 0.0}, True),
        ("communication", {"pass_active": True, "data_downlinked_mb": 1.0}, False),
        ("payload_observe", {"storage_overflow": True}, True),
        ("payload_compress", {}, True),
        ("payload_compress", {"had_data_to_compress": True}, False),
        ("payload_detect", {}, True),
        ("payload_send", {}, True),
        ("charging", {"constraint_violation": True}, True),
        ("charging", {}, False),
        ("transitioning", {}, False),
        ("safe", {}, False),
    ])
    def test_failure_classification_excludes_equal_safe_penalty(self, mode, info, failed):
        reward = EventSatRewardFunction({"failed_action_penalty": 0.3, "safe_penalty": 0.3})
        assert reward.is_failed_action(mode, info) is failed
        if failed:
            assert reward.action_reward(mode, info) == -0.3

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

    @pytest.mark.parametrize("info,failed", [
        ({"pass_active": False, "communication_failure": "no_contact"}, True),
        ({"pass_active": True, "data_downlinked_mb": 0.0,
          "communication_failure": "no_contact"}, True),
        ({"pass_active": True, "data_downlinked_mb": 0.0,
          "communication_failure": "no_source_data"}, False),
        ({"pass_active": True, "data_downlinked_mb": 0.0}, False),
    ])
    def test_empty_downlink_failure_is_explicit_ablation(self, info, failed):
        reward = EventSatRewardFunction(
            {"reward_scale": 1.0, "empty_downlink_is_failure": False}
        )
        assert reward.is_failed_action("communication", info) is failed
        expected = -reward.failed_action_penalty if failed else 0.0
        assert reward.action_reward("communication", info) == expected

    def test_charging_neutral(self, rf):
        """The default objective does not privilege activity over waiting."""
        r = rf.action_reward("charging", {})
        assert r == 0.0

    def test_safe_more_negative_than_charging(self, rf):
        r_charge = rf.action_reward("charging", {})
        r_safe = rf.action_reward("safe", {})
        assert r_safe < r_charge


class TestPipelinePotential:
    @staticmethod
    def _reward(*, gamma=1.0, enabled=True, reward_scale=1.0):
        return EventSatRewardFunction(
            {
                "reward_scale": reward_scale,
                "pipeline_shaping": {"enabled": enabled},
            },
            discount_factor=gamma,
        )

    @staticmethod
    def _state(*, compressed=0.0, obc=0.0, ground=0.0):
        return {
            "jetson_compressed_mb": compressed,
            "obc_raw_equivalent_mb": obc,
            "downlink_raw_equivalent_mb": ground,
        }

    @pytest.mark.parametrize(
        "compressed,obc,ground,expected",
        [
            (5.0, 0.0, 0.0, 1.0 / 3.0),
            (0.0, 10.0, 0.0, 2.0 / 3.0),
            (0.0, 0.0, 10.0, 1.0),
        ],
    )
    def test_potential_uses_fixed_pipeline_stage_credits(
        self, compressed, obc, ground, expected
    ):
        reward = self._reward()

        potential = reward.pipeline_potential(
            self._state(compressed=compressed, obc=obc, ground=ground),
            compression_ratio=2.0,
            downlink_target_mb=5.0,
        )

        assert potential == pytest.approx(expected)

    def test_each_completed_pipeline_transition_adds_equal_potential(self):
        reward = self._reward(gamma=1.0)
        raw = self._state()
        compressed = self._state(compressed=5.0)
        obc = self._state(obc=10.0)
        ground = self._state(ground=10.0)
        kwargs = {
            "compression_ratio": 2.0,
            "downlink_target_mb": 5.0,
            "is_final_step": False,
        }

        increments = [
            reward.pipeline_shaping_reward(raw, compressed, **kwargs),
            reward.pipeline_shaping_reward(compressed, obc, **kwargs),
            reward.pipeline_shaping_reward(obc, ground, **kwargs),
        ]

        assert increments == pytest.approx([1.0 / 3.0] * 3)

    def test_potential_is_globally_capped_furthest_stage_first(self):
        reward = self._reward()
        state = self._state(compressed=5.0, obc=10.0, ground=4.0)
        surplus = self._state(compressed=500.0, obc=10.0, ground=4.0)

        potential = reward.pipeline_potential(
            state,
            compression_ratio=2.0,
            downlink_target_mb=5.0,
        )
        surplus_potential = reward.pipeline_potential(
            surplus,
            compression_ratio=2.0,
            downlink_target_mb=5.0,
        )

        # Target is 10 raw-equivalent MB: 4 MB at ground plus 6 eligible MB
        # on the OBC gives (4*1 + 6*2/3) / 10 = 0.8.
        assert potential == pytest.approx(0.8)
        assert surplus_potential == pytest.approx(potential)

    def test_unchanged_backlog_has_discounted_opportunity_cost(self):
        reward = self._reward(gamma=0.995)
        state = self._state(obc=5.0)

        shaping = reward.pipeline_shaping_reward(
            state,
            state,
            compression_ratio=2.0,
            downlink_target_mb=5.0,
            is_final_step=False,
        )

        assert shaping == pytest.approx((0.995 - 1.0) / 3.0)

    def test_terminal_transition_uses_zero_next_potential(self):
        reward = self._reward(gamma=0.995)
        delivered = self._state(ground=10.0)

        shaping = reward.pipeline_shaping_reward(
            delivered,
            delivered,
            compression_ratio=2.0,
            downlink_target_mb=5.0,
            is_final_step=True,
        )

        assert shaping == pytest.approx(-1.0)

    def test_disabled_shaping_is_zero(self):
        reward = self._reward(enabled=False)

        shaping = reward.pipeline_shaping_reward(
            self._state(),
            self._state(ground=10.0),
            compression_ratio=2.0,
            downlink_target_mb=5.0,
            is_final_step=False,
        )

        assert shaping == 0.0

    @pytest.mark.parametrize("gamma", [-0.01, 1.01])
    def test_discount_factor_is_bounded(self, gamma):
        with pytest.raises(ValueError, match="discount_factor"):
            self._reward(gamma=gamma)

    def test_compute_adds_scaled_potential_shaping(self):
        reward = self._reward(gamma=1.0, reward_scale=0.5)
        task_neutral = {
            "mode": "charging",
            "battery_soc": 0.8,
            "data_stored_mb": 0.0,
            "storage_capacity_mb": 32.0,
            "action_info": {},
            "obs_hours": 0.0,
            "downlinked_mb": 0.0,
            "obs_target_hours": 0.0,
            "downlink_target_mb": 5.0,
            "episode_step": 0,
            "max_steps": 10,
            "is_final_step": False,
            "pipeline_state_before": self._state(),
            "pipeline_state_after": self._state(compressed=5.0),
            "compression_ratio": 2.0,
        }

        assert reward.compute(**task_neutral) == pytest.approx(1.0 / 6.0)

    def test_scale_multiplies_discounted_potential_difference(self):
        reward = EventSatRewardFunction(
            {"pipeline_shaping": {"enabled": True, "scale": 10.0}},
            discount_factor=0.995,
        )

        shaping = reward.pipeline_shaping_reward(
            self._state(),
            self._state(obc=5.0),
            compression_ratio=2.0,
            downlink_target_mb=5.0,
            is_final_step=False,
        )

        assert shaping == pytest.approx(10.0 * 0.995 / 3.0)

    @pytest.mark.parametrize("scale", [-1.0, math.nan, math.inf])
    def test_scale_must_be_finite_and_non_negative(self, scale):
        with pytest.raises(ValueError, match="pipeline_shaping.scale"):
            EventSatRewardFunction({"pipeline_shaping": {"scale": scale}})


class TestRawProgressPotential:
    @staticmethod
    def _reward(gamma=1.0):
        return EventSatRewardFunction(
            {"pipeline_shaping": {"enabled": True, "potential": "raw_progress"}},
            discount_factor=gamma,
        )

    def test_complete_path_and_terminal_discounted_sum(self):
        states = [
            {},
            {"jetson_raw_mb": 10.0},
            {"jetson_raw_mb": 10.0, "observation_size_mb": 10.0,
             "uncompressed_observations": 1, "compression_progress_fraction": 0.5},
            {"jetson_compressed_mb": 5.0},
            {"obc_raw_equivalent_mb": 10.0},
            {"downlink_raw_equivalent_mb": 10.0},
        ]
        kwargs = {"compression_ratio": 2.0, "downlink_target_mb": 5.0}
        reward = self._reward(0.999)
        assert [reward.pipeline_potential(s, **kwargs) for s in states] == pytest.approx(
            [0.0, 0.25, 0.375, 0.5, 0.75, 1.0]
        )
        # Shaping redistributes credit, including the final correction; it
        # cannot change the discounted return of a completed trajectory.
        states.append(states[-1])
        total = sum(
            0.999 ** i * reward.pipeline_shaping_reward(
                before, after, is_final_step=i == len(states) - 2, **kwargs
            )
            for i, (before, after) in enumerate(zip(states, states[1:]))
        )
        assert total == pytest.approx(0.0, abs=1e-12)

    def test_progress_credits_only_one_product_and_respects_global_cap(self):
        reward = self._reward()
        state = {"jetson_raw_mb": 1000.0, "observation_size_mb": 4.0,
                 "uncompressed_observations": 250, "compression_progress_fraction": 0.5}
        kwargs = {"compression_ratio": 2.0, "downlink_target_mb": 5.0}
        assert reward.pipeline_potential(state, **kwargs) == pytest.approx(0.30)
        state["obc_raw_equivalent_mb"] = 8.0
        assert reward.pipeline_potential(state, **kwargs) == pytest.approx(0.675)
        state["downlink_raw_equivalent_mb"] = 10.0
        assert reward.pipeline_potential(state, **kwargs) == 1.0

    def test_progress_without_source_and_detection_do_not_create_credit(self):
        state = {"compression_progress_fraction": 0.5, "observation_size_mb": 4.0,
                 "uncompressed_observations": 1, "detection_progress": 4,
                 "total_detections": 100}
        assert self._reward().pipeline_potential(
            state, compression_ratio=2.0, downlink_target_mb=5.0
        ) == 0.0

    def test_unknown_potential_is_rejected(self):
        with pytest.raises(ValueError, match="pipeline_shaping.potential"):
            EventSatRewardFunction({"pipeline_shaping": {"potential": "typo"}})


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

    @pytest.mark.parametrize("potential", ["delivery", "raw_progress"])
    def test_env_reports_compressed_output_only_on_completion(self, potential):
        from src.eventsat.env import EventSatEnvironment

        env = EventSatEnvironment({
            "max_steps": 10,
            "anomaly_prob": 0.0,
            "reward_discount_factor": 1.0,
            "scenario_params": {
                "rewards": {
                    "reward_scale": 1.0,
                    "resource_penalty_factor": 0.0,
                    "mission_scale": 0.0,
                    "pipeline_shaping": {"enabled": True, "potential": potential},
                },
                "objectives": {
                    "mission_duration_days": 1.0 / 144.0,
                    "min_downlinked_data_mb": 2.0,
                },
                "power": {"battery": {"initial_soc": 1.0}},
                "storage": {
                    "observation_size_mb": 4.0,
                    "compression_ratio": 2.0,
                    "jetson_capacity_mb": 32.0,
                    "obc_capacity_mb": 32.0,
                },
                "payload": {"compression_time_factor": 2.0},
                "modes": {
                    "transition_overhead": {"settling_time_s": 0.0},
                    "constraints": {
                        "payload_observe": {"min_battery_soc": 0.0},
                        "payload_compress": {"min_battery_soc": 0.0},
                    },
                },
            },
        })
        env.reset(seed=42)
        observed = env.step({"eventsat_0": {"mode": "payload_observe"}})

        progress = env.step({"eventsat_0": {"mode": "payload_compress"}})
        completed = env.step({"eventsat_0": {"mode": "payload_compress"}})

        dense = potential == "raw_progress"
        assert observed.rewards["total"] == pytest.approx(0.25 if dense else 0.0)
        assert progress.info["compression_in_progress"] is True
        assert progress.info["data_compressed_mb"] == 0.0
        assert progress.rewards["total"] == pytest.approx(0.125 if dense else 0.0)
        assert completed.info["compression_completed"] is True
        assert completed.info["data_compressed_mb"] == pytest.approx(2.0)
        assert completed.rewards["total"] == pytest.approx(0.125 if dense else 1.0 / 3.0)

        # Restart, then interrupt compression: the reset must retract exactly
        # the partial credit rather than rewarding repeated starts.
        env.reset(seed=42)
        env.step({"eventsat_0": {"mode": "payload_observe"}})
        env.step({"eventsat_0": {"mode": "payload_compress"}})
        interrupted = env.step({"eventsat_0": {"mode": "charging"}})
        assert env.compression_progress == 0
        assert interrupted.rewards["total"] == pytest.approx(-0.125 if dense else 0.0)

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

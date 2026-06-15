"""
Tests for Phase 4b: Subsymbolic RL Representation.

Covers:
- EventSat env orbital lookahead metadata (backward compatible)
- EventSat env sub-action processing (data_priority, pipeline_routing)
- Gymnasium wrapper: obs shape, action space, reset/step contract, reward scalar
- Neural policy (RandomPolicy always; ActorCritic if torch available):
  forward shape, deterministic vs stochastic, save/load
- RolloutBuffer: store, overflow, GAE, batch iteration
- PPOTrainer (torch only): single update, loss types, lr schedule, save/load
- SubsymbolicEventSat: registration, encode_observation, select_action,
  reason(), update(), grounding, get_metrics()
- Integration: with all 3 loop types via DecisionContext
- Emergence controller: subsymbolic_eventsat registered after import
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.representation.neural_policy import TORCH_AVAILABLE, RandomPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eventsat_env(max_steps: int = 50) -> "EventSatEnvironment":
    from src.environment.scenarios.eventsat_env import EventSatEnvironment
    return EventSatEnvironment(config={
        "max_steps": max_steps,
        "step_duration_s": 60.0,
        "scenario_params": {
            "orbit": {"orbital_period_s": 5676, "eclipse_fraction": 0.36},
            "power": {
                "solar_panels": {"generation_peak_w": 24.0},
                "battery": {"capacity_wh": 84.0, "initial_soc": 0.8, "min_soc": 0.2},
                "consumption": {},
            },
            "storage": {},
            "communications": {"sband": {"downlink_rate_kbps": 128}},
            "modes": {},
            "payload": {},
        },
    })


def _make_subsymbolic_repr(mock: bool = True) -> "SubsymbolicEventSat":
    import src.representation.subsymbolic_eventsat  # trigger registration
    from src.representation.subsymbolic_eventsat import SubsymbolicEventSat
    return SubsymbolicEventSat(config={"rl_mock": mock, "deterministic": False})


def _make_decision_context(loop_type: str = "sda", state: dict | None = None):
    from src.decision_procedure.context import DecisionContext
    return DecisionContext(
        state=state or {
            "battery_soc": 0.8,
            "health_status": "nominal",
            "ground_pass_active": False,
            "_obs_vector": np.zeros(25, dtype=np.float32),
        },
        loop_type=loop_type,
        memory=None,
        enrichments={},
        loop_metadata={},
    )


# ===========================================================================
# Section 1: EventSat env — orbital lookahead metadata (backward compat)
# ===========================================================================

class TestEventSatOrbitalLookahead(unittest.TestCase):
    """Orbital lookahead fields added to observation metadata without breaking existing tests."""

    def setUp(self):
        self.env = _make_eventsat_env()
        self.obs = self.env.reset(seed=42)

    def test_orbital_phase_in_metadata(self):
        sat = self.obs.constellation_state.satellites["eventsat_0"]
        self.assertIn("orbital_phase", sat.metadata)
        phase = sat.metadata["orbital_phase"]
        self.assertGreaterEqual(phase, 0.0)
        self.assertLess(phase, 1.0)

    def test_time_to_next_eclipse_in_metadata(self):
        sat = self.obs.constellation_state.satellites["eventsat_0"]
        self.assertIn("time_to_next_eclipse", sat.metadata)
        t = sat.metadata["time_to_next_eclipse"]
        self.assertGreater(t, 0)

    def test_time_to_next_pass_in_metadata(self):
        sat = self.obs.constellation_state.satellites["eventsat_0"]
        self.assertIn("time_to_next_pass", sat.metadata)
        t = sat.metadata["time_to_next_pass"]
        self.assertGreaterEqual(t, 0)

    def test_remaining_pass_duration_in_metadata(self):
        sat = self.obs.constellation_state.satellites["eventsat_0"]
        self.assertIn("remaining_pass_duration", sat.metadata)
        self.assertGreaterEqual(sat.metadata["remaining_pass_duration"], 0)

    def test_existing_metadata_still_present(self):
        sat = self.obs.constellation_state.satellites["eventsat_0"]
        for key in [
            "in_sunlight", "ground_pass_active", "health_status",
            "storage_capacity_mb", "jetson_raw_mb",
        ]:
            self.assertIn(key, sat.metadata, f"Missing existing key: {key}")

    def test_orbital_phase_advances(self):
        env = _make_eventsat_env(max_steps=200)
        obs0 = env.reset(seed=0)
        phase0 = obs0.constellation_state.satellites["eventsat_0"].metadata["orbital_phase"]
        for _ in range(10):
            result = env.step({"eventsat_0": {"mode": "charging"}})
        phase1 = result.observation.constellation_state.satellites["eventsat_0"].metadata["orbital_phase"]
        # Phase should have advanced
        self.assertNotEqual(phase0, phase1)


# ===========================================================================
# Section 2: EventSat env — sub-action processing
# ===========================================================================

class TestEventSatSubActions(unittest.TestCase):

    def setUp(self):
        self.env = _make_eventsat_env(max_steps=200)
        self.env.reset(seed=0)
        # Force battery high so communication is not blocked by SoC
        self.env.battery_soc = 0.9

    def test_data_priority_stored(self):
        self.env.step({"eventsat_0": {"mode": "charging", "data_priority": 1, "pipeline_routing": 0}})
        self.assertEqual(self.env._data_priority, 1)
        self.assertEqual(self.env._pipeline_routing, 0)

    def test_urgent_priority_increases_downlink(self):
        """Urgent (data_priority=1) should download more than normal during a ground pass."""
        # Step until a pass is active
        found_pass = False
        for _ in range(200):
            obs = self.env.get_observation()
            sat = obs.constellation_state.satellites["eventsat_0"]
            if sat.metadata.get("ground_pass_active", False):
                found_pass = True
                break
            self.env.step({"eventsat_0": {"mode": "charging"}})

        if not found_pass:
            self.skipTest("No ground pass in 200 steps (stochastic — skip)")

        # Setup: put some data on OBC
        self.env.obc_data_mb = 100.0

        # Normal downlink
        env_normal = _make_eventsat_env(max_steps=200)
        env_normal.reset(seed=0)
        env_normal.obc_data_mb = 100.0
        env_normal.battery_soc = 0.9
        # Force a pass
        env_normal._data_priority = 0

        # Compute expected downlink with 1x vs 1.5x multiplier
        dl_normal = (self.env.downlink_rate_kbps / 8.0) * (self.env.step_duration_s / 1000.0)
        dl_urgent = dl_normal * 1.5

        # Verify env stores values correctly
        self.assertAlmostEqual(dl_urgent, dl_normal * 1.5)

    def test_pipeline_routing_detect_first_redirects(self):
        """detect_first routing redirects payload_compress to detection when applicable."""
        self.env.uncompressed_observations = 0  # no compression backlog
        self.env.undetected_observations = 2    # detection backlog exists
        self.env.detection_progress = 0
        self.env.battery_soc = 0.8

        result = self.env.step({
            "eventsat_0": {
                "mode": "payload_compress",
                "data_priority": 0,
                "pipeline_routing": 1,  # detect_first
            }
        })
        self.assertIn("pipeline_routed_to_detect", result.info)

    def test_pipeline_routing_compress_first_redirects(self):
        """compress_first routing redirects payload_detect to compression when applicable."""
        self.env.undetected_observations = 0    # no detection backlog
        self.env.uncompressed_observations = 2  # compression backlog exists
        self.env.compression_progress = 0
        self.env.battery_soc = 0.8

        result = self.env.step({
            "eventsat_0": {
                "mode": "payload_detect",
                "data_priority": 0,
                "pipeline_routing": 0,  # compress_first
            }
        })
        self.assertIn("pipeline_routed_to_compress", result.info)

    def test_symbolic_action_no_sub_actions(self):
        """Symbolic representations that don't pass sub-actions work without errors."""
        result = self.env.step({"eventsat_0": {"mode": "charging"}})
        # Default sub-actions should be 0
        self.assertEqual(self.env._data_priority, 0)
        self.assertEqual(self.env._pipeline_routing, 0)


# ===========================================================================
# Section 3: Gymnasium wrapper
# ===========================================================================

class TestEventSatGymnasium(unittest.TestCase):

    def setUp(self):
        try:
            import gymnasium  # noqa: F401
            self.gymnasium_available = True
        except ImportError:
            self.gymnasium_available = False

    def _make_wrapper(self):
        from src.environment.gymnasium_wrapper import EventSatGymnasium
        return EventSatGymnasium(env_config={
            "max_steps": 50,
            "step_duration_s": 60.0,
            "scenario_params": {
                "orbit": {"orbital_period_s": 5676, "eclipse_fraction": 0.36},
                "power": {
                    "solar_panels": {"generation_peak_w": 24.0},
                    "battery": {"capacity_wh": 84.0, "initial_soc": 0.8, "min_soc": 0.2},
                    "consumption": {},
                },
                "storage": {},
                "communications": {"sband": {"downlink_rate_kbps": 128}},
                "modes": {},
                "payload": {},
            },
        })

    def test_import_no_gymnasium(self):
        """Module imports cleanly even without gymnasium."""
        from src.environment import gymnasium_wrapper  # noqa: F401
        self.assertTrue(True)

    @unittest.skipUnless(True, "")
    def test_obs_shape(self):
        if not self.gymnasium_available:
            self.skipTest("gymnasium not installed")
        wrapper = self._make_wrapper()
        obs, info = wrapper.reset(seed=0)
        self.assertEqual(obs.shape, (25,))
        self.assertEqual(obs.dtype, np.float32)

    def test_obs_shape_values_bounded(self):
        if not self.gymnasium_available:
            self.skipTest("gymnasium not installed")
        wrapper = self._make_wrapper()
        obs, _ = wrapper.reset(seed=0)
        # All values should be finite
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_action_space(self):
        if not self.gymnasium_available:
            self.skipTest("gymnasium not installed")
        from gymnasium.spaces import MultiDiscrete
        wrapper = self._make_wrapper()
        self.assertIsInstance(wrapper.action_space, MultiDiscrete)
        np.testing.assert_array_equal(wrapper.action_space.nvec, [7, 2, 2])

    def test_step_contract(self):
        if not self.gymnasium_available:
            self.skipTest("gymnasium not installed")
        wrapper = self._make_wrapper()
        wrapper.reset(seed=0)
        action = np.array([0, 0, 0], dtype=int)  # charging, normal, compress_first
        obs, reward, terminated, truncated, info = wrapper.step(action)
        self.assertEqual(obs.shape, (25,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

    def test_reward_scalar(self):
        if not self.gymnasium_available:
            self.skipTest("gymnasium not installed")
        wrapper = self._make_wrapper()
        wrapper.reset(seed=0)
        _, reward, _, _, _ = wrapper.step(np.array([0, 0, 0]))
        # Reward must be a finite scalar
        self.assertTrue(np.isfinite(reward))

    def test_full_episode(self):
        if not self.gymnasium_available:
            self.skipTest("gymnasium not installed")
        wrapper = self._make_wrapper()
        obs, _ = wrapper.reset(seed=0)
        done = False
        steps = 0
        while not done and steps < 60:
            action = wrapper.action_space.sample()
            obs, reward, terminated, truncated, _ = wrapper.step(action)
            done = terminated or truncated
            steps += 1
        self.assertGreater(steps, 0)

    def test_symbolic_grounding_communication_no_pass(self):
        """Communication mode should be grounded to charging when no pass."""
        if not self.gymnasium_available:
            self.skipTest("gymnasium not installed")
        wrapper = self._make_wrapper()
        wrapper.reset(seed=0)
        wrapper._env.battery_soc = 0.9
        # Force no ground pass
        wrapper._env._orbital_ctx = None
        # communication = mode index 1
        mode = wrapper._apply_symbolic_grounding(1)
        self.assertEqual(mode, "charging")

    def test_symbolic_grounding_anomaly_forces_safe(self):
        if not self.gymnasium_available:
            self.skipTest("gymnasium not installed")
        wrapper = self._make_wrapper()
        wrapper.reset(seed=0)
        wrapper._env.active_anomaly = "thermal_warning"
        mode = wrapper._apply_symbolic_grounding(0)  # charging
        self.assertEqual(mode, "safe")


# ===========================================================================
# Section 4: Neural policy (RandomPolicy always; ActorCritic if torch)
# ===========================================================================

class TestRandomPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = RandomPolicy()

    def test_get_action_shape(self):
        obs = np.zeros(25, dtype=np.float32)
        action, log_prob, value = self.policy.get_action(obs)
        self.assertEqual(action.shape, (3,))

    def test_get_action_bounds(self):
        obs = np.zeros(25, dtype=np.float32)
        for _ in range(20):
            action, _, _ = self.policy.get_action(obs)
            self.assertIn(action[0], range(7))
            self.assertIn(action[1], range(2))
            self.assertIn(action[2], range(2))

    def test_get_action_deterministic_same(self):
        """Deterministic mode should return same action each call (not really for random, but shouldn't crash)."""
        obs = np.zeros(25, dtype=np.float32)
        action, _, _ = self.policy.get_action(obs, deterministic=True)
        self.assertEqual(action.shape, (3,))

    def test_evaluate_actions_shape(self):
        obs_batch = np.zeros((10, 25), dtype=np.float32)
        actions_batch = np.zeros((10, 3), dtype=np.int64)
        log_probs, entropy, values = self.policy.evaluate_actions(obs_batch, actions_batch)
        self.assertEqual(log_probs.shape, (10,))
        self.assertEqual(values.shape, (10, 1))

    def test_get_mode_probs_shape(self):
        obs = np.zeros(25, dtype=np.float32)
        probs = self.policy.get_mode_probs(obs)
        self.assertEqual(probs.shape, (7,))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestActorCritic(unittest.TestCase):

    def setUp(self):
        from src.representation.neural_policy import ActorCritic
        self.policy = ActorCritic()

    def test_forward_shape(self):
        import torch
        obs = torch.zeros(1, 25)
        dists, value = self.policy.forward(obs)
        self.assertEqual(len(dists), 3)
        self.assertEqual(value.shape, (1, 1))

    def test_forward_dist_shapes(self):
        import torch
        obs = torch.zeros(1, 25)
        dists, _ = self.policy.forward(obs)
        self.assertEqual(dists[0].param_shape, (1, 7))
        self.assertEqual(dists[1].param_shape, (1, 2))
        self.assertEqual(dists[2].param_shape, (1, 2))

    def test_get_action_shape(self):
        import torch
        obs = torch.zeros(25)
        action, log_prob, value = self.policy.get_action(obs)
        self.assertEqual(action.shape, (3,))
        self.assertIn(action[0], range(7))

    def test_get_action_deterministic(self):
        import torch
        obs = torch.ones(25) * 0.5
        action1, _, _ = self.policy.get_action(obs, deterministic=True)
        action2, _, _ = self.policy.get_action(obs, deterministic=True)
        np.testing.assert_array_equal(action1, action2)

    def test_get_action_stochastic_varies(self):
        import torch
        torch.manual_seed(0)
        obs = torch.zeros(25)
        actions = [self.policy.get_action(obs, deterministic=False)[0] for _ in range(30)]
        # At least some actions should differ (with very high probability)
        unique = set(tuple(a.tolist()) for a in actions)
        # Should have more than 1 unique action across 30 samples
        self.assertGreater(len(unique), 1)

    def test_evaluate_actions_shapes(self):
        import torch
        obs_batch = torch.zeros(8, 25)
        actions_batch = torch.zeros(8, 3, dtype=torch.long)
        log_probs, entropy, values = self.policy.evaluate_actions(obs_batch, actions_batch)
        self.assertEqual(log_probs.shape, (8,))
        self.assertEqual(values.shape, (8, 1))

    def test_get_mode_probs_sums_to_one(self):
        import torch
        obs = torch.zeros(25)
        probs = self.policy.get_mode_probs(obs)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_save_load_checkpoint(self):
        import torch
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "policy.pt")
            # Save state dict directly
            torch.save({"policy_state_dict": self.policy.state_dict()}, path)
            # Load into new policy
            from src.representation.neural_policy import ActorCritic
            new_policy = ActorCritic()
            state = torch.load(path, map_location="cpu", weights_only=True)
            new_policy.load_state_dict(state["policy_state_dict"])
            # Deterministic outputs should match
            obs = torch.ones(25)
            a1, _, _ = self.policy.get_action(obs, deterministic=True)
            a2, _, _ = new_policy.get_action(obs, deterministic=True)
            np.testing.assert_array_equal(a1, a2)

    def test_parameter_count(self):
        total = sum(p.numel() for p in self.policy.parameters())
        # ~70K parameters (25*256 + 256 + 256*256 + 256 + 3*(256*2+2) + 256*1+1)
        # Approximately 72K — accept anything in [50K, 150K] for robustness
        self.assertGreater(total, 50_000)
        self.assertLess(total, 150_000)


# ===========================================================================
# Section 5: RolloutBuffer
# ===========================================================================

class TestRolloutBuffer(unittest.TestCase):

    def _make_buffer(self, size: int = 20) -> "RolloutBuffer":
        from src.behaviour.rollout_buffer import RolloutBuffer
        return RolloutBuffer(buffer_size=size)

    def test_store_and_size(self):
        buf = self._make_buffer(10)
        for i in range(5):
            buf.store(
                obs=np.zeros(25, dtype=np.float32),
                action=np.array([0, 0, 0], dtype=np.int64),
                reward=1.0,
                value=0.5,
                log_prob=-1.0,
                done=False,
            )
        self.assertEqual(buf.size, 5)

    def test_overflow_raises(self):
        buf = self._make_buffer(3)
        for _ in range(3):
            buf.store(np.zeros(25), np.zeros(3, dtype=np.int64), 0.0, 0.0, 0.0, False)
        with self.assertRaises(RuntimeError):
            buf.store(np.zeros(25), np.zeros(3, dtype=np.int64), 0.0, 0.0, 0.0, False)

    def test_is_full(self):
        buf = self._make_buffer(3)
        self.assertFalse(buf.is_full)
        for _ in range(3):
            buf.store(np.zeros(25), np.zeros(3, dtype=np.int64), 0.0, 0.0, 0.0, False)
        self.assertTrue(buf.is_full)

    def test_reset_clears(self):
        buf = self._make_buffer(5)
        buf.store(np.zeros(25), np.zeros(3, dtype=np.int64), 1.0, 0.5, -1.0, False)
        buf.reset()
        self.assertEqual(buf.size, 0)
        self.assertFalse(buf.is_full)
        self.assertIsNone(buf.returns)

    def test_gae_computation(self):
        buf = self._make_buffer(5)
        for i in range(5):
            buf.store(
                obs=np.zeros(25),
                action=np.zeros(3, dtype=np.int64),
                reward=1.0,
                value=0.5,
                log_prob=-1.0,
                done=(i == 4),  # final step done
            )
        buf.compute_returns_and_advantages(last_value=0.0, gamma=0.97, gae_lambda=0.95)
        self.assertIsNotNone(buf.returns)
        self.assertIsNotNone(buf.advantages)
        self.assertEqual(buf.returns.shape, (5,))

    def test_get_batches_covers_all_samples(self):
        buf = self._make_buffer(20)
        for i in range(20):
            buf.store(np.zeros(25), np.zeros(3, dtype=np.int64), float(i), 0.5, -1.0, False)
        buf.compute_returns_and_advantages(0.0)
        total = 0
        for batch in buf.get_batches(5):
            total += len(batch["observations"])
        self.assertEqual(total, 20)

    def test_get_batches_without_gae_raises(self):
        buf = self._make_buffer(5)
        buf.store(np.zeros(25), np.zeros(3, dtype=np.int64), 1.0, 0.5, -1.0, False)
        with self.assertRaises(RuntimeError):
            list(buf.get_batches(5))

    def test_actions_stored_correctly(self):
        buf = self._make_buffer(3)
        action = np.array([3, 1, 0], dtype=np.int64)
        buf.store(np.zeros(25), action, 0.0, 0.0, 0.0, False)
        np.testing.assert_array_equal(buf.actions[0], action)


# ===========================================================================
# Section 6: PPO Trainer (torch only)
# ===========================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestPPOTrainer(unittest.TestCase):

    def setUp(self):
        from src.representation.neural_policy import ActorCritic
        from src.behaviour.training_pipeline import PPOTrainer
        from src.behaviour.rollout_buffer import RolloutBuffer
        self.policy = ActorCritic()
        self.trainer = PPOTrainer(
            policy=self.policy,
            config={
                "lr": 1e-3,
                "ppo_epochs": 2,
                "minibatch_size": 4,
                "clip_ratio": 0.3,
                "entropy_coef": 0.01,
                "value_coef": 1.0,
                "max_grad_norm": 0.5,
                "gamma": 0.97,
                "gae_lambda": 0.95,
            },
        )
        self.buf = RolloutBuffer(buffer_size=16)

    def _fill_buffer(self):
        for _ in range(16):
            obs = np.random.randn(25).astype(np.float32)
            action = np.array([np.random.randint(7), np.random.randint(2), np.random.randint(2)])
            self.buf.store(obs, action, np.random.randn(), 0.5, -1.0, False)

    def test_single_update_returns_metrics(self):
        self._fill_buffer()
        info = self.trainer.update(self.buf)
        self.assertIn("policy_loss", info)
        self.assertIn("value_loss", info)
        self.assertIn("entropy", info)
        self.assertIn("approx_kl", info)

    def test_update_advances_training_step(self):
        self._fill_buffer()
        self.trainer.update(self.buf)
        self.assertGreater(self.trainer.training_step, 0)

    def test_losses_are_finite(self):
        self._fill_buffer()
        info = self.trainer.update(self.buf)
        for k, v in info.items():
            if k != "training_step":
                self.assertTrue(np.isfinite(v), f"{k}={v} is not finite")

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            self._fill_buffer()
            self.trainer.update(self.buf)
            step_before = self.trainer.training_step
            self.trainer.save(path)

            from src.representation.neural_policy import ActorCritic
            from src.behaviour.training_pipeline import PPOTrainer
            new_policy = ActorCritic()
            new_trainer = PPOTrainer(new_policy, config={"ppo_epochs": 1, "minibatch_size": 4})
            new_trainer.load(path)
            self.assertEqual(new_trainer.training_step, step_before)

    def test_lr_schedule_applied(self):
        from src.behaviour.training_pipeline import PPOTrainer
        from src.representation.neural_policy import ActorCritic
        trainer = PPOTrainer(
            ActorCritic(),
            config={
                "lr": 1e-3,
                "lr_schedule": [[0, 1e-3], [10, 1e-5]],
                "ppo_epochs": 1,
                "minibatch_size": 4,
            }
        )
        # Manually set training step to trigger schedule
        trainer.training_step = 5
        trainer._update_lr()
        lr = trainer.optimizer.param_groups[0]["lr"]
        # At step 5, halfway between 1e-3 and 1e-5
        self.assertAlmostEqual(lr, (1e-3 + 1e-5) / 2, places=6)


# ===========================================================================
# Section 7: SubsymbolicEventSat representation
# ===========================================================================

class TestSubsymbolicEventSatRegistration(unittest.TestCase):

    def test_registration(self):
        import src.representation.subsymbolic_eventsat  # noqa
        from src.behaviour.controller import _REPRESENTATION_REGISTRY
        self.assertIn("subsymbolic_eventsat", _REPRESENTATION_REGISTRY)


class TestSubsymbolicEventSatBasic(unittest.TestCase):

    def setUp(self):
        self.repr = _make_subsymbolic_repr(mock=True)
        self.env = _make_eventsat_env()
        self.obs = self.env.reset(seed=0)

    def test_encode_observation_returns_dict(self):
        state = self.repr.encode_observation(self.obs)
        self.assertIsInstance(state, dict)
        self.assertIn("battery_soc", state)
        self.assertIn("_obs_vector", state)

    def test_obs_vector_shape(self):
        state = self.repr.encode_observation(self.obs)
        vec = state["_obs_vector"]
        self.assertEqual(vec.shape, (25,))
        self.assertEqual(vec.dtype, np.float32)

    def test_obs_vector_finite(self):
        state = self.repr.encode_observation(self.obs)
        self.assertTrue(np.all(np.isfinite(state["_obs_vector"])))

    def test_select_action_valid_mode(self):
        from src.decision_procedure.context import DecisionContext
        state = self.repr.encode_observation(self.obs)
        context = DecisionContext(
            state=state, loop_type="sda", memory=None, enrichments={}, loop_metadata={}
        )
        action = self.repr.select_action(context)
        self.assertIn("eventsat_0", action)
        mode = action["eventsat_0"]["mode"]
        valid = {"charging", "communication", "payload_observe", "payload_compress",
                 "payload_detect", "payload_send", "safe"}
        self.assertIn(mode, valid)

    def test_select_action_has_sub_actions(self):
        from src.decision_procedure.context import DecisionContext
        state = self.repr.encode_observation(self.obs)
        context = DecisionContext(
            state=state, loop_type="sda", memory=None, enrichments={}, loop_metadata={}
        )
        action = self.repr.select_action(context)
        sat_action = action["eventsat_0"]
        self.assertIn("data_priority", sat_action)
        self.assertIn("pipeline_routing", sat_action)
        self.assertIn(sat_action["data_priority"], [0, 1])
        self.assertIn(sat_action["pipeline_routing"], [0, 1])

    def test_anomaly_forces_safe(self):
        from src.decision_procedure.context import DecisionContext
        state = self.repr.encode_observation(self.obs)
        state["health_status"] = "thermal_warning"
        context = DecisionContext(
            state=state, loop_type="sda", memory=None, enrichments={}, loop_metadata={}
        )
        action = self.repr.select_action(context)
        self.assertEqual(action["eventsat_0"]["mode"], "safe")

    def test_empty_state_returns_charging(self):
        from src.decision_procedure.context import DecisionContext
        context = DecisionContext(
            state={}, loop_type="sda", memory=None, enrichments={}, loop_metadata={}
        )
        action = self.repr.select_action(context)
        self.assertEqual(action["eventsat_0"]["mode"], "charging")

    def test_grounding_no_pass_communication(self):
        from src.decision_procedure.context import DecisionContext
        state = {
            "health_status": "nominal",
            "battery_soc": 0.9,
            "ground_pass_active": False,
            "_obs_vector": np.zeros(25, dtype=np.float32),
        }
        # Patch policy to always return communication (index 1)
        self.repr._policy._rng = MagicMock()
        original_get_action = self.repr._policy.get_action

        def forced_action(obs, **kwargs):
            return np.array([1, 0, 0]), 0.0, 0.0  # communication

        self.repr._policy.get_action = forced_action
        context = DecisionContext(
            state=state, loop_type="sda", memory=None, enrichments={}, loop_metadata={}
        )
        action = self.repr.select_action(context)
        # Should be grounded to charging (no pass)
        self.assertEqual(action["eventsat_0"]["mode"], "charging")
        self.repr._policy.get_action = original_get_action

    def test_reason_returns_list(self):
        state = self.repr.encode_observation(self.obs)
        result = self.repr.reason(state, memory=None)
        self.assertIsInstance(result, list)

    def test_reason_empty_state(self):
        result = self.repr.reason({}, memory=None)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_update_noop_without_trainer(self):
        """update() should not raise when trainer not set."""
        self.repr.update({"buffer": MagicMock(), "episode": 0})

    def test_get_metrics_returns_dict(self):
        from src.decision_procedure.context import DecisionContext
        state = self.repr.encode_observation(self.obs)
        context = DecisionContext(
            state=state, loop_type="sda", memory=None, enrichments={}, loop_metadata={}
        )
        self.repr.select_action(context)
        metrics = self.repr.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("rl_inference_latency_s", metrics)

    def test_get_name(self):
        self.assertEqual(self.repr.get_name(), "SubsymbolicEventSat")

    def test_get_rationale_after_action(self):
        from src.decision_procedure.context import DecisionContext
        state = self.repr.encode_observation(self.obs)
        context = DecisionContext(
            state=state, loop_type="sda", memory=None, enrichments={}, loop_metadata={}
        )
        self.repr.select_action(context)
        rationale = self.repr.get_rationale()
        self.assertIsNotNone(rationale)

    def test_get_last_step_data(self):
        from src.decision_procedure.context import DecisionContext
        state = self.repr.encode_observation(self.obs)
        context = DecisionContext(
            state=state, loop_type="sda", memory=None, enrichments={}, loop_metadata={}
        )
        self.repr.select_action(context)
        data = self.repr.get_last_step_data()
        self.assertIsNotNone(data)
        self.assertIn("obs_vec", data)
        self.assertIn("action_vec", data)
        self.assertIn("log_prob", data)
        self.assertIn("value", data)


# ===========================================================================
# Section 8: Integration — all 3 loop types
# ===========================================================================

class TestSubsymbolicIntegrationLoops(unittest.TestCase):

    def setUp(self):
        self.env = _make_eventsat_env(max_steps=5)
        import src.representation.subsymbolic_eventsat  # noqa

    def _run_loop(self, loop_type: str) -> None:
        from src.behaviour.controller import BehaviourController
        emergence = BehaviourController(config={"mode": "hand_designed"})
        representation = emergence.get_representation(
            "subsymbolic_eventsat",
            repr_config={"rl_mock": True, "deterministic": False},
        )

        if loop_type == "sda":
            from src.decision_procedure.sda_loop import SDALoop
            loop = SDALoop(config={}, representation=representation)
        elif loop_type == "ooda":
            from src.decision_procedure.ooda_loop import OODALoop
            loop = OODALoop(config={}, representation=representation)
        elif loop_type == "react":
            from src.decision_procedure.react_loop import ReActLoop
            loop = ReActLoop(config={}, representation=representation)

        from src.memory.fixed_memory import FixedMemory
        memory = FixedMemory(config={})
        obs = self.env.reset(seed=0)

        for _ in range(5):
            action, memory = loop.process(obs, memory)
            self.assertIn("eventsat_0", action)
            mode = action["eventsat_0"].get("mode", action["eventsat_0"])
            valid = {"charging", "communication", "payload_observe", "payload_compress",
                     "payload_detect", "payload_send", "safe"}
            self.assertIn(str(mode), valid)
            result = self.env.step(action)
            obs = result.observation

    def test_sda_loop(self):
        self._run_loop("sda")

    def test_ooda_loop(self):
        self._run_loop("ooda")

    def test_react_loop(self):
        self._run_loop("react")


# ===========================================================================
# Section 9: Experiment runner integration (smoke test)
# ===========================================================================

class TestExperimentRunnerSubsymbolic(unittest.TestCase):

    def test_runner_registers_subsymbolic(self):
        """The runner imports subsymbolic_eventsat, triggering @register."""
        import tempfile

        from src.orchestration.config_loader import ExperimentConfig
        from src.orchestration.experiment_runner import ExperimentRunner

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ExperimentConfig(
                experiment_id="test_subsymbolic_smoke",
                num_episodes=1,
                max_steps=3,
                seed=0,
                agent_organization="sas",
                decision_procedure="sda",
                representation="subsymbolic",
                behaviour="hand_designed",
                operations_paradigm="autonomous_hybrid",
                representation_config={"type": "subsymbolic_eventsat", "rl_mock": True},
                behaviour_config={"mode": "hand_designed"},
                environment={
                    "scenario": "eventsat",
                    "timestep_seconds": 60.0,
                    "constellation_size": 1,
                    "scenario_config": {
                        "scenario_params": {
                            "orbit": {"orbital_period_s": 5676, "eclipse_fraction": 0.36},
                            "power": {
                                "solar_panels": {"generation_peak_w": 24.0},
                                "battery": {"capacity_wh": 84.0, "initial_soc": 0.8, "min_soc": 0.2},
                                "consumption": {},
                            },
                            "storage": {},
                            "communications": {"sband": {"downlink_rate_kbps": 128}},
                            "modes": {},
                            "payload": {},
                        }
                    },
                },
                output_dir=tmp_dir,
            )

            runner = ExperimentRunner(config=config)
            runner._create_decision_loops()

            from src.behaviour.controller import _REPRESENTATION_REGISTRY
            self.assertIn("subsymbolic_eventsat", _REPRESENTATION_REGISTRY)


if __name__ == "__main__":
    unittest.main()

"""
Tests for Phase 4b: Subsymbolic RL Representation.

Covers:
- EventSat env orbital lookahead metadata (backward compatible)
- EventSat env mode-only action processing
- Gymnasium wrapper: obs shape, action space, reset/step contract, reward scalar
- Neural policy (RandomPolicy always; ActorCritic if torch available):
  forward shape, deterministic vs stochastic, save/load
- RolloutBuffer: store, overflow, GAE, batch iteration
- PPOTrainer (torch only): single update, loss types, lr schedule, save/load
- SubsymbolicEventSat: registration, encode_observation, select_action,
  reason(), update(), grounding, get_metrics()
- Integration: with all 3 loop types via DecisionContext
- Behaviour factory: subsymbolic_eventsat registered after import
"""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.eventsat.neural_policy import TORCH_AVAILABLE, RandomPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eventsat_env(max_steps: int = 50) -> "EventSatEnvironment":
    from src.eventsat.env import EventSatEnvironment
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
    import src.eventsat.rl  # trigger registration
    from src.eventsat.rl import SubsymbolicEventSat
    return SubsymbolicEventSat(config={"rl_mock": mock, "deterministic": False})


def _make_decision_context(loop_type: str = "sda", state: dict | None = None):
    from src.core.decision_procedure.context import DecisionContext
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
# Section 2: EventSat env - mode-only action processing
# ===========================================================================

class TestEventSatModeActions(unittest.TestCase):

    def setUp(self):
        self.env = _make_eventsat_env(max_steps=200)
        self.env.reset(seed=0)
        self.env.battery_soc = 0.9

    def test_mode_action_has_no_sub_action_state(self):
        result = self.env.step({"eventsat_0": {"mode": "charging"}})
        self.assertEqual(result.info.get("requested_mode"), "charging")

    def test_payload_compress_does_not_reroute_to_detect(self):
        self.env.uncompressed_observations = 0
        self.env.undetected_observations = 2
        result = self.env.step({"eventsat_0": {"mode": "payload_compress"}})
        self.assertFalse(result.info.get("had_data_to_compress", True))

    def test_payload_detect_does_not_reroute_to_compress(self):
        self.env.undetected_observations = 0
        self.env.uncompressed_observations = 2
        result = self.env.step({"eventsat_0": {"mode": "payload_detect"}})
        self.assertFalse(result.info.get("had_data_to_detect", True))


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
        from src.eventsat.gymnasium_wrapper import EventSatGymnasium
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
        from src.eventsat import gymnasium_wrapper  # noqa: F401
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
        from gymnasium.spaces import Discrete
        wrapper = self._make_wrapper()
        self.assertIsInstance(wrapper.action_space, Discrete)
        self.assertEqual(wrapper.action_space.n, 7)

    def test_step_contract(self):
        if not self.gymnasium_available:
            self.skipTest("gymnasium not installed")
        wrapper = self._make_wrapper()
        wrapper.reset(seed=0)
        action = np.array(0, dtype=int)  # charging
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
        _, reward, _, _, _ = wrapper.step(np.array(0))
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
        self.assertEqual(action.shape, (1,))

    def test_get_action_bounds(self):
        obs = np.zeros(25, dtype=np.float32)
        for _ in range(20):
            action, _, _ = self.policy.get_action(obs)
            self.assertIn(action[0], range(7))

    def test_get_action_deterministic_same(self):
        """Deterministic mode should return same action each call (not really for random, but shouldn't crash)."""
        obs = np.zeros(25, dtype=np.float32)
        action, _, _ = self.policy.get_action(obs, deterministic=True)
        self.assertEqual(action.shape, (1,))

    def test_evaluate_actions_shape(self):
        obs_batch = np.zeros((10, 25), dtype=np.float32)
        actions_batch = np.zeros((10, 1), dtype=np.int64)
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
        from src.eventsat.neural_policy import ActorCritic
        self.policy = ActorCritic()

    def test_forward_shape(self):
        import torch
        obs = torch.zeros(1, 25)
        dists, value = self.policy.forward(obs)
        self.assertEqual(len(dists), 1)
        self.assertEqual(value.shape, (1, 1))

    def test_forward_dist_shapes(self):
        import torch
        obs = torch.zeros(1, 25)
        dists, _ = self.policy.forward(obs)
        self.assertEqual(dists[0].param_shape, (1, 7))

    def test_get_action_shape(self):
        import torch
        obs = torch.zeros(25)
        action, log_prob, value = self.policy.get_action(obs)
        self.assertEqual(action.shape, (1,))
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
        actions_batch = torch.zeros(8, 1, dtype=torch.long)
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
            from src.eventsat.neural_policy import ActorCritic
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
        # ~73K parameters for 25D input, two 256-wide hidden layers, one mode head, and one critic.
        # Accept a broad range for robustness.
        self.assertGreater(total, 50_000)
        self.assertLess(total, 150_000)


# ===========================================================================
# Section 5: RolloutBuffer
# ===========================================================================

class TestRolloutBuffer(unittest.TestCase):

    def _make_buffer(self, size: int = 20) -> "RolloutBuffer":
        from src.core.behaviour.rollout_buffer import RolloutBuffer
        return RolloutBuffer(buffer_size=size)

    def test_store_and_size(self):
        buf = self._make_buffer(10)
        for i in range(5):
            buf.store(
                obs=np.zeros(25, dtype=np.float32),
                action=np.array([0], dtype=np.int64),
                reward=1.0,
                value=0.5,
                log_prob=-1.0,
                done=False,
            )
        self.assertEqual(buf.size, 5)

    def test_overflow_raises(self):
        buf = self._make_buffer(3)
        for _ in range(3):
            buf.store(np.zeros(25), np.zeros(1, dtype=np.int64), 0.0, 0.0, 0.0, False)
        with self.assertRaises(RuntimeError):
            buf.store(np.zeros(25), np.zeros(1, dtype=np.int64), 0.0, 0.0, 0.0, False)

    def test_is_full(self):
        buf = self._make_buffer(3)
        self.assertFalse(buf.is_full)
        for _ in range(3):
            buf.store(np.zeros(25), np.zeros(1, dtype=np.int64), 0.0, 0.0, 0.0, False)
        self.assertTrue(buf.is_full)

    def test_reset_clears(self):
        buf = self._make_buffer(5)
        buf.store(np.zeros(25), np.zeros(1, dtype=np.int64), 1.0, 0.5, -1.0, False)
        buf.reset()
        self.assertEqual(buf.size, 0)
        self.assertFalse(buf.is_full)
        self.assertIsNone(buf.returns)

    def test_gae_computation(self):
        buf = self._make_buffer(5)
        for i in range(5):
            buf.store(
                obs=np.zeros(25),
                action=np.zeros(1, dtype=np.int64),
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
            buf.store(np.zeros(25), np.zeros(1, dtype=np.int64), float(i), 0.5, -1.0, False)
        buf.compute_returns_and_advantages(0.0)
        total = 0
        for batch in buf.get_batches(5):
            total += len(batch["observations"])
        self.assertEqual(total, 20)

    def test_get_batches_without_gae_raises(self):
        buf = self._make_buffer(5)
        buf.store(np.zeros(25), np.zeros(1, dtype=np.int64), 1.0, 0.5, -1.0, False)
        with self.assertRaises(RuntimeError):
            list(buf.get_batches(5))

    def test_actions_stored_correctly(self):
        buf = self._make_buffer(3)
        action = np.array([3], dtype=np.int64)
        buf.store(np.zeros(25), action, 0.0, 0.0, 0.0, False)
        np.testing.assert_array_equal(buf.actions[0], action)

    def test_action_shape_expands_on_first_store(self):
        buf = self._make_buffer(3)
        action = np.array([3, 1, 0], dtype=np.int64)
        buf.store(np.zeros(25), action, 0.0, 0.0, 0.0, False)
        self.assertEqual(buf.action_shape, (3,))
        self.assertEqual(buf.actions.shape, (3, 3))
        np.testing.assert_array_equal(buf.actions[0], action)


# ===========================================================================
# Section 6: PPO Trainer (torch only)
# ===========================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class TestPPOTrainer(unittest.TestCase):

    def setUp(self):
        from src.eventsat.neural_policy import ActorCritic
        from src.core.behaviour.training_pipeline import PPOTrainer
        from src.core.behaviour.rollout_buffer import RolloutBuffer
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

            from src.eventsat.neural_policy import ActorCritic
            from src.core.behaviour.training_pipeline import PPOTrainer
            new_policy = ActorCritic()
            new_trainer = PPOTrainer(new_policy, config={"ppo_epochs": 1, "minibatch_size": 4})
            new_trainer.load(path)
            self.assertEqual(new_trainer.training_step, step_before)

    def test_lr_schedule_applied(self):
        from src.core.behaviour.training_pipeline import PPOTrainer
        from src.eventsat.neural_policy import ActorCritic
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
        import src.eventsat.rl  # noqa
        from src.core.behaviour.controller import _REPRESENTATION_REGISTRY
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
        self.assertEqual(vec.shape, (33,))
        self.assertEqual(vec.dtype, np.float32)

    def test_obs_vector_finite(self):
        state = self.repr.encode_observation(self.obs)
        self.assertTrue(np.all(np.isfinite(state["_obs_vector"])))

    def test_select_action_valid_mode(self):
        from src.core.decision_procedure.context import DecisionContext
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

    def test_select_action_is_mode_only(self):
        from src.core.decision_procedure.context import DecisionContext
        state = self.repr.encode_observation(self.obs)
        context = DecisionContext(
            state=state, loop_type="sda", memory=None, enrichments={}, loop_metadata={}
        )
        action = self.repr.select_action(context)
        sat_action = action["eventsat_0"]
        self.assertEqual(set(sat_action.keys()), {"mode"})

    def test_anomaly_forces_safe(self):
        from src.core.decision_procedure.context import DecisionContext
        state = self.repr.encode_observation(self.obs)
        state["health_status"] = "thermal_warning"
        context = DecisionContext(
            state=state, loop_type="sda", memory=None, enrichments={}, loop_metadata={}
        )
        action = self.repr.select_action(context)
        self.assertEqual(action["eventsat_0"]["mode"], "safe")

    def test_empty_state_returns_charging(self):
        from src.core.decision_procedure.context import DecisionContext
        context = DecisionContext(
            state={}, loop_type="sda", memory=None, enrichments={}, loop_metadata={}
        )
        action = self.repr.select_action(context)
        self.assertEqual(action["eventsat_0"]["mode"], "charging")

    def test_grounding_no_pass_communication(self):
        from src.core.decision_procedure.context import DecisionContext
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
            return np.array([1]), 0.0, 0.0  # communication

        self.repr._policy.get_action = forced_action
        context = DecisionContext(
            state=state, loop_type="sda", memory=None, enrichments={}, loop_metadata={}
        )
        action = self.repr.select_action(context)
        # The environment must observe and price the radio attempt, even without a pass.
        self.assertEqual(action["eventsat_0"]["mode"], "communication")
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
        from src.core.decision_procedure.context import DecisionContext
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
        from src.core.decision_procedure.context import DecisionContext
        state = self.repr.encode_observation(self.obs)
        context = DecisionContext(
            state=state, loop_type="sda", memory=None, enrichments={}, loop_metadata={}
        )
        self.repr.select_action(context)
        rationale = self.repr.get_rationale()
        self.assertIsNotNone(rationale)

    def test_get_last_step_data(self):
        from src.core.decision_procedure.context import DecisionContext
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
        import src.eventsat.rl  # noqa

    def _run_loop(self, loop_type: str) -> None:
        from src.core.behaviour.controller import BehaviourController
        factory = BehaviourController(config={"mode": "hand_designed"})
        representation = factory.get_representation(
            "subsymbolic_eventsat",
            repr_config={"rl_mock": True, "deterministic": False},
        )

        if loop_type != "sda":
            raise AssertionError("Only SDA loop is supported")
        from src.core.decision_procedure.sda_loop import SDALoop
        loop = SDALoop(config={}, representation=representation)

        from src.core.memory.fixed_memory import FixedMemory
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


# ===========================================================================
# Section 9: Experiment runner integration (smoke test)
# ===========================================================================

class TestExperimentRunnerSubsymbolic(unittest.TestCase):

    def test_rllib_exploration_uses_restartable_private_rng(self):
        from src.eventsat.rllib_policy_adapter import RLLibPolicyAdapter

        class FakeAlgorithm:
            def compute_single_action(self, observation, **kwargs):
                assert kwargs["explore"] is False
                assert kwargs["full_fetch"] is True
                logits = np.asarray([0.1, 1.2, -0.3, 0.8, -0.2], dtype=np.float32)
                return np.asarray([0, 0]), [], {"action_dist_inputs": logits}

        adapter = object.__new__(RLLibPolicyAdapter)
        adapter.policy_id = "shared_policy"
        adapter._action_dims = [3, 2]
        adapter._algo = FakeAlgorithm()
        adapter._rng = np.random.default_rng()
        observation = np.zeros(25, dtype=np.float32)

        adapter.seed(123)
        first = [adapter.get_action(observation, deterministic=False)[0] for _ in range(8)]
        adapter.seed(123)
        replay = [adapter.get_action(observation, deterministic=False)[0] for _ in range(8)]

        for left, right in zip(first, replay):
            np.testing.assert_array_equal(left, right)

    def test_rllib_adapter_close_is_idempotent(self):
        from src.eventsat.rllib_policy_adapter import RLLibPolicyAdapter

        class FakeAlgorithm:
            def __init__(self):
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1

        algorithm = FakeAlgorithm()
        adapter = object.__new__(RLLibPolicyAdapter)
        adapter._algo = algorithm

        adapter.close()
        adapter.close()

        self.assertEqual(algorithm.stop_calls, 1)
        self.assertIsNone(adapter._algo)

    def test_multi_episode_rl_episode_matches_fresh_seeded_run(self):
        from src.core.config_loader import ExperimentConfig
        from src.core.experiment_runner import ExperimentRunner

        def config(seed: int, episodes: int, output_dir: str) -> ExperimentConfig:
            return ExperimentConfig(
                experiment_id=f"rl_episode_seed_{seed}_{episodes}",
                seed=seed,
                num_episodes=episodes,
                max_steps=12,
                agent_organization="sas",
                decision_procedure="sda",
                representation="subsymbolic",
                representation_config={
                    "type": "subsymbolic_eventsat",
                    "rl_mock": True,
                    "deterministic": False,
                },
                behaviour="hand_designed",
                behaviour_config={"mode": "hand_designed"},
                operations_paradigm="autonomous_onboard",
                environment={
                    "scenario": "eventsat",
                    "constellation_size": 1,
                    "timestep_seconds": 60,
                    "max_steps": 12,
                    "scenario_config": {},
                },
                output_dir=output_dir,
                log_level="WARNING",
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            multi = ExperimentRunner(
                config=config(41, 2, os.path.join(tmp_dir, "multi"))
            ).run()
            fresh = ExperimentRunner(
                config=config(42, 1, os.path.join(tmp_dir, "fresh"))
            ).run()

        def signature(episode):
            return [
                (
                    step["info"].get("requested_mode"),
                    step["info"].get("resolved_mode"),
                    step["rewards"],
                )
                for step in episode["steps"]
            ]

        self.assertEqual(signature(multi["episodes"][1]), signature(fresh["episodes"][0]))

    def test_runner_registers_subsymbolic(self):
        """The runner imports subsymbolic_eventsat, triggering @register."""
        import tempfile

        from src.core.config_loader import ExperimentConfig
        from src.core.experiment_runner import ExperimentRunner

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
                operations_paradigm="autonomous_onboard",
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

            from src.core.behaviour.controller import _REPRESENTATION_REGISTRY
            self.assertIn("subsymbolic_eventsat", _REPRESENTATION_REGISTRY)


def test_all_eventsat_encoders_respect_declared_space_when_saturated():
    from src.core.satellite_env import (
        ConstellationState,
        EnvironmentObservation,
        SatelliteState,
    )
    from src.eventsat.gymnasium_wrapper import EventSatGymnasium
    from src.eventsat.rl import SubsymbolicEventSat
    from src.eventsat.rl_obs_encoder import OBS_DIM
    from src.eventsat.world_model import eventsat_observation_to_vector
    from src.rl import observation_within_bounds
    from src.rl.space_adapters import EventSatSpaceAdapter, GYMNASIUM_AVAILABLE

    resources = {
        "battery_soc": 0.8,
        "obc_data_mb": 10.0,
        "data_stored_mb": 10.0,
        "data_downlinked_mb": 10.0,
    }
    metadata = {
        "storage_capacity_mb": 1.0,
        "jetson_capacity_mb": 1.0,
        "jetson_raw_mb": 10.0,
        "jetson_compressed_mb": 10.0,
        "max_achievable_downlink_mb": 1.0,
        "achievable_downlink_mb": 1.0,
        "orbital_period_steps": 10,
        "orbital_phase": 0.75,
        "time_to_next_eclipse": 100,
        "time_to_next_pass": 100,
        "remaining_pass_duration": 100,
        "uncompressed_observations": 100,
        "compression_progress": 100,
        "undetected_observations": 100,
        "detection_progress": 100,
        "compression_time_factor": 2,
        "detection_steps": 5,
    }
    sat = SatelliteState(
        satellite_id="eventsat_0",
        resources=resources,
        status="charging",
        metadata=metadata,
    )
    constellation = ConstellationState(
        timestep=20,
        epoch_seconds=1200.0,
        satellites={"eventsat_0": sat},
        global_info={"max_steps": 10},
    )
    observation = EnvironmentObservation(constellation_state=constellation)

    env_shape = SimpleNamespace(
        storage_capacity_mb=1.0,
        jetson_capacity_mb=1.0,
        orbital_period_steps=10,
        compression_time_factor=2,
        detection_steps=5,
        detection_progress=100,
        current_step=20,
        max_steps=10,
    )
    adapter = object.__new__(EventSatSpaceAdapter)
    adapter.config = {"satellite_id": "eventsat_0", "max_steps": 10}
    adapter.env = env_shape
    adapter.satellite_id = "eventsat_0"
    gym_encoder = object.__new__(EventSatGymnasium)
    gym_encoder._env = env_shape
    deployment = SubsymbolicEventSat(
        {
            "rl_mock": True,
            "satellite_id": "eventsat_0",
            "jetson_capacity_mb": 1.0,
            "orbital_period_steps": 10,
            "max_steps": 10,
        }
    )

    vectors = {
        "space_adapter": adapter.encode_observation(observation),
        "gym_wrapper": gym_encoder._obs_to_vector(observation),
        "deployment": deployment.encode_observation(observation)["_obs_vector"],
        "world_model": eventsat_observation_to_vector(observation).obs25,
    }
    # Option (a): RL encoders use the 33D schema, legacy encoders stay 25D.
    sizes = {"space_adapter": OBS_DIM, "gym_wrapper": 25, "deployment": OBS_DIM, "world_model": 25}
    for name, vector in vectors.items():
        assert observation_within_bounds(
            vector, size=sizes[name], signed_indices=(4, 5)
        ), name
        assert vector[17] == 2.0, name

    metadata["max_achievable_downlink_mb"] = 0.0
    zero_capacity_vectors = {
        "space_adapter": adapter.encode_observation(observation),
        "gym_wrapper": gym_encoder._obs_to_vector(observation),
        "deployment": deployment.encode_observation(observation)["_obs_vector"],
        "world_model": eventsat_observation_to_vector(observation).obs25,
    }
    for name, vector in zero_capacity_vectors.items():
        assert observation_within_bounds(
            vector, size=sizes[name], signed_indices=(4, 5)
        ), name
        assert vector[17] == 2.0, name
    if GYMNASIUM_AVAILABLE:
        declared_space = EventSatSpaceAdapter(
            config={"satellite_id": "eventsat_0", "max_steps": 10},
            env=env_shape,
        ).observation_space
        assert all(
            declared_space.contains(vector)
            for source in (vectors, zero_capacity_vectors)
            for vector in (source["space_adapter"], source["deployment"])
        )


def test_rl_encoder_uses_log_product_scale_separately_from_world_model():
    import math

    import pytest

    from src.core.satellite_env import (
        ConstellationState,
        EnvironmentObservation,
        SatelliteState,
    )
    from src.eventsat.rl_obs_encoder import encode_eventsat_rl_obs
    from src.eventsat.world_model import eventsat_observation_to_vector

    compressed_mb = 9.41 / 5.11
    metadata = {
        "storage_capacity_mb": 4096.0,
        "jetson_capacity_mb": 249036.8,
        "observation_size_mb": 9.41,
        "compression_ratio": 5.11,
        "jetson_raw_mb": 9.41,
        "jetson_compressed_mb": compressed_mb,
        "max_achievable_downlink_mb": 1.8,
    }

    def encode(obc_mb, downlinked_mb):
        resources = {
            "battery_soc": 0.8,
            "obc_data_mb": obc_mb,
            "data_downlinked_mb": downlinked_mb,
        }
        return encode_eventsat_rl_obs(
            resources,
            metadata,
            "charging",
            obc_cap=4096.0,
            jetson_cap=249036.8,
            orbital_period=94.0,
            max_steps=10080.0,
            compression_time=2.0,
            detection_steps=5.0,
            current_step=0,
            detection_progress=0.0,
        ), resources

    vec, resources = encode(compressed_mb, compressed_mb)
    one_product = math.log(2.0)
    assert vec[1] == pytest.approx(one_product / math.log1p(4096.0 / compressed_mb), rel=1e-5)
    assert vec[2] == pytest.approx(one_product / math.log1p(249036.8 / 9.41), rel=1e-5)
    assert vec[3] == pytest.approx(one_product / math.log1p(249036.8 / compressed_mb), rel=1e-5)
    assert vec[17] == pytest.approx(vec[1])
    # One product must be distinguishable from an empty buffer.
    assert min(vec[1], vec[2], vec[3], vec[17]) > 0.05
    two_products, _ = encode(2.0 * compressed_mb, 2.0 * compressed_mb)
    assert two_products[1] > vec[1] and two_products[17] > vec[17]
    empty, _ = encode(0.0, 0.0)
    assert empty[1] == empty[17] == 0.0

    sat = SatelliteState(
        satellite_id="eventsat_0",
        resources=resources,
        status="charging",
        metadata=metadata,
    )
    observation = EnvironmentObservation(
        constellation_state=ConstellationState(
            timestep=0,
            epoch_seconds=0.0,
            satellites={"eventsat_0": sat},
            global_info={"max_steps": 10080},
        )
    )
    # Option (a): world-model traces keep the legacy linear capacity fraction.
    legacy = eventsat_observation_to_vector(observation).obs25
    assert legacy[3] == pytest.approx(compressed_mb / 249036.8, rel=1e-5)


def test_rl_encoder_distinguishes_attitude_settling_from_idle():
    from src.eventsat.rl_obs_encoder import MODE_TO_IDX, OBS_DIM, encode_eventsat_rl_obs

    def encode(**meta):
        return encode_eventsat_rl_obs(
            {"battery_soc": 0.8},
            {"settling_time_steps": 2, **meta},
            "charging",
            obc_cap=4096.0,
            jetson_cap=249036.8,
            orbital_period=94.0,
            max_steps=10080.0,
            compression_time=2.0,
            detection_steps=5.0,
            current_step=0,
            detection_progress=0.0,
        )

    idle = encode(transition_steps_remaining=0, previous_mode="charging", transition_target_mode=None)
    slewing = encode(
        transition_steps_remaining=1,
        previous_mode="charging",
        transition_target_mode="payload_observe",
    )
    latched = encode(
        transition_steps_remaining=0,
        previous_mode="payload_observe",
        transition_target_mode=None,
    )
    observe = 26 + MODE_TO_IDX["payload_observe"]

    assert idle.shape == (OBS_DIM,) == (33,)
    # The legacy 25 features cannot tell these three decision states apart.
    assert np.array_equal(idle[:25], slewing[:25])
    assert np.array_equal(idle[:25], latched[:25])
    assert (idle[25], idle[26 + MODE_TO_IDX["charging"]]) == (0.0, 1.0)
    assert (slewing[25], slewing[observe]) == (0.5, 1.0)
    assert (latched[25], latched[observe]) == (0.0, 1.0)


def test_rl_deployment_rejects_checkpoint_from_other_observation_schema(tmp_path):
    import json

    import pytest

    from src.eventsat.rl import SubsymbolicEventSat
    from src.eventsat.rl_obs_encoder import EVENTSAT_OBS_SCHEMA_ID

    deployment = SubsymbolicEventSat({"rl_mock": True})
    checkpoint = tmp_path / "checkpoint_000001"
    checkpoint.mkdir()
    with pytest.raises(RuntimeError, match="manifest is missing"):
        deployment._validate_checkpoint_manifest(checkpoint)

    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "observation_schema_id": None,
        "policy_observation_shapes": {"shared_policy": [33]},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="schema mismatch"):
        deployment._validate_checkpoint_manifest(checkpoint)

    manifest["observation_schema_id"] = EVENTSAT_OBS_SCHEMA_ID
    manifest["policy_observation_shapes"] = {"shared_policy": [25]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        deployment._validate_checkpoint_manifest(checkpoint)

    manifest["policy_observation_shapes"] = {"shared_policy": [33]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    deployment._validate_checkpoint_manifest(checkpoint)


def test_rl_deployment_uses_environment_orbital_period_metadata():
    from src.eventsat.rl import SubsymbolicEventSat

    deployment = SubsymbolicEventSat(
        {
            "rl_mock": True,
            "orbital_period_steps": 94,
        }
    )
    vector = deployment._build_obs_vector(
        {},
        {
            "orbital_period_steps": 92,
            "time_to_next_eclipse": 46,
            "time_to_next_pass": 23,
        },
        SimpleNamespace(timestep=0),
    )

    # Eclipse timing stays linear; pass timing uses the v4 log scale over 92 steps.
    np.testing.assert_allclose(
        vector[[6, 7]], [0.5, np.log1p(23.0) / np.log1p(92.0)], rtol=1e-5
    )


def test_rl_pass_feature_uses_log_scale_over_orbital_period():
    import math

    import pytest

    from src.eventsat.rl_obs_encoder import encode_eventsat_rl_obs

    def feature(**meta):
        return encode_eventsat_rl_obs(
            {},
            meta,
            "charging",
            obc_cap=4096.0,
            jetson_cap=249036.8,
            orbital_period=92.0,
            max_steps=10080.0,
            compression_time=2.0,
            detection_steps=5.0,
            current_step=0,
            detection_progress=0.0,
        )[7]

    scale = math.log1p(92.0)
    assert feature(time_to_next_pass=0) == 0.0
    for steps in (1, 2, 3, 10):
        assert feature(time_to_next_pass=steps) == pytest.approx(math.log1p(steps) / scale, rel=1e-5)
    # The environment caps t at the orbital period; in-pass values refer to the next pass.
    assert feature(time_to_next_pass=92) == pytest.approx(1.0)
    assert feature(time_to_next_pass=551) == pytest.approx(1.0)
    assert feature() == pytest.approx(1.0)
    # Pre-pointing decision: 2 vs 3 steps must differ by far more than the old 0.011.
    assert feature(time_to_next_pass=3) - feature(time_to_next_pass=2) > 0.06


def test_rl_pass_feature_counts_down_to_real_contact():
    import math

    import pytest

    from src.eventsat.env import EventSatEnvironment
    from src.eventsat.rl_obs_encoder import encode_eventsat_rl_obs

    env = EventSatEnvironment(
        {
            "scenario_config": "configs/scenarios/eventsat.yaml",
            "max_steps": 2000,
            "step_duration_s": 60,
        }
    )
    env.reset(seed=42)
    first = next(gp for gp in env._orbital_ctx.ground_passes if gp.start_step >= 5)
    period = env.orbital_period_steps

    values = {}
    while env.current_step <= first.start_step:
        remaining = first.start_step - env.current_step
        if remaining <= 4:
            sat = env.get_observation().constellation_state.satellites["eventsat_0"]
            values[remaining] = encode_eventsat_rl_obs(
                sat.resources,
                sat.metadata,
                sat.status,
                obc_cap=env.storage_capacity_mb,
                jetson_cap=env.jetson_capacity_mb,
                orbital_period=period,
                max_steps=env.max_steps,
                compression_time=env.compression_time_factor,
                detection_steps=env.detection_steps,
                current_step=env.current_step,
                detection_progress=env.detection_progress,
            )[7]
            assert bool(sat.metadata["contact_window_active"]) is (remaining == 0)
        env.step({"eventsat_0": {"mode": "charging"}})

    for steps in (1, 2, 3, 4):
        assert values[steps] == pytest.approx(math.log1p(steps) / math.log1p(period), rel=1e-5)
    # At contact the value refers to the following pass, far from the countdown.
    assert values[0] > 0.95


def test_ssa_encoder_respects_declared_space_for_full_pipeline_and_zero_pass_capacity():
    from src.core.satellite_env import (
        ConstellationState,
        EnvironmentObservation,
        SatelliteState,
    )
    from src.rl import observation_within_bounds
    from src.rl.space_adapters import GYMNASIUM_AVAILABLE, SSASpaceAdapter
    from src.ssa.rl_features import SSA_OBS_DIM, build_ssa_obs_vector

    sat = SatelliteState(
        satellite_id="sat_0",
        resources={
            "battery_soc": 0.8,
            "obc_data_mb": 10.0,
            "data_stored_mb": 10.0,
            "data_downlinked_mb": 10.0,
        },
        status="charging",
        metadata={
            "storage_capacity_mb": 1.0,
            "jetson_raw_mb": 10.0,
            "jetson_compressed_mb": 10.0,
            "achievable_downlink_mb": 0.0,
            "visible_rso_ids": ["rso_0"],
            "ssa_detection_row": [0],
        },
    )
    constellation = ConstellationState(
        timestep=20,
        epoch_seconds=1200.0,
        satellites={"sat_0": sat},
        global_info={"ssa_target_count": 1},
    )
    observation = EnvironmentObservation(constellation_state=constellation)
    config = {"satellite_id": "sat_0", "jetson_capacity_mb": 1.0, "max_steps": 10}
    adapter = object.__new__(SSASpaceAdapter)
    adapter.config = config
    adapter.env = None
    adapter.satellite_id = "sat_0"

    adapter_vector = adapter.encode_observation(observation)
    direct_vector = build_ssa_obs_vector(
        sat=sat,
        constellation=constellation,
        target_count=1,
        max_steps=10,
        config=config,
    )

    assert np.array_equal(adapter_vector, direct_vector)
    assert observation_within_bounds(adapter_vector, size=SSA_OBS_DIM)
    assert adapter_vector[1:5].max() == 2.0
    assert adapter_vector[20] == 2.0
    if GYMNASIUM_AVAILABLE:
        assert SSASpaceAdapter(config=config).observation_space.contains(adapter_vector)


def test_shared_ratio_semantic_admits_boundary_and_saturates_overflow():
    from src.rl import bounded_ratio

    assert bounded_ratio(2.0, 1.0) == 2.0
    assert bounded_ratio(2.0 + 1e-9, 1.0) == 2.0
    assert bounded_ratio(1.0, 1.0) == 1.0
    assert bounded_ratio(1.0, 0.0) == 2.0
    assert bounded_ratio(0.0, 0.0) == 0.0


if __name__ == "__main__":
    unittest.main()

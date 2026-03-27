"""
Rollout Buffer for PPO Training.

Pre-allocated numpy arrays for on-policy trajectory collection.
Implements Generalized Advantage Estimation (GAE) for variance reduction.

Design notes:
- Stores MultiDiscrete([7, 2, 2]) actions as int arrays of shape (T, 3)
- GAE computation follows Schulman et al. 2017 (PPO paper) and
  Oliver et al. EUCASS 2025 (8KDZ5Z53) trajectory collection
- CPU-only (numpy) — GPU transfer happens in PPOTrainer.update()
"""
from __future__ import annotations

from typing import Dict, Generator, Optional

import numpy as np

OBS_DIM = 25
ACTION_SHAPE = (3,)  # MultiDiscrete([7, 2, 2])


class RolloutBuffer:
    """Pre-allocated rollout buffer for PPO with GAE.

    Stores a single rollout of length ``buffer_size`` steps before
    calling ``compute_returns_and_advantages()``.

    Args:
        buffer_size: Maximum steps per rollout (PPO fragment length).
        obs_dim: Observation dimension (default 25).
    """

    def __init__(self, buffer_size: int, obs_dim: int = OBS_DIM) -> None:
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self._pos = 0
        self._full = False

        # Pre-allocated arrays
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, 3), dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # Filled by compute_returns_and_advantages()
        self.returns: Optional[np.ndarray] = None
        self.advantages: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clear the buffer for a new rollout."""
        self._pos = 0
        self._full = False
        self.returns = None
        self.advantages = None

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Store a single transition.

        Args:
            obs: float32 array of shape (obs_dim,)
            action: int array of shape (3,) — MultiDiscrete action
            reward: scalar reward
            value: critic value estimate
            log_prob: joint log-probability of the action
            done: episode done flag
        """
        if self._pos >= self.buffer_size:
            raise RuntimeError(
                f"RolloutBuffer overflow: buffer_size={self.buffer_size}, "
                "call reset() before storing new transitions."
            )
        self.observations[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.values[self._pos] = value
        self.log_probs[self._pos] = log_prob
        self.dones[self._pos] = float(done)
        self._pos += 1
        if self._pos >= self.buffer_size:
            self._full = True

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.97,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Implements GAE-λ from Schulman et al. 2017, Equation 11.
        Oliver et al. EUCASS 2025 uses gamma=0.966-0.98.

        Args:
            last_value: Critic estimate for the state after the last step
                (bootstrap; 0 if the episode terminated).
            gamma: Discount factor (default 0.97 per EUCASS 2025).
            gae_lambda: GAE lambda for bias-variance tradeoff.
        """
        n = self._pos
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = float(self.values[t + 1])

            delta = (
                float(self.rewards[t])
                + gamma * next_value * next_non_terminal
                - float(self.values[t])
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.returns = advantages + self.values[:n]
        self.advantages = advantages

    def get_batches(self, minibatch_size: int) -> Generator[Dict[str, np.ndarray], None, None]:
        """Yield shuffled minibatches for PPO update epochs.

        Requires ``compute_returns_and_advantages()`` to have been called.

        Args:
            minibatch_size: Number of samples per minibatch.

        Yields:
            dict with keys: observations, actions, log_probs, returns,
            advantages, values
        """
        if self.returns is None or self.advantages is None:
            raise RuntimeError("Call compute_returns_and_advantages() before get_batches().")

        n = self._pos
        indices = np.random.permutation(n)
        start = 0
        while start < n:
            batch_idx = indices[start : start + minibatch_size]
            yield {
                "observations": self.observations[batch_idx],
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "returns": self.returns[batch_idx],
                "advantages": self.advantages[batch_idx],
                "values": self.values[batch_idx],
            }
            start += minibatch_size

    @property
    def size(self) -> int:
        """Number of stored transitions."""
        return self._pos

    @property
    def is_full(self) -> bool:
        """True if the buffer has reached buffer_size."""
        return self._full

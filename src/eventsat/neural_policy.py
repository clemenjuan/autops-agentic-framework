"""
Neural Policy for RL-based EventSat Representation.

Implements an Actor-Critic network for 7-mode EventSat actions:
  - Shared MLP trunk: 25 → 256 → 256 (Tanh activations)
  - 1 actor head: → 7 mode logits
  - 1 critic head: → 1 value estimate

Architecture matches Oliver et al. EUCASS 2025 (8KDZ5Z53): [256, 256] hidden layers
with tanh activation, PPO, ~70K parameters, confirmed feasible for Jetson Orin Nano
(50μs inference per the paper).

Log-probability and entropy come from a single categorical mode distribution.

RandomPolicy provides the same interface without torch, for CI/mock mode (rl_mock: true).

Papers:
- Oliver et al. EUCASS 2025 (8KDZ5Z53): network architecture [256,256] tanh
- Hamilton et al. 2025 (GWQ3LK6H): mode action space design
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    Categorical = None  # type: ignore

# Single 7-way mode action head; must match EventSatGymnasium.action_space
ACTION_DIMS = [7]
OBS_DIM = 25
HIDDEN_SIZE = 256


_BaseModule = nn.Module if TORCH_AVAILABLE else object


class ActorCritic(_BaseModule):  # type: ignore[misc]
    """Actor-Critic for the 7-mode EventSat action space.

    Shared trunk -> mode actor head + critic head.
    Architecture: 25→256→256 (Tanh) per Juan Oliver et al. EUCASS 2025.

    Args:
        obs_dim: Observation dimension (default 25).
        hidden_size: Hidden layer width (default 256).
        action_dims: List containing the mode action dimension (default [7]).
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_size: int = HIDDEN_SIZE,
        action_dims: Optional[List[int]] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for ActorCritic. Install with: uv sync --extra rl")
        super().__init__()
        if action_dims is None:
            action_dims = ACTION_DIMS

        # Shared trunk — tanh activation per EUCASS 2025
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # Actor head for the 7 operational modes
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden_size, dim) for dim in action_dims
        ])

        # Critic head
        self.critic_head = nn.Linear(hidden_size, 1)

        self._action_dims = action_dims
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialisation — standard for PPO (stable training)."""
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        for head in self.actor_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: "torch.Tensor") -> Tuple[List["Categorical"], "torch.Tensor"]:
        """Forward pass.

        Args:
            obs: float32 tensor of shape (batch, obs_dim) or (obs_dim,)

        Returns:
            dists: List containing the mode Categorical distribution
            value: float32 tensor of shape (batch, 1) or (1,)
        """
        features = self.trunk(obs)
        dists = [Categorical(logits=head(features)) for head in self.actor_heads]
        value = self.critic_head(features)
        return dists, value

    def get_action(
        self, obs: "torch.Tensor", deterministic: bool = False
    ) -> Tuple[np.ndarray, "torch.Tensor", "torch.Tensor"]:
        """Sample or select greedy action.

        Args:
            obs: float32 tensor, shape (obs_dim,) — single observation
            deterministic: If True, take argmax mode (evaluation mode)

        Returns:
            action_vec: int numpy array of shape (1,) containing the mode index
            log_prob: scalar tensor
            value: scalar tensor — critic value estimate
        """
        with torch.no_grad():
            dists, value = self.forward(obs.unsqueeze(0))
            actions = []
            log_prob = torch.tensor(0.0)
            for dist in dists:
                if deterministic:
                    a = dist.probs.argmax(dim=-1)
                else:
                    a = dist.sample()
                log_prob = log_prob + dist.log_prob(a)
                actions.append(int(a.item()))
        return np.array(actions, dtype=int), log_prob, value.squeeze(0)

    def evaluate_actions(
        self,
        obs_batch: "torch.Tensor",
        actions_batch: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Evaluate a batch of (obs, actions) for PPO update.

        Args:
            obs_batch: float32 tensor, shape (batch, obs_dim)
            actions_batch: int64 tensor, shape (batch,) or (batch, 1) mode actions

        Returns:
            log_probs: shape (batch,) — log-prob per sample
            entropy: scalar — mean mode entropy (for entropy regularisation)
            values: shape (batch, 1)
        """
        if actions_batch.ndim == 1:
            actions_batch = actions_batch[:, None]
        dists, values = self.forward(obs_batch)
        log_probs = torch.zeros(obs_batch.shape[0], device=obs_batch.device)
        entropy = torch.tensor(0.0, device=obs_batch.device)
        for i, dist in enumerate(dists):
            log_probs = log_probs + dist.log_prob(actions_batch[:, i])
            entropy = entropy + dist.entropy().mean()
        return log_probs, entropy, values

    def get_mode_probs(self, obs: "torch.Tensor") -> np.ndarray:
        """Return mode probabilities from the first actor head (for rationale).

        Args:
            obs: float32 tensor, shape (obs_dim,)

        Returns:
            probs: float32 numpy array of shape (7,)
        """
        with torch.no_grad():
            dists, _ = self.forward(obs.unsqueeze(0))
            return dists[0].probs.squeeze(0).numpy()


class RandomPolicy:
    """Random policy with the same interface as ActorCritic.

    Used when rl_mock=True (CI mode without torch or trained checkpoint).
    Samples uniformly from the 7 EventSat modes.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_size: int = HIDDEN_SIZE,
        action_dims: Optional[List[int]] = None,
    ) -> None:
        self._action_dims = action_dims or ACTION_DIMS
        self._rng = np.random.default_rng()

    def get_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """Return random action with dummy log_prob and value.

        Returns:
            action_vec: int array of shape (1,)
            log_prob: 0.0 (placeholder)
            value: 0.0 (placeholder)
        """
        actions = np.array(
            [self._rng.integers(0, d) for d in self._action_dims], dtype=int
        )
        return actions, 0.0, 0.0

    def evaluate_actions(self, obs_batch: np.ndarray, actions_batch: np.ndarray):
        """Placeholder — returns zeros for all outputs."""
        n = len(obs_batch)
        return np.zeros(n), np.float64(0.0), np.zeros((n, 1))

    def get_mode_probs(self, obs: np.ndarray) -> np.ndarray:
        """Return uniform probabilities."""
        n = self._action_dims[0]
        return np.ones(n, dtype=np.float32) / n

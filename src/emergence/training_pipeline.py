"""
PPO Training Pipeline for Subsymbolic EventSat.

Implements Proximal Policy Optimization (Schulman et al. 2017) with:
- Clipped surrogate objective
- Value function loss (MSE)
- Entropy regularisation
- Gradient norm clipping
- Linear learning rate schedule

Hyperparameters grounded in Oliver et al. EUCASS 2025 (8KDZ5Z53):
  lr=1e-4→1e-5, gamma=0.97, clip_ratio=0.3, ppo_epochs=30,
  batch_size=4096, minibatch_size=256

PPO only — on-policy, aligns with the per-episode trajectory collection
pattern in experiment_runner.py. No DQN.

Papers:
- Oliver et al. EUCASS 2025 (8KDZ5Z53): PPO hyperparameters
- Hamilton et al. 2025 (GWQ3LK6H): clip_ratio=0.3, 30 SGD epochs
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    Adam = None  # type: ignore

from src.emergence.rollout_buffer import RolloutBuffer

logger = logging.getLogger(__name__)

# Default hyperparameters — Oliver et al. EUCASS 2025
_DEFAULTS = {
    "lr": 1e-4,
    "lr_schedule": [[0, 1e-4], [3_000_000, 1e-5]],
    "gamma": 0.97,
    "gae_lambda": 0.95,
    "clip_ratio": 0.3,
    "ppo_epochs": 30,
    "entropy_coef": 0.01,
    "value_coef": 1.0,
    "max_grad_norm": 0.5,
    "minibatch_size": 256,
}


class PPOTrainer:
    """PPO trainer for ActorCritic with MultiDiscrete([7, 2, 2]) actions.

    Args:
        policy: ActorCritic instance (must have torch available).
        config: Training config dict. Keys match _DEFAULTS above.
    """

    def __init__(self, policy: Any, config: Optional[Dict[str, Any]] = None) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for PPOTrainer. Install with: uv sync --extra rl")

        self.policy = policy
        cfg = dict(_DEFAULTS)
        if config:
            cfg.update(config)

        self.clip_ratio: float = cfg["clip_ratio"]
        self.ppo_epochs: int = int(cfg["ppo_epochs"])
        self.entropy_coef: float = cfg["entropy_coef"]
        self.value_coef: float = cfg["value_coef"]
        self.max_grad_norm: float = cfg["max_grad_norm"]
        self.minibatch_size: int = int(cfg["minibatch_size"])
        self.gamma: float = cfg["gamma"]
        self.gae_lambda: float = cfg["gae_lambda"]
        self._lr_schedule: List[List[float]] = cfg["lr_schedule"]

        self.optimizer = Adam(policy.parameters(), lr=float(cfg["lr"]))
        self.training_step: int = 0
        self._last_update_info: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run PPO update using transitions in buffer.

        Calls compute_returns_and_advantages() on the buffer, then runs
        ppo_epochs of minibatch SGD updates.

        Args:
            buffer: RolloutBuffer with stored transitions.

        Returns:
            Dict with: policy_loss, value_loss, entropy, approx_kl, grad_norm
        """
        # GAE computation
        last_value = 0.0  # Terminal state bootstrap
        if not buffer.dones[buffer.size - 1]:
            # Episode not done — bootstrap from last value stored
            last_obs = torch.FloatTensor(buffer.observations[buffer.size - 1])
            with torch.no_grad():
                _, last_v = self.policy.forward(last_obs.unsqueeze(0))
            last_value = float(last_v.item())

        buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Track aggregated metrics across epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_grad_norm = 0.0
        update_count = 0

        for _ in range(self.ppo_epochs):
            for batch in buffer.get_batches(self.minibatch_size):
                metrics = self._update_step(batch)
                total_policy_loss += metrics["policy_loss"]
                total_value_loss += metrics["value_loss"]
                total_entropy += metrics["entropy"]
                total_approx_kl += metrics["approx_kl"]
                total_grad_norm += metrics["grad_norm"]
                update_count += 1

        self.training_step += buffer.size
        self._update_lr()

        n = max(update_count, 1)
        info = {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": total_approx_kl / n,
            "grad_norm": total_grad_norm / n,
            "training_step": float(self.training_step),
        }
        self._last_update_info = info
        logger.debug(
            "PPO update: policy_loss=%.4f value_loss=%.4f entropy=%.4f kl=%.4f",
            info["policy_loss"], info["value_loss"], info["entropy"], info["approx_kl"],
        )
        return info

    def save(self, path: str | Path) -> None:
        """Save checkpoint (policy state_dict + optimizer + training_step)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
            },
            path,
        )
        logger.info("Checkpoint saved to %s (step %d)", path, self.training_step)

    def load(self, path: str | Path) -> None:
        """Load checkpoint into policy and optimizer."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        logger.info("Checkpoint loaded from %s (step %d)", path, self.training_step)

    def get_last_update_info(self) -> Dict[str, float]:
        return self._last_update_info

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Single PPO minibatch gradient step."""
        obs = torch.FloatTensor(batch["observations"])
        actions = torch.LongTensor(batch["actions"])
        old_log_probs = torch.FloatTensor(batch["log_probs"])
        returns = torch.FloatTensor(batch["returns"])
        advantages = torch.FloatTensor(batch["advantages"])

        # Normalize advantages per minibatch (standard PPO)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy evaluation
        new_log_probs, entropy, values = self.policy.evaluate_actions(obs, actions)
        values = values.squeeze(-1)

        # PPO clipped surrogate objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss (MSE)
        value_loss = nn.functional.mse_loss(values, returns)

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Approximate KL for monitoring (early stopping not implemented)
        with torch.no_grad():
            approx_kl = float(((old_log_probs - new_log_probs).mean()).abs().item())

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": approx_kl,
            "grad_norm": float(grad_norm.item()),
        }

    def _update_lr(self) -> None:
        """Apply piecewise linear learning rate schedule."""
        schedule = self._lr_schedule
        if len(schedule) < 2:
            return
        step = self.training_step
        # Find surrounding schedule points
        for i in range(len(schedule) - 1):
            s0, lr0 = schedule[i]
            s1, lr1 = schedule[i + 1]
            if step <= s1:
                if s1 == s0:
                    lr = lr1
                else:
                    t = (step - s0) / (s1 - s0)
                    lr = lr0 + t * (lr1 - lr0)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr
                return
        # Past last schedule point — use final lr
        final_lr = schedule[-1][1]
        for pg in self.optimizer.param_groups:
            pg["lr"] = final_lr

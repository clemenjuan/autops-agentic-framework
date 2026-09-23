"""Inference adapter for RLlib checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class RLLibPolicyAdapter:
    """Small wrapper exposing the policy interface used by SubsymbolicEventSat."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        policy_id: str = "shared_policy",
        action_dims: list[int] | None = None,
    ) -> None:
        try:
            from ray.rllib.algorithms.algorithm import Algorithm
        except ImportError as exc:
            raise ImportError("ray[rllib] is required to load RLlib checkpoints") from exc

        from src.rl.models import register_autops_models

        register_autops_models()
        raw_checkpoint_path = str(checkpoint_path)
        if "://" in raw_checkpoint_path:
            self.checkpoint_path = raw_checkpoint_path
        else:
            path = Path(raw_checkpoint_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"RLlib checkpoint not found: {path}")
            self.checkpoint_path = str(path.resolve())
            self._register_checkpoint_env_names(path.resolve())
        self.policy_id = policy_id
        self._action_dims = list(action_dims or [7])
        self._algo = Algorithm.from_checkpoint(self.checkpoint_path)
        self._rng = np.random.default_rng()

    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, float, float]:
        obs_array = np.asarray(obs, dtype=np.float32)
        if deterministic:
            action = self._algo.compute_single_action(
                obs_array,
                policy_id=self.policy_id,
                explore=False,
            )
            if isinstance(action, tuple):
                action = action[0]
        else:
            # Ask RLlib for policy logits without invoking its process-global
            # exploration RNG, then sample the MultiDiscrete action through a
            # private generator that can be restarted at every episode.
            result = self._algo.compute_single_action(
                obs_array,
                policy_id=self.policy_id,
                explore=False,
                full_fetch=True,
            )
            action = result[0] if isinstance(result, tuple) else result
            info = result[2] if isinstance(result, tuple) and len(result) >= 3 else {}
            logits = np.asarray(info.get("action_dist_inputs", []), dtype=np.float64)
            sampled = self._sample_multidiscrete(logits)
            if sampled is not None:
                action = sampled
        return np.asarray(action, dtype=int), 0.0, 0.0

    def seed(self, seed: int) -> None:
        """Restart the adapter-owned exploration stream."""
        self._rng = np.random.default_rng(int(seed))

    def reset(self) -> None:
        """Policy inference has no recurrent episode state in this adapter."""

    def _sample_multidiscrete(self, logits: np.ndarray) -> np.ndarray | None:
        """Sample independent categorical heads from flattened RLlib logits.

        AUTOPS checkpoints use a MultiDiscrete action distribution. If a
        checkpoint does not expose complete logits, the caller keeps RLlib's
        deterministic action instead of falling back to an unseeded sampler.
        """
        expected = sum(self._action_dims)
        if logits.size < expected or not np.all(np.isfinite(logits[:expected])):
            return None
        actions = []
        offset = 0
        for dim in self._action_dims:
            head = logits[offset : offset + dim]
            head = head - np.max(head)
            probs = np.exp(head)
            total = float(np.sum(probs))
            if not np.isfinite(total) or total <= 0.0:
                return None
            actions.append(int(self._rng.choice(dim, p=probs / total)))
            offset += dim
        return np.asarray(actions, dtype=int)

    def get_mode_probs(self, obs: np.ndarray) -> np.ndarray:
        """Best-effort mode probabilities for explanation metrics."""
        try:
            result = self._algo.compute_single_action(
                np.asarray(obs, dtype=np.float32),
                policy_id=self.policy_id,
                explore=False,
                full_fetch=True,
            )
            info = result[2] if isinstance(result, tuple) and len(result) >= 3 else {}
            logits = np.asarray(info.get("action_dist_inputs", []), dtype=np.float32)
            mode_dim = int(self._action_dims[0]) if self._action_dims else 1
            mode_logits = logits[:mode_dim]
            if mode_logits.shape[0] == mode_dim:
                mode_logits = mode_logits - np.max(mode_logits)
                probs = np.exp(mode_logits)
                return probs / np.sum(probs)
        except Exception:
            pass
        mode_dim = int(self._action_dims[0]) if self._action_dims else 1
        return np.ones(mode_dim, dtype=np.float32) / mode_dim

    def close(self) -> None:
        # Representations also call close() from __del__; detach first so
        # runner teardown and garbage collection cannot stop the algorithm twice.
        algo = self._algo
        self._algo = None
        if hasattr(algo, "stop"):
            algo.stop()

    def _register_checkpoint_env_names(self, path: Path) -> None:
        """Register likely AUTOPS env names needed by restored RLlib configs."""
        try:
            from ray.tune.registry import register_env

            from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv
        except ImportError:
            return

        for candidate in {path.name, path.parent.name}:
            if not candidate:
                continue
            env_name = f"autops_{candidate}_rllib"
            register_env(env_name, lambda cfg: AUTOPSRLLibMultiAgentEnv(cfg))

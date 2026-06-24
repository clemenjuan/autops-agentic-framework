"""Inference adapter for RLlib checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.rl.space_adapters import ACTION_DIMS


class RLLibPolicyAdapter:
    """Small wrapper exposing the policy interface used by SubsymbolicEventSat."""

    def __init__(self, checkpoint_path: str | Path, policy_id: str = "shared_policy") -> None:
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
        self._algo = Algorithm.from_checkpoint(self.checkpoint_path)

    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, float, float]:
        action = self._algo.compute_single_action(
            np.asarray(obs, dtype=np.float32),
            policy_id=self.policy_id,
            explore=not deterministic,
        )
        if isinstance(action, tuple):
            action = action[0]
        return np.asarray(action, dtype=int), 0.0, 0.0

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
            mode_logits = logits[: ACTION_DIMS[0]]
            if mode_logits.shape[0] == ACTION_DIMS[0]:
                mode_logits = mode_logits - np.max(mode_logits)
                probs = np.exp(mode_logits)
                return probs / np.sum(probs)
        except Exception:
            pass
        return np.ones(ACTION_DIMS[0], dtype=np.float32) / ACTION_DIMS[0]

    def close(self) -> None:
        if hasattr(self._algo, "stop"):
            self._algo.stop()

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

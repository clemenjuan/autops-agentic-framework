"""RLlib wrapper for the original AUTOPS EventSat ActorCritic architecture."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
except ImportError as exc:  # pragma: no cover - exercised by optional-dep imports
    raise ImportError("ray[rllib] and torch are required for AUTOPSActorCriticModel") from exc


class AUTOPSActorCriticModel(TorchModelV2, nn.Module):
    """RLlib model matching the legacy ``representation.neural_policy.ActorCritic``.

    Architecture:
    - Shared trunk: 25 -> 256 -> 256 with Tanh activations
    - Three independent actor heads: 7, 2, 2 logits
    - One critic head: scalar value
    - Orthogonal init: sqrt(2) trunk, 0.01 actor heads, 1.0 critic

    RLlib consumes the concatenated actor logits for MultiDiscrete actions, but
    the underlying module still keeps the original per-head structure.
    """

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        num_outputs: int,
        model_config: Dict[str, Any],
        name: str,
        **kwargs: Any,
    ) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_cfg = dict(model_config.get("custom_model_config", {}))
        obs_dim = int(np.prod(obs_space.shape))
        hidden_size = int(custom_cfg.get("hidden_size", 256))
        action_dims = self._resolve_action_dims(action_space, custom_cfg)
        if num_outputs != sum(action_dims):
            raise ValueError(
                f"AUTOPSActorCriticModel expected num_outputs={sum(action_dims)}, "
                f"got {num_outputs}"
            )

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor_heads = nn.ModuleList(
            [nn.Linear(hidden_size, dim) for dim in action_dims]
        )
        self.critic_head = nn.Linear(hidden_size, 1)
        self._value_out: torch.Tensor | None = None
        self._action_dims = action_dims
        self._init_weights()

    def forward(
        self,
        input_dict: Dict[str, Any],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        obs = input_dict.get("obs_flat")
        if obs is None:
            obs = input_dict["obs"]
        obs = obs.float()
        features = self.trunk(obs)
        logits = torch.cat([head(features) for head in self.actor_heads], dim=-1)
        self._value_out = self.critic_head(features).squeeze(-1)
        return logits, state

    def value_function(self) -> torch.Tensor:
        if self._value_out is None:
            raise RuntimeError("value_function() called before forward()")
        return self._value_out

    def _init_weights(self) -> None:
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        for head in self.actor_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def _resolve_action_dims(
        self,
        action_space: Any,
        custom_cfg: Dict[str, Any],
    ) -> List[int]:
        configured = custom_cfg.get("action_dims")
        if configured is not None:
            return self._validate_action_dims(configured)
        if hasattr(action_space, "nvec"):
            return self._validate_action_dims(list(action_space.nvec))
        return self._validate_action_dims([7, 2, 2])

    def _validate_action_dims(self, action_dims: Sequence[Any]) -> List[int]:
        dims = [int(dim) for dim in action_dims]
        if not dims or any(dim <= 0 for dim in dims):
            raise ValueError("action_dims must be a non-empty sequence of positive integers")
        return dims

"""RLlib model registry for AUTOPS RL policies."""
from __future__ import annotations

_REGISTERED = False


def register_autops_models() -> None:
    """Register AUTOPS custom models with RLlib once per process."""
    global _REGISTERED
    if _REGISTERED:
        return

    try:
        from ray.rllib.models import ModelCatalog
    except ImportError as exc:
        raise ImportError("ray[rllib] is required to register AUTOPS RL models") from exc

    from src.rl.models.autops_actor_critic import AUTOPSActorCriticModel

    ModelCatalog.register_custom_model(
        "autops_actor_critic_v1",
        AUTOPSActorCriticModel,
    )
    _REGISTERED = True

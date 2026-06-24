"""Policy-sharing helpers for RLlib multi-agent training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List


@dataclass(frozen=True)
class PolicySharingConfig:
    """Declarative policy sharing strategy for RLlib.

    This is not a morphological-matrix dimension.  It only controls whether
    multiple RLlib agents share neural weights during training.
    """

    mode: str = "shared_all"

    @classmethod
    def from_config(cls, config: Dict[str, Any] | str | None) -> "PolicySharingConfig":
        if config is None:
            return cls()
        if isinstance(config, str):
            return cls(mode=config)
        return cls(mode=str(config.get("mode", "shared_all")))

    def policy_id_for(self, agent_id: str) -> str:
        if self.mode == "shared_all":
            return "shared_policy"
        if self.mode == "independent_per_agent":
            return f"policy_{agent_id}"
        if self.mode == "shared_by_role":
            if agent_id == "mission_manager":
                return "manager_policy"
            if agent_id.startswith("sat_agent_"):
                return "satellite_policy"
            if agent_id == "central_agent":
                return "central_policy"
            return "shared_policy"
        raise ValueError(
            "policy_sharing.mode must be one of "
            "{'shared_all', 'shared_by_role', 'independent_per_agent'}, "
            f"got '{self.mode}'"
        )

    def policy_ids(self, agent_ids: Iterable[str]) -> List[str]:
        return sorted({self.policy_id_for(agent_id) for agent_id in agent_ids})

    def mapping_fn(self) -> Callable[..., str]:
        def _map(agent_id: str, *args: Any, **kwargs: Any) -> str:
            return self.policy_id_for(agent_id)

        return _map


def build_policy_specs(
    agent_ids: Iterable[str],
    observation_space: Any,
    action_space: Any,
    sharing: PolicySharingConfig,
) -> Dict[str, Any]:
    """Build RLlib PolicySpec objects for the selected sharing strategy."""
    try:
        from ray.rllib.policy.policy import PolicySpec
    except ImportError as exc:
        raise ImportError("ray[rllib] is required to build RLlib policies") from exc

    return {
        policy_id: PolicySpec(
            policy_class=None,
            observation_space=observation_space,
            action_space=action_space,
            config={},
        )
        for policy_id in sharing.policy_ids(agent_ids)
    }

"""
Independent MAS Organization.

Implements the Independent MAS topology from Kim et al. (2025) [FVFQ73RF]
"Towards a Science of Scaling Agent Systems":
  S = (A, E, C, Ω) where
  A = {sat_agent_0, ..., sat_agent_n-1},
  C = ∅  (no inter-agent communication),
  Ω = independent (each agent acts without coordination).

Each agent acts based solely on its local observation without any
communication or coordination with other agents. Unlike Decentralized MAS,
there is no consensus mechanism — agents are truly independent.

One agent per satellite; at N=1 independent operation is equivalent to
SingleAgentSystem (SAS) with no coordination overhead.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agent_organization.base import AgentAction, AgentObservation, AgentOrganization
from src.environment.satellite_env import scope_observation


class IndependentMAS(AgentOrganization):
    """Multiple independent agents with no inter-agent communication."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._agent_ids: List[str] = []

    def initialize(self, constellation_size: int, **kwargs: Any) -> None:
        self._agent_ids = [f"sat_agent_{i}" for i in range(constellation_size)]

    def satellite_for_agent(self, agent_id: str) -> str:
        """Map ``sat_agent_i`` -> ``sat_i``.

        Raises ``ValueError`` on an id that does not follow the convention,
        rather than silently returning it (mirrors ``policy_mapping.py``, which
        also raises on unexpected ids).
        """
        prefix = "sat_agent_"
        if not agent_id.startswith(prefix):
            raise ValueError(
                f"IndependentMAS expects 'sat_agent_i' agent ids, got '{agent_id}'"
            )
        return f"sat_{agent_id[len(prefix):]}"

    def distribute_observation(
        self,
        env_observation: Any,
    ) -> Dict[str, AgentObservation]:
        """Give each agent a partial view of only its own satellite.

        This is the *who-sees-what* decision for the Independent topology
        (C = ∅): each agent perceives solely its own satellite, with no
        inter-agent messages. The view is built with :func:`scope_observation`;
        because the organization's view is authoritative, the agent cannot see
        any other satellite even downstream.
        """
        result: Dict[str, AgentObservation] = {}
        for agent_id in self._agent_ids:
            sat = self.satellite_for_agent(agent_id)
            result[agent_id] = AgentObservation(
                agent_id=agent_id,
                local_state={
                    "full_observation": scope_observation(env_observation, [sat])
                },
                metadata={"satellite_id": sat},
            )
        return result

    def collect_actions(
        self,
        agent_actions: Dict[str, AgentAction],
    ) -> Dict[str, Any]:
        """Merge per-agent ``{satellite_id: action}`` dicts into one env action."""
        env_actions: Dict[str, Any] = {}
        for agent_action in agent_actions.values():
            if agent_action is not None and isinstance(agent_action.action, dict):
                env_actions.update(agent_action.action)
        return env_actions

    def get_agents(self) -> List[str]:
        return list(self._agent_ids)

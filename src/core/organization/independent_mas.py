"""
Independent MAS Organization.

Implements the Independent MAS topology from Kim et al. (2025) [FVFQ73RF]
"Towards a Science of Scaling Agent Systems":
  S = (A, E, C, Ω) where
  A = {sat_agent_0, ..., sat_agent_n-1},
  C = ∅  (no inter-agent communication),
  Ω = independent (each agent acts without coordination).

Each agent acts based solely on its own satellite's local observation, with no
communication or coordination with the others. Unlike Decentralized MAS there is
no consensus step — agents are truly independent.

Coordination consequence (the point of the organisation axis): because each
agent only sees its own satellite and picks greedily, several agents can
independently select the *same* high-priority RSO when their visibility windows
overlap. ``collect_actions`` merges the per-satellite actions verbatim — it does
**not** deconflict — so those collisions reach the environment as wasted
duplicate observations. A coordinated organisation (SAS / CentralizedMAS that
plans over the full constellation) avoids this; IndependentMAS does not. This is
the duplicate/waste behaviour Flamingo-lite is built to measure.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.organization.base import AgentAction, AgentObservation, AgentOrganization
from src.core.satellite_env import scope_observation


class IndependentMAS(AgentOrganization):
    """Multiple independent per-satellite agents with no inter-agent communication.

    Each agent receives a local view of only its own satellite (and only that
    satellite's visible tasks) and produces an action for that satellite alone.
    C = ∅, Ω = independent. See Kim et al. (2025) [FVFQ73RF] §4.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._agent_ids: List[str] = []

    def initialize(self, constellation_size: int, **kwargs: Any) -> None:
        self._agent_ids = [f"sat_agent_{i}" for i in range(constellation_size)]

    def satellite_for_agent(self, agent_id: str) -> str:
        """Map ``sat_agent_i`` to the satellite it controls.

        ``MultiEventsat`` uses the default ``sat_i`` ids. Other scenarios can set
        ``agent_organization_config.satellite_prefix`` (for example
        ``"flamingo"`` or ``"sat"``) or an explicit ``satellite_ids`` list.
        """
        idx = self._agent_index(agent_id)
        explicit = self.config.get("satellite_ids")
        if explicit is not None:
            try:
                return str(explicit[idx])
            except IndexError as exc:
                raise ValueError(
                    f"No satellite_ids[{idx}] configured for agent '{agent_id}'"
                ) from exc
        prefix = str(self.config.get("satellite_prefix", "sat"))
        return f"{prefix}_{idx}"

    def distribute_observation(
        self,
        env_observation: Any,
    ) -> Dict[str, AgentObservation]:
        """Give each agent a local view of only its own satellite.

        The configured ``satellite_for_agent`` mapping is authoritative when it
        names a satellite present in the observation. If not, we fall back to the
        observation's satellite ordering so existing scenario-specific ids such
        as ``flamingo_0`` still receive correctly scoped local views.
        """
        sat_ids = list(env_observation.constellation_state.satellites.keys())
        result: Dict[str, AgentObservation] = {}
        for idx, agent_id in enumerate(self._agent_ids):
            mapped_sat_id = self.satellite_for_agent(agent_id)
            sat_id = mapped_sat_id
            if sat_id not in env_observation.constellation_state.satellites:
                if idx >= len(sat_ids):
                    continue
                sat_id = sat_ids[idx]
            result[agent_id] = AgentObservation(
                agent_id=agent_id,
                local_state={
                    "full_observation": scope_observation(env_observation, [sat_id])
                },
                metadata={"satellite_id": sat_id},
            )
        return result

    def collect_actions(
        self,
        agent_actions: Dict[str, AgentAction],
    ) -> Dict[str, Any]:
        """Merge per-satellite actions verbatim — no deconfliction (Ω = independent).

        Each agent's payload is a ``{satellite_id: action}`` mapping for its own
        satellite; merging them composes the environment action dict. Collisions
        on the same target are intentionally left in place so the environment can
        count them as duplicate observations.
        """
        merged: Dict[str, Any] = {}
        for agent_action in agent_actions.values():
            if agent_action is not None and isinstance(agent_action.action, dict):
                merged.update(agent_action.action)
        return merged

    def get_agents(self) -> List[str]:
        return list(self._agent_ids)

    @staticmethod
    def _agent_index(agent_id: str) -> int:
        prefix = "sat_agent_"
        if not agent_id.startswith(prefix):
            raise ValueError(
                f"IndependentMAS expects 'sat_agent_i' agent ids, got '{agent_id}'"
            )
        try:
            return int(agent_id[len(prefix):])
        except ValueError as exc:
            raise ValueError(
                f"IndependentMAS expects 'sat_agent_i' agent ids, got '{agent_id}'"
            ) from exc

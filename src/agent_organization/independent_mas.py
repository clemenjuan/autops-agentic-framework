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

from src.agent_organization.base import AgentAction, AgentObservation, AgentOrganization


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

    def distribute_observation(
        self,
        env_observation: Any,
    ) -> Dict[str, AgentObservation]:
        """Give each agent a local view of only its own satellite.

        Agent ``sat_agent_i`` is mapped by index to the i-th satellite in the
        constellation observation and receives an observation containing just
        that satellite's state and just that satellite's visible tasks — no
        sight of what the others can see or intend (C = ∅).
        """
        sat_ids = list(env_observation.constellation_state.satellites.keys())
        result: Dict[str, AgentObservation] = {}
        for idx, agent_id in enumerate(self._agent_ids):
            if idx >= len(sat_ids):
                continue
            sat_id = sat_ids[idx]
            result[agent_id] = AgentObservation(
                agent_id=agent_id,
                local_state={
                    "full_observation": self._local_view(env_observation, sat_id)
                },
            )
        return result

    def collect_actions(
        self,
        agent_actions: Dict[str, AgentAction],
    ) -> Dict[str, Any]:
        """Merge per-satellite actions verbatim — no deconfliction (Ω = independent).

        Each agent's payload is a ``{satellite_id: action}`` mapping for its own
        satellite; merging them composes the environment action dict. Collisions
        on the same RSO are intentionally left in place so the environment counts
        them as duplicate observations.
        """
        merged: Dict[str, Any] = {}
        for agent_action in agent_actions.values():
            if agent_action is not None and isinstance(agent_action.action, dict):
                merged.update(agent_action.action)
        return merged

    def get_agents(self) -> List[str]:
        return list(self._agent_ids)

    @staticmethod
    def _local_view(env_observation: Any, sat_id: str) -> Any:
        """Build a single-satellite slice of the environment observation."""
        # Import lazily to keep the organisation layer free of a hard dependency
        # on any one scenario's environment module at import time.
        from src.environment.satellite_env import (
            ConstellationState,
            EnvironmentObservation,
        )

        cstate = env_observation.constellation_state
        sat_state = cstate.satellites.get(sat_id)
        local_satellites = {sat_id: sat_state} if sat_state is not None else {}
        local_tasks = [
            task
            for task in (getattr(env_observation, "tasks", []) or [])
            if task.get("satellite_id") == sat_id
        ]
        return EnvironmentObservation(
            constellation_state=ConstellationState(
                timestep=cstate.timestep,
                epoch_seconds=cstate.epoch_seconds,
                satellites=local_satellites,
                global_info=dict(getattr(cstate, "global_info", {}) or {}),
            ),
            tasks=local_tasks,
            events=list(getattr(env_observation, "events", []) or []),
        )

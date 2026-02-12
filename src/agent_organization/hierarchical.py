"""
Hierarchical Agent Organization.

A mission manager agent coordinates with local satellite agents.
The manager receives global observations and distributes high-level
directives; local agents handle satellite-level decisions.

Placeholder — awaiting scenario selection and Phase 2+ implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agent_organization.base import AgentAction, AgentObservation, AgentOrganization


class HierarchicalOrganization(AgentOrganization):
    """Two-level hierarchy: mission manager + local satellite agents."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._manager_id: str = "mission_manager"
        self._local_agent_ids: List[str] = []

    def initialize(self, constellation_size: int, **kwargs: Any) -> None:
        self._local_agent_ids = [
            f"sat_agent_{i}" for i in range(constellation_size)
        ]

    def distribute_observation(
        self,
        env_observation: Any,
    ) -> Dict[str, AgentObservation]:
        raise NotImplementedError(
            "HierarchicalOrganization.distribute_observation not yet implemented. "
            "Requires scenario-specific observation partitioning logic."
        )

    def collect_actions(
        self,
        agent_actions: Dict[str, AgentAction],
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "HierarchicalOrganization.collect_actions not yet implemented."
        )

    def get_agents(self) -> List[str]:
        return [self._manager_id] + self._local_agent_ids

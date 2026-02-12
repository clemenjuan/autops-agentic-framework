"""
Centralized Agent Organization.

A single agent controls the entire constellation. All observations are
aggregated into one global view; the single agent produces actions for
every satellite.

Placeholder — awaiting scenario selection and Phase 2 implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agent_organization.base import AgentAction, AgentObservation, AgentOrganization


class CentralizedOrganization(AgentOrganization):
    """Single-agent centralized control of the constellation.

    One agent receives the full environment observation and produces
    actions for all satellites.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._agent_id: str = "central_agent"
        self._constellation_size: int = 0

    def initialize(self, constellation_size: int, **kwargs: Any) -> None:
        self._constellation_size = constellation_size

    def distribute_observation(
        self,
        env_observation: Any,
    ) -> Dict[str, AgentObservation]:
        return {
            self._agent_id: AgentObservation(
                agent_id=self._agent_id,
                local_state={"full_observation": env_observation},
            )
        }

    def collect_actions(
        self,
        agent_actions: Dict[str, AgentAction],
    ) -> Dict[str, Any]:
        central_action = agent_actions.get(self._agent_id)
        if central_action is None:
            return {}
        # The central agent's action payload should already be a mapping
        # of satellite_id → action.
        return central_action.action if isinstance(central_action.action, dict) else {}

    def get_agents(self) -> List[str]:
        return [self._agent_id]

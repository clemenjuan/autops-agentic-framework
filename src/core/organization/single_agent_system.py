"""
Single-Agent System (SAS) Organization.

Implements the Single-Agent System topology from Kim et al. (2025)
[FVFQ73RF] "Towards a Science of Scaling Agent Systems":
  S = (A, E, C, Ω) where
  A = {central_agent},   |A| = 1,
  C = undefined (no inter-agent communication),
  Ω = direct (agent action is the environment action directly).

Complexity: O(k) where k = max reasoning iterations.

A single agent receives the full constellation observation and produces
actions for all satellites. Maximum context integration — upper bound for
context quality, lower bound for parallelism.

Empirical prediction (Kim et al. 2025, 180 configs): satellite mode
selection is sequential constraint satisfaction → SAS is expected to
outperform distributed organizations. Capability saturation effect
(β̂ = −0.404): multi-agent overhead negates gains once the SAS baseline
exceeds ~45% utility.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.organization.base import AgentAction, AgentObservation, AgentOrganization


class SingleAgentSystem(AgentOrganization):
    """Single-agent centralized control of the constellation (SAS).

    One agent receives the full environment observation and produces
    actions for all satellites. No inter-agent communication.
    Zero coordination overhead.

    Kim et al. (2025): |A| = 1, C undefined, Ω = direct, complexity O(k).
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

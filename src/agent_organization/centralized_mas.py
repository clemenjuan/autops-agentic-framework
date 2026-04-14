"""
Centralized MAS Organization.

Implements the Centralized MAS topology from Kim et al. (2025) [FVFQ73RF]
"Towards a Science of Scaling Agent Systems":
  S = (A, E, C, Ω) where
  A = {mission_manager, sat_agent_0, ...},
  C = star (orchestrator → locals, unidirectional),
  Ω = hierarchical (manager sets directive; local agent executes).

Complexity: O(r·n·k) where r = orchestrator rounds (1 here), n = constellation
size, k = max reasoning iterations per agent.

The mission_manager receives the full environment observation and produces a
strategic directive (mode recommendation, budget constraint). Local satellite
agents receive the same observation plus the manager's directive from the
previous step as a message, then produce the final environment action.

Design notes:
- Manager and local agents share the same decision loop class and representation
  (morphological matrix varies organization, not representation).
- Manager directive carries over step-to-step, reset to None on initialize().
- Episode-boundary bleed (last directive of episode N → first step of episode
  N+1) is acceptable: the first step only has this as context, not as a hard
  constraint.
- EventSat (N=1): AG/CG ops paradigms are intentionally absent for this org
  because the ground station already acts as the strategic planning layer —
  adding an onboard manager creates overhead with no coordination benefit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agent_organization.base import AgentAction, AgentObservation, AgentOrganization


class CentralizedMAS(AgentOrganization):
    """Two-level hierarchy: mission manager + local satellite agents.

    Kim et al. (2025) Centralized MAS: orchestrator coordinates sub-agents
    via sequential message passing. Manager acts first (strategic, long-horizon);
    local agent acts second with manager directive as context (tactical,
    real-time).

    Aggregation policy Ω = hierarchical:
      - Local agent's action is used as the environment action.
      - If no local agent is active, manager's action is used as fallback.
      - Manager's action is stored and passed to local agents as a directive
        message at the next timestep.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._manager_id: str = "mission_manager"
        self._local_agent_ids: List[str] = []
        # Manager directive from the previous timestep (None at episode start)
        self._last_manager_directive: Optional[Any] = None

    def initialize(self, constellation_size: int, **kwargs: Any) -> None:
        self._local_agent_ids = [
            f"sat_agent_{i}" for i in range(constellation_size)
        ]
        self._last_manager_directive = None

    def distribute_observation(
        self,
        env_observation: Any,
    ) -> Dict[str, AgentObservation]:
        """Distribute env observation to manager and local agents.

        Mission manager receives the full observation (strategic view).
        Each local satellite agent receives the full observation plus the
        manager's directive from the previous step as a message (Kim et al.
        C = star, Ω = hierarchical).
        """
        manager_obs = AgentObservation(
            agent_id=self._manager_id,
            local_state={"full_observation": env_observation},
        )

        # Directive message: manager's action payload from t-1
        messages: List[Dict[str, Any]] = []
        if self._last_manager_directive is not None:
            messages = [
                {
                    "from": self._manager_id,
                    "directive": self._last_manager_directive,
                }
            ]

        result: Dict[str, AgentObservation] = {self._manager_id: manager_obs}
        for local_id in self._local_agent_ids:
            result[local_id] = AgentObservation(
                agent_id=local_id,
                local_state={"full_observation": env_observation},
                messages=messages,
            )
        return result

    def collect_actions(
        self,
        agent_actions: Dict[str, AgentAction],
    ) -> Dict[str, Any]:
        """Aggregate agent actions using hierarchical policy Ω.

        Stores manager's action as the directive for the next step, then
        returns the first local agent's action as the environment action.
        Falls back to manager's action if no local agent produced an output.
        """
        manager_action = agent_actions.get(self._manager_id)

        # Store manager's action as directive for next timestep's messages
        if manager_action is not None:
            self._last_manager_directive = manager_action.action

        # Aggregation policy Ω = hierarchical: local agent action is env action
        for local_id in self._local_agent_ids:
            local_action = agent_actions.get(local_id)
            if local_action is not None and isinstance(local_action.action, dict):
                return local_action.action

        # Fallback: manager's action (no local agents or no valid local action)
        if manager_action is not None and isinstance(manager_action.action, dict):
            return manager_action.action
        return {}

    def get_agents(self) -> List[str]:
        return [self._manager_id] + self._local_agent_ids

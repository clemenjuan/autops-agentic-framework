"""Operations Paradigm — Abstract Base Class.

The operations paradigm sits between the agent organization and the
environment, controlling information flow and action timing. It models
the fundamental difference between ground-based and onboard operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GroundKnowledge:
    """What ground operators know about the satellite.

    This represents the ground segment's view of the satellite state,
    which may be stale (only updated during downlink passes).
    """

    last_update_step: int = 0
    battery_soc: float = 0.8
    data_stored_mb: float = 0.0
    current_mode: str = "charging"
    health_status: str = "nominal"
    observation_hours: float = 0.0
    staleness_steps: int = 0


class OperationsParadigm(ABC):
    """Abstract base class for operations paradigms.

    An operations paradigm controls:
    - What the agent can observe (full state vs. stale ground knowledge)
    - When the agent can act (every step vs. uplink windows only)
    - How actions are buffered and delivered to the environment
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._ground_knowledge = GroundKnowledge()
        self._action_buffer: List[Dict[str, Any]] = []

    @abstractmethod
    def filter_observation(
        self,
        full_observation: Any,
        step: int,
    ) -> Any:
        """Filter the full environment observation based on this paradigm.

        For onboard autonomy, this is a pass-through. For ground-based ops,
        this returns a stale snapshot based on last downlinked telemetry.

        Args:
            full_observation: The complete environment observation.
            step: Current simulation step.

        Returns:
            The observation the agent is allowed to see.
        """
        ...

    @abstractmethod
    def can_act(self, step: int, ground_pass_active: bool) -> bool:
        """Whether the agent can issue commands at this step.

        Args:
            step: Current simulation step.
            ground_pass_active: Whether a ground pass is currently active.

        Returns:
            True if the agent may issue an action this step.
        """
        ...

    @abstractmethod
    def process_action(
        self,
        action: Dict[str, Any],
        step: int,
        ground_pass_active: bool,
    ) -> Dict[str, Any]:
        """Process an agent action through the operations paradigm.

        May buffer, delay, or pass through depending on the paradigm.

        Args:
            action: The action the agent wants to execute.
            step: Current simulation step.
            ground_pass_active: Whether a ground pass is currently active.

        Returns:
            The action to actually execute (may be from buffer or no-op).
        """
        ...

    def update_ground_knowledge(
        self,
        full_observation: Any,
        step: int,
    ) -> None:
        """Update ground knowledge from a successful telemetry downlink.

        Called by the environment/runner when telemetry is downlinked
        during a communication pass.

        Args:
            full_observation: Current full satellite state.
            step: Current simulation step.
        """
        obs = full_observation
        if obs is None:
            return
        cs = obs.constellation_state
        for sat_id, sat in cs.satellites.items():
            self._ground_knowledge.last_update_step = step
            self._ground_knowledge.battery_soc = sat.resources.get(
                "battery_soc", self._ground_knowledge.battery_soc
            )
            self._ground_knowledge.data_stored_mb = sat.resources.get(
                "data_stored_mb", self._ground_knowledge.data_stored_mb
            )
            self._ground_knowledge.current_mode = sat.status
            self._ground_knowledge.health_status = sat.metadata.get(
                "health_status", "nominal"
            )
            self._ground_knowledge.observation_hours = (
                sat.metadata.get("total_observation_s", 0.0) / 3600.0
            )
            self._ground_knowledge.staleness_steps = 0

    def get_ground_knowledge(self) -> GroundKnowledge:
        """Return the current ground knowledge snapshot."""
        return self._ground_knowledge

    def reset(self) -> None:
        """Reset paradigm state for a new episode."""
        self._ground_knowledge = GroundKnowledge()
        self._action_buffer.clear()

    def get_name(self) -> str:
        """Return the paradigm name."""
        return self.__class__.__name__

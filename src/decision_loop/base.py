"""
Decision Loop — Abstract Base Class.

Defines the temporal control flow pattern for agent reasoning. Each
concrete implementation must **strictly follow** an established research
paper (e.g. OODA follows Miller et al. 2021, ReAct follows Yao et al. 2023).

The decision loop orchestrates *when* and *how* the representation module
is invoked; the representation module provides the *what* (actual
knowledge/decision logic).

⚠️  CRITICAL: Do not predefine specific internal steps here. Each
implementation will be created step-by-step following the literature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class DecisionLoop(ABC):
    """Abstract base class for decision-making patterns.

    Decision loops define the temporal control flow of agent reasoning.
    Specific implementations must follow established research papers.

    Attributes:
        representation: The representation module providing decision logic.
        config: Loop-specific configuration from the experiment YAML.
    """

    def __init__(
        self,
        representation: Any,  # Will be ``Representation`` once imported
        config: Dict[str, Any] | None = None,
    ) -> None:
        """Initialise the decision loop.

        Args:
            representation: Representation module to use for decision-making.
            config: Optional loop-specific configuration dictionary.
        """
        self.representation = representation
        self.config = config or {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def process(
        self,
        observation: Any,
        memory: Any,
    ) -> Tuple[Any, Any]:
        """Execute one decision cycle.

        Args:
            observation: Current observation for this agent
                (an :class:`AgentObservation`).
            memory: Agent's memory state from the previous step.

        Returns:
            A tuple of ``(action, updated_memory)`` where *action* is the
            selected action to execute and *updated_memory* is the memory
            state to carry forward.

        Raises:
            ValueError: If the observation format is invalid.
        """
        ...

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return decision loop performance metrics.

        Returns:
            Dictionary of metric names to values (e.g. latency,
            number of reasoning iterations, token count, etc.).
        """
        ...

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset any internal state at the start of a new episode."""

    def get_name(self) -> str:
        """Return a human-readable name for this decision loop."""
        return self.__class__.__name__

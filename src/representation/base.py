"""
Representation — Abstract Base Class.

Defines how knowledge and decisions are represented within decision loops.
The representation provides the "what" — the actual decision logic — while
the decision loop provides the "when/how" (temporal control flow).

The same representation can work with different decision loops.

Types:
- **Symbolic**: Rules, planners, constraint solvers (hand-designed logic).
- **Hybrid / Neuro-symbolic**: LLM reasoning + symbolic tools + MARL-networks.
- **Neural**: Learned policies (RL-trained networks).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Representation(ABC):
    """Abstract base class for knowledge / decision representations.

    Attributes:
        config: Representation-specific configuration from experiment YAML.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialise the representation.

        Args:
            config: Optional representation-specific configuration.
        """
        self.config = config or {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def encode_observation(self, observation: Any) -> Any:
        """Transform a raw observation into the internal representation.

        Args:
            observation: Raw observation (typically an
                :class:`AgentObservation`).

        Returns:
            Encoded / internal state suitable for decision-making in this
            representation.
        """
        ...

    @abstractmethod
    def select_action(self, state: Any, memory: Any) -> Any:
        """Core decision-making logic.

        Given an encoded state and a memory object, produce an action.

        Args:
            state: Internal state produced by :meth:`encode_observation`.
            memory: Agent memory object.

        Returns:
            Selected action.
        """
        ...

    # ------------------------------------------------------------------
    # Optional extension points
    # ------------------------------------------------------------------

    def update(self, experience: Any) -> None:
        """Update the representation from experience (for learned variants).

        The default implementation is a no-op (hand-designed representations
        do not learn).

        Args:
            experience: Experience data for updating (e.g. trajectory).
        """

    def get_metrics(self) -> Dict[str, float]:
        """Return representation-level metrics (e.g. inference time).

        Returns:
            Dictionary of metric name → value.
        """
        return {}

    def get_name(self) -> str:
        """Return a human-readable name for this representation."""
        return self.__class__.__name__

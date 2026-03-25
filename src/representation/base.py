"""
Representation — Abstract Base Class.

Defines how knowledge and decisions are represented within decision loops.
The representation provides the "what" — the actual decision logic — while
the decision loop provides the "when/how" (temporal control flow).

The same representation can work with different decision loops.

Cognitive paradigms (Brooks 1991, Colelough & Regli 2025):
- **Symbolic**: Explicit declarative knowledge — rules, planners, constraint solvers.
- **Subsymbolic**: Implicit learned representations — RL policies, DNNs, base LLMs.
- **Hybrid**: Integration of symbolic + subsymbolic — LLM + tools/memory, DNN + logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.decision_loop.context import DecisionContext


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
    def select_action(self, context: DecisionContext) -> Any:
        """Core decision-making logic.

        Given a :class:`DecisionContext` produced by a decision loop,
        select an action.  The context carries the encoded state, loop
        type, memory reference, and any loop-specific enrichments.

        Args:
            context: Structured decision context from the decision loop.

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

    def reason(self, state: Dict[str, Any], memory: Any) -> List[Dict[str, Any]]:
        """Optional reasoning step for deliberative loops (ReAct).

        Returns a list of structured reasoning step dicts, e.g.:
          [{"check": "battery", "value": 0.45, "implication": "charging_required"}]

        Default is a no-op (empty list) so all existing representations
        continue to work without modification.

        Args:
            state: Encoded state dict from encode_observation.
            memory: Agent memory from the previous step.

        Returns:
            List of reasoning step dicts. Empty list if not overridden.
        """
        return []

    def get_rationale(self) -> Optional[str]:
        """Return a human-readable rationale for the last decision.

        Symbolic representations should always provide a rationale.
        Subsymbolic representations return ``None`` unless post-hoc
        explanation is available.

        Returns:
            Rationale string, or ``None`` if unavailable.
        """
        return None

    def get_metrics(self) -> Dict[str, float]:
        """Return representation-level metrics (e.g. inference time).

        Returns:
            Dictionary of metric name → value.
        """
        return {}

    def get_name(self) -> str:
        """Return a human-readable name for this representation."""
        return self.__class__.__name__

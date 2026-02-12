"""
Emergence Controller.

Controls whether decision-making logic is hand-designed or learned from
experience. Acts as a factory that returns a properly configured
representation module based on the experiment configuration.

Modes:
- **hand_designed**: Logic designed by human experts
  (rules, prompts, hand-crafted models).
- **learned**: Logic learned from training data
  (RL policies, learned heuristics, fine-tuned models).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from src.representation.base import Representation


# Registry of known representation classes, populated by ``register()``.
_REPRESENTATION_REGISTRY: Dict[str, Type[Representation]] = {}


def register(name: str) -> Any:
    """Decorator to register a representation class in the factory.

    Usage::

        @register("symbolic_rule_based")
        class RuleBasedRepresentation(Representation):
            ...

    Args:
        name: Unique identifier used in YAML configuration files.

    Returns:
        The original class, unmodified.
    """

    def _decorator(cls: Type[Representation]) -> Type[Representation]:
        _REPRESENTATION_REGISTRY[name] = cls
        return cls

    return _decorator


class EmergenceController:
    """Factory for creating representation modules based on emergence mode.

    The controller reads the experiment configuration and returns the
    appropriate representation instance — either a hand-designed variant
    or a learned variant.

    Attributes:
        config: Emergence-specific configuration from experiment YAML.
        mode: Emergence mode (``"hand_designed"`` or ``"learned"``).
    """

    VALID_MODES = {"hand_designed", "learned"}

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the emergence controller.

        Args:
            config: Emergence configuration section from experiment YAML.
                Must contain at least ``mode``.

        Raises:
            ValueError: If the mode is not recognised.
        """
        self.config = config
        self.mode: str = config.get("mode", "hand_designed")
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown emergence mode '{self.mode}'. "
                f"Valid modes: {self.VALID_MODES}"
            )

    def get_representation(
        self,
        repr_type: str,
        repr_config: Dict[str, Any] | None = None,
    ) -> Representation:
        """Create and return a configured representation instance.

        Args:
            repr_type: Registered name of the representation class
                (e.g. ``"symbolic_rule_based"``).
            repr_config: Optional extra configuration for the representation.

        Returns:
            An initialised :class:`Representation` instance.

        Raises:
            KeyError: If ``repr_type`` is not found in the registry.
        """
        if repr_type not in _REPRESENTATION_REGISTRY:
            available = ", ".join(sorted(_REPRESENTATION_REGISTRY)) or "(none registered)"
            raise KeyError(
                f"Representation '{repr_type}' not found in registry. "
                f"Available: {available}"
            )

        representation_cls = _REPRESENTATION_REGISTRY[repr_type]
        merged_config = {**self.config, **(repr_config or {})}
        return representation_cls(config=merged_config)

    @staticmethod
    def list_registered() -> list[str]:
        """Return names of all registered representation classes.

        Returns:
            Sorted list of registered representation names.
        """
        return sorted(_REPRESENTATION_REGISTRY)

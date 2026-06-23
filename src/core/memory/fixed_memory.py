"""
Fixed Memory — Unified memory design for all experiments.

All cognitive architecture variants receive the same memory structure so
that the experimental comparison is fair. Only the *representation* module
determines how memory contents are interpreted and used.

Memory slots:
- ``constellation_state``: Current state snapshot of all satellites.
- ``history``: Sliding window of past states (configurable depth).
- ``task_queue``: Currently active / pending tasks.
- ``task_history``: Completed tasks with outcomes.
- ``resource_budgets``: Remaining resource budgets per satellite.
- ``custom``: Scenario-specific additional data.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Deque, Dict, List, Optional

from src.core.memory.base import Memory


class FixedMemory(Memory):
    """Fixed memory implementation shared by all architecture variants.

    Attributes:
        config: Memory configuration from experiment YAML.
        history_depth: Number of past states to retain in the sliding window.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialise fixed memory.

        Args:
            config: Memory configuration. Recognised keys:
                - ``history_depth`` (int): Sliding-window size (default 50).
        """
        self.config = config or {}
        self.history_depth: int = self.config.get("history_depth", 50)

        # Memory slots
        self._constellation_state: Dict[str, Any] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.history_depth)
        self._task_queue: List[Dict[str, Any]] = []
        self._task_history: List[Dict[str, Any]] = []
        self._resource_budgets: Dict[str, Dict[str, float]] = {}
        self._custom: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Memory interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all memory slots to empty state."""
        self._constellation_state = {}
        self._history.clear()
        self._task_queue = []
        self._task_history = []
        self._resource_budgets = {}
        self._custom = {}

    def get_state(self) -> Dict[str, Any]:
        """Return full memory contents as a dictionary."""
        return {
            "constellation_state": deepcopy(self._constellation_state),
            "history": list(self._history),
            "task_queue": list(self._task_queue),
            "task_history": list(self._task_history),
            "resource_budgets": dict(self._resource_budgets),
            "custom": dict(self._custom),
        }

    def update(self, key: str, value: Any) -> None:
        """Update a named memory slot.

        Supported keys: ``constellation_state``, ``task_queue``,
        ``task_history``, ``resource_budgets``, ``custom``.
        Updating ``constellation_state`` also pushes the previous
        state into the history sliding window.

        Args:
            key: Memory slot name.
            value: Data to store.

        Raises:
            KeyError: If the key is not a recognised memory slot.
        """
        if key == "constellation_state":
            if self._constellation_state:
                self._history.append(deepcopy(self._constellation_state))
            self._constellation_state = value
        elif key == "task_queue":
            self._task_queue = value
        elif key == "task_history":
            self._task_history = value
        elif key == "resource_budgets":
            self._resource_budgets = value
        elif key == "custom":
            self._custom = value
        else:
            raise KeyError(
                f"Unknown memory slot '{key}'. "
                "Recognised: constellation_state, task_queue, task_history, "
                "resource_budgets, custom."
            )

    def query(self, key: str) -> Any:
        """Retrieve the contents of a named memory slot.

        Args:
            key: Memory slot name.

        Returns:
            Stored data, or ``None`` if the key is not recognised.
        """
        return self.get_state().get(key)

"""
Memory — Abstract Base Class.

Memory interface for the experimental framework. This is distinct from the
demo application's memory system in ``agent/memory/``. The experimental
memory is a *fixed* design shared identically across all cognitive
architecture variants to ensure fair comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Memory(ABC):
    """Abstract base class for the experimental memory system.

    The memory system is fixed across all experiments to ensure fair
    comparison. All architecture variants have access to the same
    information — only the representation determines how that
    information is used.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset memory to its initial state (start of episode)."""
        ...

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return the full memory state as a dictionary.

        Returns:
            Complete memory contents.
        """
        ...

    @abstractmethod
    def update(self, key: str, value: Any) -> None:
        """Update a memory entry.

        Args:
            key: Memory slot identifier.
            value: Data to store.
        """
        ...

    @abstractmethod
    def query(self, key: str) -> Any:
        """Retrieve a memory entry.

        Args:
            key: Memory slot identifier.

        Returns:
            Stored data, or ``None`` if not found.
        """
        ...

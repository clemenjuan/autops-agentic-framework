"""
Writable Memory — CoALA-style learning memory for agentic representations.

Extends FixedMemory with writable semantic and episodic stores to enable
CoALA-style learning (Sumers et al. 2024 [CoALA]). Used exclusively by
``writable_coala`` learned-emergence variants.

**Fairness note**: All hand-designed and non-CoALA learned variants use
``FixedMemory`` for fair comparison. This class is used ONLY when
``behaviour_config.mechanism = "writable_coala"`` is explicitly set.
The writable_coala experiment configs document this trade-off.

Memory architecture (CoALA §3):
- Working memory:   Per-step context from FixedMemory (inherited, unchanged)
- Episodic memory:  Episode-level trajectory summaries; written by agent via
                    ``write_episodic_entry()``; persists across episodes in run
- Semantic memory:  Domain-rule accretion; written by agent via
                    ``write_semantic_rule()``; persists across entire run
- Procedural:       Tool definitions (stateless YAML — unchanged per CLAUDE.md)
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from src.memory.fixed_memory import FixedMemory


class WritableMemory(FixedMemory):
    """Memory with writable semantic and episodic stores for CoALA learning.

    Inherits all FixedMemory slots (working/task/resource state). Adds:
    - ``semantic_store``: List of domain rules accreted across the run.
    - ``episodic_store``: Ring buffer of episode trajectory summaries.

    Args:
        config: Memory configuration. Recognised keys beyond FixedMemory:
            - ``episodic_max_size`` (int): Max episode summaries to keep (default 50).
            - ``memory_path`` (str): Path to persist memory state between runs.
              If provided, ``load()`` is called in ``__init__`` if the file exists.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._episodic_max_size: int = (config or {}).get("episodic_max_size", 50)
        self._memory_path: Optional[str] = (config or {}).get("memory_path")

        # Writable stores (persist across episodes within a run)
        self._semantic_store: List[Dict[str, Any]] = []
        self._episodic_store: Deque[Dict[str, Any]] = deque(
            maxlen=self._episodic_max_size
        )

        # Load persisted state if a path is configured and the file exists
        if self._memory_path and Path(self._memory_path).exists():
            self.load(self._memory_path)

    # ------------------------------------------------------------------
    # Semantic memory (domain rules, accreted across entire run)
    # ------------------------------------------------------------------

    def write_semantic_rule(
        self,
        rule_text: str,
        condition: str = "",
        action: str = "",
        provenance: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Append a domain rule to the semantic store.

        Args:
            rule_text: Human-readable rule description.
            condition: Trigger condition (e.g. "battery < 20%").
            action: Recommended action (e.g. "switch to charging").
            provenance: Optional metadata (episode_id, step, source).

        Returns:
            Confirmation message for the agent's tool result.
        """
        entry = {
            "rule": rule_text,
            "condition": condition,
            "action": action,
            "provenance": provenance or {},
        }
        self._semantic_store.append(entry)
        return f"Rule written to semantic memory (total rules: {len(self._semantic_store)})."

    def recall_semantic(self, query: str = "") -> List[Dict[str, Any]]:
        """Return semantic rules, optionally filtered by query string.

        Args:
            query: Substring filter applied to rule text (case-insensitive).
                   Empty string returns all rules.

        Returns:
            List of matching rule dicts.
        """
        if not query:
            return list(self._semantic_store)
        q = query.lower()
        return [r for r in self._semantic_store if q in r.get("rule", "").lower()]

    # ------------------------------------------------------------------
    # Episodic memory (trajectory summaries, across episodes within run)
    # ------------------------------------------------------------------

    def write_episodic_entry(
        self,
        summary: str,
        outcome: str = "",
        episode_id: Optional[int] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Append an episode trajectory summary to the episodic store.

        Args:
            summary: Summary of what happened this episode.
            outcome: Quantified outcome (e.g. "utility=0.72, anomalies=1").
            episode_id: Episode number (for traceability).
            provenance: Optional metadata dict.

        Returns:
            Confirmation message for the agent's tool result.
        """
        entry = {
            "summary": summary,
            "outcome": outcome,
            "episode_id": episode_id,
            "provenance": provenance or {},
        }
        self._episodic_store.append(entry)
        return (
            f"Episode entry written to episodic memory "
            f"(total entries: {len(self._episodic_store)})."
        )

    def recall_episodic(
        self, query: str = "", last_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Return recent episode summaries, optionally filtered.

        Args:
            query: Substring filter on summary text (case-insensitive).
            last_n: Maximum number of most-recent entries to return.

        Returns:
            List of matching episode dicts (most recent first).
        """
        entries = list(self._episodic_store)[-last_n:]
        entries.reverse()
        if not query:
            return entries
        q = query.lower()
        return [e for e in entries if q in e.get("summary", "").lower()]

    # ------------------------------------------------------------------
    # Extended get_state (for serialisation and introspection)
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return full memory contents including writable stores."""
        base = super().get_state()
        base["semantic_store"] = list(self._semantic_store)
        base["episodic_store"] = list(self._episodic_store)
        return base

    # ------------------------------------------------------------------
    # Persistence (between runs, NOT between episodes — use reset() for that)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist semantic and episodic stores to a JSON file.

        Note: Only the writable stores are persisted — the working/task
        memory is transient and reset between episodes.

        Args:
            path: Destination file path.
        """
        data = {
            "semantic_store": list(self._semantic_store),
            "episodic_store": list(self._episodic_store),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, path: str) -> None:
        """Load semantic and episodic stores from a JSON file.

        Args:
            path: Source file path.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._semantic_store = data.get("semantic_store", [])
        episodic = data.get("episodic_store", [])
        self._episodic_store = deque(episodic, maxlen=self._episodic_max_size)

    # ------------------------------------------------------------------
    # reset() — resets working/task memory but NOT the writable stores
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset transient memory slots (working/task/resource state).

        Semantic and episodic stores are deliberately NOT reset — they
        accumulate across episodes within a run to enable CoALA learning.
        Call ``clear_learned_state()`` to fully wipe the writable stores.
        """
        super().reset()
        # Intentionally NOT clearing _semantic_store or _episodic_store

    def clear_learned_state(self) -> None:
        """Wipe all accreted semantic rules and episodic summaries.

        Use this to start a fresh training run. Does not affect working memory.
        """
        self._semantic_store = []
        self._episodic_store.clear()

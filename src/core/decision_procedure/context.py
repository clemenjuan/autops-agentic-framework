"""
DecisionContext — Structured interface between decision loops and representations.

Every decision loop produces a DecisionContext; every representation consumes one.
This decouples the loop's enrichments (situation assessment, urgency, CBR cases,
LLM prompts, RL tensors, ...) from the raw encoded state, so representations can
opt into loop-specific data without breaking the base interface.

Examples:
  - SDA:   DecisionContext(state=encoded, loop_type="sda", enrichments={})
  - Hybrid/Agentic representations may attach prompt/tool metadata in enrichments.
  - RL representations may attach tensor/logit/value metadata in enrichments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DecisionContext:
    """Structured wrapper passed from decision loop to representation.

    Attributes:
        state: Raw encoded observation (always present, loop-agnostic).
        loop_type: Identifier for the producing loop. Current benchmark configs use "sda" only.
        memory: Agent memory reference (FixedMemory or similar).
        enrichments: Loop-specific data (orient assessment, LLM prompts, tensors).
        loop_metadata: Operational metadata (latency, iterations, timing).
    """

    state: Dict[str, Any] = field(default_factory=dict)
    loop_type: str = "sda"
    memory: Any = None
    enrichments: Dict[str, Any] = field(default_factory=dict)
    loop_metadata: Dict[str, Any] = field(default_factory=dict)

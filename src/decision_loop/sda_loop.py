"""
SDA (Sense-Decide-Act) Decision Loop.

Simplest possible decision loop: one pass through sense -> decide -> act
with no iteration, reflection, or memory update.
"""
from __future__ import annotations
import time
from typing import Any, Dict, Tuple
from src.decision_loop.base import DecisionLoop
from src.representation.base import Representation


class SDALoop(DecisionLoop):
    """Sense-Decide-Act: single-pass decision making."""

    def __init__(self, config: Dict[str, Any], representation: Representation) -> None:
        self.config = config
        self.representation = representation
        self._last_latency: float = 0.0
        self._total_steps: int = 0

    def process(
        self, observation: Any, memory: Any
    ) -> Tuple[Dict[str, Any], Any]:
        t0 = time.perf_counter()
        # Unwrap AgentObservation if needed
        raw_obs = observation
        if hasattr(observation, "local_state") and isinstance(observation.local_state, dict):
            raw_obs = observation.local_state.get("full_observation", observation)
        encoded = self.representation.encode_observation(raw_obs)
        action = self.representation.select_action(encoded, memory)
        self._last_latency = time.perf_counter() - t0
        self._total_steps += 1
        return action, memory

    def get_metrics(self) -> Dict[str, float]:
        return {
            "decision_latency_s": self._last_latency,
            "total_decisions": float(self._total_steps),
        }

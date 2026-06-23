"""
SDA (Sense-Decide-Act) Decision Loop.

Simplest possible decision loop: one pass through sense -> decide -> act
with no iteration, reflection, or memory update.

Grounded in the classical situated-action / reactive-agent tradition:
  - Agre, P. E. & Chapman, D. (1990), "What are plans for?", Robotics and
    Autonomous Systems 6(1-2):17-34 [Zotero: BERWVN2V]. Plan-as-communication
    view motivates treating each cycle as a fresh sensing-then-action step
    rather than execution of a pre-computed program.

The SDA variant is the fixed benchmark driver: no memory read, no orient step,
and no iteration.
"""
from __future__ import annotations
import time
from typing import Any, Dict, Tuple
from src.core.decision_procedure.base import DecisionProcedure
from src.core.decision_procedure.context import DecisionContext
from src.core.representation import Representation


class SDALoop(DecisionProcedure):
    """Sense-Decide-Act: single-pass decision making."""

    def __init__(self, config: Dict[str, Any], representation: Representation) -> None:
        self.config = config
        self.representation = representation
        self._last_latency: float = 0.0
        self._total_steps: int = 0
        self._last_has_rationale: bool = False

    def process(
        self, observation: Any, memory: Any
    ) -> Tuple[Dict[str, Any], Any]:
        t0 = time.perf_counter()
        # Unwrap AgentObservation if needed
        raw_obs = observation
        if hasattr(observation, "local_state") and isinstance(observation.local_state, dict):
            raw_obs = observation.local_state.get("full_observation", observation)
        encoded = self.representation.encode_observation(raw_obs)
        context = DecisionContext(
            state=encoded,
            loop_type="sda",
            memory=memory,
            enrichments={},
            loop_metadata={},
        )
        action = self.representation.select_action(context)
        self._last_latency = time.perf_counter() - t0
        self._total_steps += 1
        # Check if representation provides a rationale (explainability)
        self._last_has_rationale = (
            hasattr(self.representation, "get_rationale")
            and self.representation.get_rationale() is not None
        )
        return action, memory

    def get_metrics(self) -> Dict[str, Any]:
        rationale = ""
        if hasattr(self.representation, "get_rationale"):
            rationale = self.representation.get_rationale() or ""
        metrics = {
            "decision_latency_s": self._last_latency,
            "total_decisions": float(self._total_steps),
            "has_rationale": float(self._last_has_rationale),
            "rationale": rationale,
        }
        if hasattr(self.representation, "get_metrics"):
            for key, value in self.representation.get_metrics().items():
                if key not in metrics:
                    metrics[key] = value
        return metrics

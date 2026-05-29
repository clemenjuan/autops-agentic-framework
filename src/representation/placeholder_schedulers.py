"""
Placeholder schedule-producer representations for ground paradigms (AG/CG).

These exist to make the **non-symbolic** ground cells coherent. The ground
paradigms (AutonomousGround / ConventionalGround) drive between-pass behavior
from a ``schedule`` emitted in the agent's action. Only the symbolic schedule
planners emit that key; ``subsymbolic_eventsat`` / ``llm_eventsat`` /
``agentic_eventsat`` do not. Pairing those per-step controllers with a ground
paradigm therefore produced a degenerate trajectory (charging every inter-pass
step — the representation had almost no influence). See the ops-paradigm review.

Each class here emits a real ``schedule`` by delegating to the symbolic greedy
planner (``ScheduleBasedEventSat``). **They are PLACEHOLDERS**: the schedule is
NOT produced by the family's actual policy. They mark the drop-in point for the
future "learned scheduling" research line (RL vs LLM schedule producers):

  - ``subsymbolic_scheduler_eventsat`` → TODO(P3): PPO-trained schedule producer
  - ``llm_scheduler_eventsat``         → TODO(P3): LLM-generated schedule
  - ``agentic_scheduler_eventsat``     → TODO(P3): agentic (tool-using) planner

``is_placeholder = True`` is recorded in results metadata
(see ``ExperimentRunner._compile_results``) so analysis can exclude these cells
from headline comparisons until the real producers land.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from src.emergence.controller import register
from src.representation.schedule_based_eventsat import ScheduleBasedEventSat

if TYPE_CHECKING:
    from src.decision_loop.context import DecisionContext


class _PlaceholderScheduler(ScheduleBasedEventSat):
    """Base placeholder: symbolic greedy planner standing in for a learned one.

    Inherits the full schedule-generation logic from ``ScheduleBasedEventSat``
    and only tags the rationale and exposes ``is_placeholder`` so the cell is
    clearly distinguishable in traces and results.
    """

    is_placeholder: bool = True
    _family: str = "placeholder"

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        action = super().select_action(context)
        if self._last_rationale:
            self._last_rationale = (
                f"[PLACEHOLDER {self._family} scheduler — symbolic planner stand-in] "
                f"{self._last_rationale}"
            )
        return action


@register("subsymbolic_scheduler_eventsat")
class SubsymbolicSchedulerEventSat(_PlaceholderScheduler):
    """Placeholder RL scheduler. TODO(P3): replace with a PPO-trained schedule producer."""

    _family = "subsymbolic"


@register("llm_scheduler_eventsat")
class LLMSchedulerEventSat(_PlaceholderScheduler):
    """Placeholder LLM scheduler. TODO(P3): replace with an LLM-generated schedule."""

    _family = "llm"


@register("agentic_scheduler_eventsat")
class AgenticSchedulerEventSat(_PlaceholderScheduler):
    """Placeholder agentic scheduler. TODO(P3): replace with a tool-using planner."""

    _family = "agentic"

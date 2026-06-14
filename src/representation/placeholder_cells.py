"""
Placeholder representations for the two framework cells without a real
implementation yet: ``hrl`` (hybrid RL + symbolic) and ``llm-a`` (pure LLM
agentic, no symbolic layer). See the 7-cell representation table in
``docs/morphological_matrix.md`` §2.

Both cells are expressible in the vocabulary and validated, but no runnable
policy exists. Each routes to a symbolic stand-in flagged
``is_placeholder = True`` (recorded in results metadata by
``ExperimentRunner._compile_results``) so runs are clearly marked and analysis
can exclude them until the real cores land:

  - onboard slot (AO/AH) → per-step symbolic rules (``RuleBasedEventSat``)
  - ground slot (AG/CG)  → greedy symbolic schedule (``ScheduleBasedEventSat``)

TODO: replace with the real cores —
  - ``hrl``  : an RL policy gated by symbolic safety rules (hybrid RL+symbolic)
  - ``llm-a``: an LLM tool-using loop without the symbolic hybrid layer
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from src.behaviour.controller import register
from src.representation.rule_based_eventsat import RuleBasedEventSat
from src.representation.schedule_based_eventsat import ScheduleBasedEventSat

if TYPE_CHECKING:
    from src.decision_procedure.context import DecisionContext


class _OnboardCellPlaceholder(RuleBasedEventSat):
    """Onboard placeholder: symbolic per-step rules standing in for an
    unimplemented cell. Tags the rationale and exposes ``is_placeholder``."""

    is_placeholder: bool = True
    _cell: str = "placeholder"

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        action = super().select_action(context)
        if self._last_rationale:
            self._last_rationale = (
                f"[PLACEHOLDER {self._cell} cell — symbolic onboard stand-in] "
                f"{self._last_rationale}"
            )
        return action


class _GroundCellPlaceholder(ScheduleBasedEventSat):
    """Ground placeholder: symbolic greedy schedule standing in for an
    unimplemented cell. Tags the rationale and exposes ``is_placeholder``."""

    is_placeholder: bool = True
    _cell: str = "placeholder"

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        action = super().select_action(context)
        if self._last_rationale:
            self._last_rationale = (
                f"[PLACEHOLDER {self._cell} cell — symbolic schedule stand-in] "
                f"{self._last_rationale}"
            )
        return action


@register("hrl_onboard_eventsat")
class HrlOnboardPlaceholder(_OnboardCellPlaceholder):
    """Placeholder hrl onboard core. TODO: RL policy gated by symbolic safety."""

    _cell = "hrl"


@register("hrl_scheduler_eventsat")
class HrlSchedulerPlaceholder(_GroundCellPlaceholder):
    """Placeholder hrl ground planner. TODO: hybrid-RL schedule producer."""

    _cell = "hrl"


@register("llm_agentic_onboard_eventsat")
class LlmAgenticOnboardPlaceholder(_OnboardCellPlaceholder):
    """Placeholder llm-a onboard core. TODO: pure-LLM agentic loop (no symbolic)."""

    _cell = "llm-a"


@register("llm_agentic_scheduler_eventsat")
class LlmAgenticSchedulerPlaceholder(_GroundCellPlaceholder):
    """Placeholder llm-a ground planner. TODO: pure-LLM agentic schedule producer."""

    _cell = "llm-a"

"""
Single-shot LLM ground scheduler for EventSat (the `hllm-s` cell on the ground).

At each ground contact, the LLM is called **once** on fresh telemetry to generate
the inter-pass schedule — a list of ``[mode, steps]`` segments the satellite
executes autonomously until the next contact. This is the LLM analogue of
``ScheduleBasedEventSat`` (the symbolic greedy planner): same per-pass control
flow and ``{"mode": "communication", "schedule": [...]}`` output contract, but the
schedule is produced by the LLM (subsymbolic core) with a symbolic safety layer
(hybrid → ``hllm-s``).

Substrate integrity (user decision 2026-06-11): if the LLM produces no valid
schedule after retries, the episode FAILS — no silent symbolic fallback.

Papers: Rodriguez-Fernandez et al. (2024) — LLM prompt design for sat ops.

Registered as "llm_scheduler_eventsat" (replaces the former placeholder).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.behaviour.controller import register
from src.representation.base import Representation
from src.representation.llm_client import LLMClient
from src.representation.llm_eventsat import VALID_MODES
from src.representation.llm_prompts import SCHEDULE_SYSTEM_PROMPT, format_schedule_prompt
from src.representation.schedule_based_eventsat import ScheduleBasedEventSat, _merge_schedule

if TYPE_CHECKING:
    from src.decision_procedure.context import DecisionContext

logger = logging.getLogger(__name__)

# Modes valid in a between-pass schedule (no ground link → no communication).
_SCHEDULABLE_MODES = VALID_MODES - {"communication"}


@register("llm_scheduler_eventsat")
class LLMSchedulerEventSat(Representation):
    """LLM (hybrid + symbolic) single-shot ground planner — the hllm-s ground core."""

    is_placeholder: bool = False
    MAX_RETRIES: int = 2

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._client = LLMClient(self.config)
        self._last_rationale: Optional[str] = None
        self._schedule_entries: int = 0
        self._staleness_threshold: int = self.config.get("staleness_threshold", 5)
        # Needed by the borrowed ScheduleBasedEventSat.encode_observation (state default).
        self._daily_downlink_budget_mb: float = self.config.get("daily_downlink_budget_mb", 27.0)
        self._schedule_generated_this_pass: bool = False
        self._last_pass_active: bool = False

    # encode_observation: identical state needs as the symbolic planner
    # (gap, staleness, pipeline pools) — reuse it directly.
    encode_observation = ScheduleBasedEventSat.encode_observation

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        """Per-pass flow mirroring ScheduleBasedEventSat; the LLM produces the schedule."""
        state = context.state
        if not state:
            self._last_rationale = "No state; defaulting to charging."
            return {"eventsat_0": {"mode": "charging"}}

        pass_active = state.get("ground_pass_active", False)
        staleness = state.get("staleness_steps", 999)

        if not pass_active and self._last_pass_active:
            self._schedule_generated_this_pass = False
        self._last_pass_active = pass_active

        if not pass_active:
            self._last_rationale = "Between passes; schedule executing autonomously."
            return {"eventsat_0": {"mode": "charging"}}

        # During a pass: downlink stale HK first (new contact), then plan once on fresh data.
        if staleness > self._staleness_threshold:
            self._schedule_generated_this_pass = False
            self._last_rationale = (
                f"Pass active but telemetry stale ({staleness} steps); communicating for fresh HK."
            )
            return {"eventsat_0": {"mode": "communication"}}

        if self._schedule_generated_this_pass:
            self._last_rationale = "Schedule uploaded; continuing communication for data downlink."
            return {"eventsat_0": {"mode": "communication"}}

        gap_steps = int(state.get("estimated_gap_steps", 93))
        schedule = self._generate_schedule_llm(state, gap_steps)  # raises on integrity failure
        self._schedule_generated_this_pass = True
        self._schedule_entries = len(schedule)
        return {"eventsat_0": {"mode": "communication", "schedule": schedule}}

    def get_rationale(self) -> Optional[str]:
        return self._last_rationale

    def get_metrics(self) -> Dict[str, float]:
        m = self._client.get_metrics()
        m["llm_schedule_entries"] = float(self._schedule_entries)
        return m

    def get_name(self) -> str:
        return "LLMSchedulerEventSat"

    # ------------------------------------------------------------------
    # LLM schedule generation
    # ------------------------------------------------------------------

    def _generate_schedule_llm(
        self, state: Dict[str, Any], gap_steps: int
    ) -> List[Tuple[str, int]]:
        """One LLM call → validated schedule covering ~gap_steps. Raises on no valid output."""
        from src.representation.llm_eventsat import LLMEventSat  # reuse fence-stripping parser

        user_prompt = format_schedule_prompt(state, gap_steps)
        schedule: Optional[List[Tuple[str, int]]] = None
        retries = 0
        for attempt in range(1 + self.MAX_RETRIES):
            try:
                raw = self._client.generate(
                    system_prompt=SCHEDULE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    json_mode=True,
                )
                parsed = LLMEventSat._parse_response(self, raw)
                candidate = self._validate_schedule(parsed.get("schedule"), gap_steps)
                if candidate:
                    schedule = candidate
                    self._last_rationale = (
                        f"LLM schedule ({len(candidate)} segments) for {gap_steps}-step gap: "
                        f"{parsed.get('rationale', '')}"
                    )
                    break
                logger.warning("LLM schedule invalid/empty (attempt %d/%d)", attempt + 1, 1 + self.MAX_RETRIES)
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM schedule call failed (attempt %d/%d): %s", attempt + 1, 1 + self.MAX_RETRIES, e)
            retries += 1

        if schedule is None:
            raise RuntimeError(
                f"LLM scheduler integrity violation: no valid schedule after {retries} "
                f"retries — failing the episode instead of substituting a symbolic plan. "
                f"Check OLLAMA_HOST / model availability."
            )
        return schedule

    def _validate_schedule(
        self, raw_schedule: Any, gap_steps: int
    ) -> Optional[List[Tuple[str, int]]]:
        """Symbolic grounding: keep valid [mode, steps] segments, clamp total to the gap.

        Drops communication (no link between passes) and unknown modes; pads the tail
        with charging if short. Returns None if nothing valid was produced.
        """
        if not isinstance(raw_schedule, list):
            return None
        out: List[Tuple[str, int]] = []
        total = 0
        for entry in raw_schedule:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                mode, steps = entry[0], entry[1]
            elif isinstance(entry, dict):
                mode, steps = entry.get("mode"), entry.get("steps", entry.get("duration"))
            else:
                continue
            if mode not in _SCHEDULABLE_MODES:
                continue
            try:
                steps = int(steps)
            except (TypeError, ValueError):
                continue
            if steps < 1:
                continue
            if total + steps > gap_steps:
                steps = gap_steps - total
            if steps < 1:
                break
            out.append((mode, steps))
            total += steps
            if total >= gap_steps:
                break
        if not out:
            return None
        if total < gap_steps:
            out.append(("charging", gap_steps - total))
        return _merge_schedule(out)

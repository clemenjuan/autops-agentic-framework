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

# Operational (battery-consuming) modes gated by the operations SoC floor in the
# hybrid SAFETY shield. Charging/safe are never vetoed.
_OPERATIONAL_MODES = {
    "payload_observe", "payload_compress", "payload_detect", "payload_send",
}


@register("llm_scheduler_eventsat")
class LLMSchedulerEventSat(Representation):
    """LLM (hybrid + symbolic) single-shot ground planner — the hllm-s ground core."""

    is_placeholder: bool = False
    MAX_RETRIES: int = 2
    # Hybrid (hllm-s): apply the symbolic SAFETY + format layer to the LLM schedule —
    # drop non-schedulable modes, clamp to the gap, pad with charging, and veto
    # operational blocks that would run in a critical state (battery below the
    # operations floor, OBC storage full → charging). It is SAFETY grounding, not
    # behaviour: it never decides how much to observe (that is the LLM's job). The
    # pure-LLM cell (llm-s) overrides this to False — see LLMSingleSchedulerEventSat.
    _symbolic_grounding: bool = True
    _cell: str = "hllm-s"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._client = LLMClient(self.config)
        self._last_rationale: Optional[str] = None
        self._schedule_entries: int = 0
        self._staleness_threshold: int = self.config.get("staleness_threshold", 5)
        # Needed by the borrowed ScheduleBasedEventSat.encode_observation (state default).
        self._daily_downlink_budget_mb: float = self.config.get("daily_downlink_budget_mb", 27.0)
        self._settling_time_steps: int = self.config.get("settling_time_steps", 2)
        self._schedule_generated_this_pass: bool = False
        self._last_pass_active: bool = False
        # Symbolic SAFETY model for the hybrid grounding shield (hllm-s only): reuses
        # the symbolic cores' calibrated battery/storage thresholds + SoC model from
        # the scenario physics in `config`. Not built for pure llm-s (no shield), which
        # is not given the physics block.
        self._safety_model = (
            ScheduleBasedEventSat(config) if self._symbolic_grounding else None
        )

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
                candidate = self._validate_schedule(parsed.get("schedule"), gap_steps, state)
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
        self, raw_schedule: Any, gap_steps: int, state: Dict[str, Any] | None = None
    ) -> Optional[List[Tuple[str, int]]]:
        """Parse the LLM schedule into ``[mode, steps]`` segments.

        Hybrid (hllm-s, ``_symbolic_grounding=True``): apply the symbolic SAFETY +
        format layer — drop non-schedulable modes (communication / unknown), clamp the
        total to the gap, pad the tail with charging, then run ``_apply_safety_shield``
        (veto operational blocks in a critical battery/storage state). The layer is
        SAFETY/feasibility only: it never caps how much the LLM chooses to observe.
        Pure LLM (llm-s, grounding off): keep any valid-enum mode with steps>=1 as-is,
        no shield/clamp/pad — safety lives in the prompt and the environment enforces it
        at execution. Returns None if nothing parseable was produced.
        """
        if not isinstance(raw_schedule, list):
            return None
        allowed = _SCHEDULABLE_MODES if self._symbolic_grounding else VALID_MODES
        out: List[Tuple[str, int]] = []
        total = 0
        for entry in raw_schedule:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                mode, steps = entry[0], entry[1]
            elif isinstance(entry, dict):
                mode, steps = entry.get("mode"), entry.get("steps", entry.get("duration"))
            else:
                continue
            if mode not in allowed:
                continue
            try:
                steps = int(steps)
            except (TypeError, ValueError):
                continue
            if steps < 1:
                continue
            if self._symbolic_grounding and total + steps > gap_steps:
                steps = gap_steps - total
            if steps < 1:
                break
            out.append((mode, steps))
            total += steps
            if self._symbolic_grounding and total >= gap_steps:
                break
        if not out:
            return None
        if self._symbolic_grounding and total < gap_steps:
            out.append(("charging", gap_steps - total))
        if self._symbolic_grounding:
            out = self._apply_safety_shield(out, state or {})
        return _merge_schedule(out)

    def _apply_safety_shield(
        self, schedule: List[Tuple[str, int]], state: Dict[str, Any]
    ) -> List[Tuple[str, int]]:
        """Symbolic SAFETY grounding — the critical-situation rules the symbolic cores
        use, applied to the LLM's schedule. This is NOT behaviour: it never decides how
        much to observe or charge; it only replaces an operational block that would run
        in a critical state with 'charging' (the environment's own safe fallback):

          * battery — an operational mode below the operations SoC floor (forward-
            simulated SoC) → charging.
          * memory  — payload_observe while OBC storage is critically full → charging.

        Anomaly-forced safe mode is stochastic/future and is enforced by the
        environment at execution; a between-pass plan cannot pre-empt it.
        """
        sm = self._safety_model
        soc = float(state.get("battery_soc", 0.5) or 0.5)
        obc_mb = float(state.get("obc_data_mb", 0.0) or 0.0)
        obc_critical = obc_mb >= sm._obc_capacity_mb * 0.8
        floor = sm._min_soc_for_operations
        shielded: List[Tuple[str, int]] = []
        for mode, steps in schedule:
            out_mode = mode
            if mode in _OPERATIONAL_MODES:
                if soc < floor:                                  # battery-critical
                    out_mode = "charging"
                elif mode == "payload_observe" and obc_critical:  # memory-critical
                    out_mode = "charging"
            for _ in range(int(steps)):
                soc = max(0.0, min(1.0, soc + sm._soc_delta_per_step(out_mode)))
            shielded.append((out_mode, int(steps)))
        return shielded


@register("llm_single_scheduler_eventsat")
class LLMSingleSchedulerEventSat(LLMSchedulerEventSat):
    """Pure single-shot LLM ground planner — the llm-s ground core.

    Same LLM call/prompt as hllm-s, but WITHOUT the symbolic safety/format grounding:
    the schedule is taken as the model produced it (any valid-enum mode, steps≥1, no
    clamp/pad). Safety is described in the prompt only; the environment still enforces
    its own hard limits (anomaly→safe, SoC/comm gating) at execution. The hllm-s vs
    llm-s comparison isolates the value of the symbolic safety layer.
    """

    _symbolic_grounding: bool = False
    _cell: str = "llm-s"

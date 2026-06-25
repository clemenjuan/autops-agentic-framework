"""
Agentic LLM ground scheduler for EventSat (the `hllm-a` / `llm-a` cells on the ground).

This is the **agentic** analogue of the single-shot LLM ground scheduler
(``llm_scheduler_eventsat.py``): at each ground contact the LLM produces the
inter-pass schedule — a list of ``[mode, steps]`` segments executed autonomously
until the next contact — but here via a CoALA-style Plan-Tool-Reflect-Decide loop
(Sumers et al. 2024) rather than a single call. The loop reuses the same domain
tools as the per-step agentic core (``agentic_tools.py``); its terminal DECIDE step
emits a whole-pass schedule instead of a single mode.

It fills the gap noted in ``docs/implementations.md`` (Phase 4.e): the single-shot
LLM schedule producers were real (``llm_scheduler_eventsat`` /
``llm_single_scheduler_eventsat``) but the **agentic** schedule producers were
documented placeholders (symbolic greedy stand-ins). This module replaces those
placeholders with real cores, so the AG/AH ground cells ``hllm-a`` and ``llm-a``
exercise genuine agentic reasoning.

Two cells, distinguished — exactly as ``hllm-s`` vs ``llm-s`` — by the symbolic
SAFETY layer only (user decision 2026-06-22):

- ``hllm-a`` (``AgenticSchedulerEventSat``, hybrid LLM + symbolic): keeps the CoALA
  tools AND the symbolic safety/format grounding on the emitted schedule
  (``_symbolic_grounding=True``, inherited ``_apply_safety_shield``).
- ``llm-a`` (``LLMAgenticSchedulerEventSat``, pure LLM): same CoALA loop and tools,
  but the symbolic safety shield / clamp / pad is OFF (``_symbolic_grounding=False``);
  the schedule is taken as the model produced it (safety lives in the prompt and the
  environment enforces its own hard limits at execution). The CoALA tools are
  information-gathering *external actions* (CoALA §3), so they are part of the agentic
  action space, not the symbolic substrate — hence kept for both cells. The hllm-a vs
  llm-a comparison isolates exactly the symbolic safety layer, mirroring hllm-s↔llm-s.

Substrate integrity (user decision 2026-06-11): if the loop yields no valid schedule
after retries, the episode FAILS — no silent symbolic fallback.

Papers:
- Sumers et al. (2024) [CoALA] — agentic architecture, tool use, action decomposition.
- Bounded agent loop with forced answer extraction.
- Rodriguez-Fernandez et al. (2024) — LLM prompt design for satellite operations.

Registered as "agentic_scheduler_eventsat" (hllm-a) and
"llm_agentic_scheduler_eventsat" (llm-a) — replacing the former placeholders.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.core.behaviour.controller import register
from src.eventsat.agentic import parse_agentic_json
from src.eventsat.agentic_prompts import (
    AGENTIC_SCHEDULE_SYSTEM_PROMPT,
    format_forced_schedule_prompt,
    format_schedule_planning_prompt,
    format_schedule_tool_result_prompt,
)
from src.eventsat.agentic_tools import (
    SCHEDULE_TOOL_NAMES,
    execute_tool,
    get_tool_schemas,
)
from src.eventsat.llm_scheduler import LLMSchedulerEventSat

if TYPE_CHECKING:
    from src.core.decision_procedure.context import DecisionContext  # noqa: F401

logger = logging.getLogger(__name__)


@register("agentic_scheduler_eventsat")
class AgenticSchedulerEventSat(LLMSchedulerEventSat):
    """Agentic (CoALA) hybrid LLM ground planner — the hllm-a ground core.

    Inherits the per-pass control flow, ``encode_observation``, schedule validation
    and the symbolic safety shield from ``LLMSchedulerEventSat`` (the hllm-s core);
    overrides only schedule *generation* to run a Plan-Tool-Reflect-Decide loop whose
    terminal step emits the whole-pass schedule.
    """

    is_placeholder: bool = False
    _cell: str = "hllm-a"
    # Hybrid (hllm-a): apply the symbolic SAFETY + format layer to the emitted
    # schedule (drop communication, clamp to the gap, pad with charging, veto
    # operational blocks in a critical state). llm-a overrides this to False.
    _symbolic_grounding: bool = True

    # CoALA loop budget per schedule decision. Matches the per-step core's
    # default: with the echo tools folded into the planning prompt and only the
    # what-if tools advertised (SCHEDULE_TOOL_NAMES), one verify call is usually
    # enough, so 3 bounds per-pass decision latency (worst case ~4 LLM calls
    # incl. the forced Decide) — fast enough not to miss the contact window.
    DEFAULT_MAX_STEPS: int = 3

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)  # builds _client, safety model, pass-flow state
        cfg = config or {}
        self._max_agentic_steps: int = cfg.get("max_agentic_steps", self.DEFAULT_MAX_STEPS)
        self._system_prompt: str = AGENTIC_SCHEDULE_SYSTEM_PROMPT
        self._tool_schemas = get_tool_schemas(tool_names=SCHEDULE_TOOL_NAMES)
        # Agentic metrics
        self._total_tool_calls: int = 0
        self._total_agentic_steps: int = 0
        self._total_decisions: int = 0
        self._tool_call_histogram: Dict[str, int] = {}
        self._last_raw_responses: List[str] = []
        # Per-decision wall-clock latency (fresh telemetry → emitted schedule).
        # The operational figure of merit: the plan must be ready inside the
        # contact window. Cache hits replay at ~0 s (M-07), so meaningful only on
        # fresh states — i.e. a fresh experiment run.
        self._total_decision_latency_s: float = 0.0
        self._max_decision_latency_s: float = 0.0

    # ------------------------------------------------------------------
    # Schedule generation — agentic loop (overrides the single-shot call)
    # ------------------------------------------------------------------

    def _generate_schedule_llm(
        self, state: Dict[str, Any], gap_steps: int
    ) -> List[Tuple[str, int]]:
        """Run the CoALA loop → validated schedule covering ~gap_steps.

        Retries the whole loop up to ``MAX_RETRIES`` on an invalid/empty schedule;
        raises on persistent failure (substrate integrity — no symbolic substitution).
        """
        schedule: Optional[List[Tuple[str, int]]] = None
        retries = 0
        for attempt in range(1 + self.MAX_RETRIES):
            try:
                raw_schedule = self._run_agentic_schedule_loop(state, gap_steps)
                candidate = self._validate_schedule(raw_schedule, gap_steps, state)
                if candidate:
                    schedule = candidate
                    break
                logger.warning(
                    "Agentic schedule invalid/empty (attempt %d/%d)",
                    attempt + 1, 1 + self.MAX_RETRIES,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Agentic schedule loop failed (attempt %d/%d): %s",
                    attempt + 1, 1 + self.MAX_RETRIES, e,
                )
            retries += 1

        if schedule is None:
            raise RuntimeError(
                f"Agentic scheduler integrity violation: no valid schedule after "
                f"{retries} retries — failing the episode instead of substituting a "
                f"symbolic plan. Check OLLAMA_HOST / model availability."
            )
        return schedule

    def _run_agentic_schedule_loop(
        self, state: Dict[str, Any], gap_steps: int
    ) -> Any:
        """Execute the Plan-Tool-Reflect-Decide cycle; return the raw schedule.

        The ground planner reasons on fresh telemetry only (no episodic memory, like
        the single-shot scheduler); ``recall_history`` degrades gracefully on ``None``.
        """
        decision_t0 = time.perf_counter()
        accumulated_context: List[Dict[str, Any]] = []
        raw_responses: List[str] = []
        remaining_budget = self._max_agentic_steps
        schedule: Any = None
        rationale: Optional[str] = None
        steps_taken = 0
        memory = None

        # Step 1: PLAN
        try:
            raw = self._client.generate(
                system_prompt=self._system_prompt,
                user_prompt=format_schedule_planning_prompt(state, gap_steps),
                json_mode=True,
            )
            raw_responses.append(raw)
            parsed = parse_agentic_json(raw)
            steps_taken += 1
            remaining_budget -= 1

            schedule, rationale = self._extract_schedule(parsed)
            if schedule is not None:
                accumulated_context.append({
                    "step": "plan_decide",
                    "content": parsed.get("plan", parsed.get("reflection", "")),
                })
            else:
                accumulated_context.append({"step": "plan", "content": parsed.get("plan", "")})

                # Steps 2..N: TOOL-REFLECT loop
                tool_call = parsed.get("tool_call")
                while remaining_budget > 0 and tool_call and schedule is None:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})

                    tool_result = execute_tool(tool_name, tool_args, state, memory)
                    self._total_tool_calls += 1
                    self._tool_call_histogram[tool_name] = (
                        self._tool_call_histogram.get(tool_name, 0) + 1
                    )
                    accumulated_context.append({
                        "step": "tool", "name": tool_name, "result": tool_result,
                    })

                    try:
                        raw = self._client.generate(
                            system_prompt=self._system_prompt,
                            user_prompt=format_schedule_tool_result_prompt(
                                tool_name, tool_result, accumulated_context, gap_steps
                            ),
                            json_mode=True,
                        )
                        raw_responses.append(raw)
                        parsed = parse_agentic_json(raw)
                        steps_taken += 1
                        remaining_budget -= 1
                        accumulated_context.append({
                            "step": "reflect", "content": parsed.get("reflection", ""),
                        })
                        sched_candidate, rat = self._extract_schedule(parsed)
                        if sched_candidate is not None:
                            schedule, rationale = sched_candidate, rat
                        else:
                            tool_call = parsed.get("tool_call")
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Agentic schedule reflect step failed: %s", e)
                        accumulated_context.append({"step": "reflect_error", "content": str(e)})
                        break
        except Exception as e:  # noqa: BLE001
            logger.warning("Agentic schedule plan step failed: %s", e)
            accumulated_context.append({"step": "plan_error", "content": str(e)})

        # DECIDE (forced terminal step): the loop can exhaust its tool budget or stall
        # on a reflection carrying neither a decision nor a tool_call. Close the cycle
        # with one schedule-only call — no tool option offered.
        if not schedule:
            try:
                raw = self._client.generate(
                    system_prompt=self._system_prompt,
                    user_prompt=format_forced_schedule_prompt(accumulated_context, gap_steps),
                    json_mode=True,
                )
                raw_responses.append(raw)
                parsed = parse_agentic_json(raw)
                steps_taken += 1
                schedule, rationale = self._extract_schedule(parsed)
                accumulated_context.append({"step": "forced_decide", "content": rationale or ""})
            except Exception as e:  # noqa: BLE001
                logger.warning("Agentic schedule forced-decide step failed: %s", e)

        # Metrics + rationale (validation/grounding happen in the caller)
        self._total_agentic_steps += steps_taken
        self._total_decisions += 1
        decision_latency = time.perf_counter() - decision_t0
        self._total_decision_latency_s += decision_latency
        self._max_decision_latency_s = max(self._max_decision_latency_s, decision_latency)
        self._last_raw_responses = raw_responses
        self._last_rationale = (
            f"Agentic schedule [{self._summarize_chain(accumulated_context)}]: "
            f"{rationale or ''}"
        )
        return schedule

    @staticmethod
    def _extract_schedule(parsed: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
        """Pull (schedule, rationale) from a parsed response.

        Accepts the protocol form ``{"decision": {"schedule": [...], "rationale": ...}}``
        and the flattened form ``{"schedule": [...], "rationale": ...}`` that reasoning
        models (and the mock client) often emit. Returns (None, None) when neither is
        present (i.e. the step is a plan/tool_call, not a decision).
        """
        decision = parsed.get("decision")
        if isinstance(decision, dict) and decision.get("schedule") is not None:
            return decision.get("schedule"), decision.get("rationale", "")
        if parsed.get("schedule") is not None:
            return parsed.get("schedule"), parsed.get("rationale", "")
        return None, None

    @staticmethod
    def _summarize_chain(accumulated_context: List[Dict[str, Any]]) -> str:
        """One-line Plan→Tool→Reflect→Decide trace for the rationale string."""
        parts: List[str] = []
        for entry in accumulated_context:
            step_type = entry.get("step", "")
            if step_type in ("plan", "plan_decide"):
                parts.append(f"Plan: {entry.get('content', '')[:80]}")
            elif step_type == "tool":
                parts.append(f"Tool({entry.get('name', '?')})")
            elif step_type == "reflect":
                parts.append(f"Reflect: {entry.get('content', '')[:80]}")
            elif step_type == "forced_decide":
                parts.append(f"Decide: {entry.get('content', '')[:80]}")
        return " → ".join(parts) if parts else "direct"

    # ------------------------------------------------------------------
    # Metrics / identity
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, float]:
        """LLM client metrics + schedule + agentic counters."""
        m = super().get_metrics()  # llm_* + llm_schedule_entries
        m["agentic_total_tool_calls"] = float(self._total_tool_calls)
        m["agentic_total_decisions"] = float(self._total_decisions)
        avg_steps = (
            self._total_agentic_steps / self._total_decisions
            if self._total_decisions > 0 else 0.0
        )
        m["agentic_avg_steps_per_decision"] = round(avg_steps, 2)
        mean_latency = (
            self._total_decision_latency_s / self._total_decisions
            if self._total_decisions > 0 else 0.0
        )
        m["agentic_mean_decision_latency_s"] = round(mean_latency, 3)
        m["agentic_max_decision_latency_s"] = round(self._max_decision_latency_s, 3)
        m["agentic_total_decision_latency_s"] = round(self._total_decision_latency_s, 3)
        for tool_name, count in self._tool_call_histogram.items():
            m[f"agentic_tool_{tool_name}"] = float(count)
        return m

    def get_name(self) -> str:
        return "AgenticSchedulerEventSat"


@register("llm_agentic_scheduler_eventsat")
class LLMAgenticSchedulerEventSat(AgenticSchedulerEventSat):
    """Pure-LLM agentic ground planner — the llm-a ground core.

    Same CoALA loop, tools and prompt as hllm-a, but WITHOUT the symbolic safety/
    format grounding on the emitted schedule (``_symbolic_grounding=False``): the
    schedule is taken as the model produced it (any valid-enum mode, steps≥1, no
    clamp/pad/shield). Safety is described in the prompt only; the environment still
    enforces its own hard limits (anomaly→safe, SoC/comm gating) at execution. The
    hllm-a vs llm-a comparison isolates the value of the symbolic safety layer —
    the agentic analogue of the llm-s↔hllm-s ablation.
    """

    _symbolic_grounding: bool = False
    _cell: str = "llm-a"

    def get_name(self) -> str:
        return "LLMAgenticSchedulerEventSat"

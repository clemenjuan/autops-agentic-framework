"""
Agentic Hybrid Representation for EventSat (CoALA Architecture).

CoALA-style (Sumers et al. 2024) multi-step agentic representation that uses
an LLM-orchestrated Plan-Tool-Reflect-Decide loop for mode selection. The LLM
queries domain-specific tools (battery, pass, pipeline, constraints, history,
plan evaluation) before committing to a mode, enabling structured reasoning
with information gathering — unlike the single-shot llm_eventsat.

Memory architecture (mapped to FixedMemory without modification):
- Working memory: DecisionContext.state + tool results accumulated in-loop
- Episodic memory: FixedMemory.history + task_history (via recall_history tool)
- Semantic memory: Domain rules hardcoded in AGENTIC_SYSTEM_PROMPT
- Procedural memory: Tool definitions in TOOL_SCHEMAS

Papers:
- Sumers et al. (2024) [CoALA] — 4-memory architecture, action decomposition
- Sapkota et al. (2026) — agentic satellite operations
- Li (2025) — tool-augmented AI agents for satellite operations
- Rodriguez-Fernandez et al. (2024) — LLM prompt design for sat ops

Registered as "agentic_eventsat" in the emergence controller.
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.emergence.controller import register
from src.representation.base import Representation
from src.representation.llm_client import LLMClient
from src.representation.agentic_prompts import (
    AGENTIC_SYSTEM_PROMPT,
    format_agentic_reasoning_prompt,
    format_planning_prompt,
    format_tool_result_prompt,
)
from src.representation.agentic_tools import (
    VALID_MODES,
    execute_tool,
)

if TYPE_CHECKING:
    from src.decision_loop.context import DecisionContext

logger = logging.getLogger(__name__)


@register("agentic_eventsat")
class AgenticEventSat(Representation):
    """CoALA-style agentic hybrid representation for EventSat.

    The LLM follows a Plan-Tool-Reflect-Decide protocol:
    1. PLAN: Analyze state and decide which tool(s) to query.
    2. TOOL: Execute domain tool, get structured result.
    3. REFLECT: Incorporate tool results, refine reasoning.
    4. DECIDE: Select final mode after sufficient information gathering.

    Max LLM calls per decision = max_agentic_steps (default 3).
    Same symbolic grounding as llm_eventsat (anomaly→safe, SoC<0.20→charging,
    no-pass→no-comms).
    """

    DEFAULT_MAX_STEPS: int = 3

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._client = LLMClient(config)
        self._max_agentic_steps: int = (config or {}).get(
            "max_agentic_steps", self.DEFAULT_MAX_STEPS
        )

        # State
        self._last_rationale: Optional[str] = None
        self._last_raw_responses: List[str] = []
        self._last_accumulated_context: List[Dict[str, Any]] = []

        # Metrics
        self._grounding_overrides: int = 0
        self._total_tool_calls: int = 0
        self._total_agentic_steps: int = 0
        self._total_decisions: int = 0
        self._tool_call_histogram: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        """Extract decision-relevant features (same as llm_eventsat for comparability)."""
        if hasattr(observation, "constellation_state"):
            sat = observation.constellation_state.satellites.get("eventsat_0")
            if sat is None:
                return {}
            res = sat.resources or {}
            meta = sat.metadata or {}
            return {
                "battery_soc": res.get("battery_soc", 0.5),
                "current_mode": sat.status,
                "in_sunlight": meta.get("in_sunlight", False),
                "ground_pass_active": meta.get("ground_pass_active", False),
                "data_stored_mb": res.get("data_stored_mb", 0.0),
                "obc_data_mb": res.get("obc_data_mb", meta.get("obc_data_mb", 0.0)),
                "jetson_raw_mb": meta.get("jetson_raw_mb", 0.0),
                "jetson_compressed_mb": meta.get("jetson_compressed_mb", 0.0),
                "storage_capacity_mb": meta.get("storage_capacity_mb", 512.0),
                "uncompressed_observations": meta.get("uncompressed_observations", 0),
                "compression_progress": meta.get("compression_progress", 0),
                "total_observation_s": meta.get("total_observation_s", 0.0),
                "health_status": meta.get("health_status", "nominal"),
                "undetected_observations": meta.get("undetected_observations", 0),
                "daily_downlink_budget_mb": meta.get("daily_downlink_budget_mb", 27.0),
                # Extended metadata from Phase 4b env extension
                "time_to_next_pass": meta.get("time_to_next_pass", None),
                "remaining_pass_duration": meta.get("remaining_pass_duration", 0),
                "time_to_next_eclipse": meta.get("time_to_next_eclipse", None),
                "orbital_phase": meta.get("orbital_phase", None),
            }
        return {}

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        """Select mode via agentic Plan-Tool-Reflect-Decide loop.

        Flow:
        1. Safety pre-checks (anomaly, empty state)
        2. Mock mode short-circuit
        3. PLAN step (LLM call 1)
        4. TOOL-REFLECT loop (LLM calls 2..N)
        5. Symbolic grounding
        """
        state = context.state
        enrichments = context.enrichments
        memory = context.memory

        # --- Safety pre-checks (0 LLM calls) ---
        health = state.get("health_status", "nominal")
        if health != "nominal":
            self._last_rationale = f"Symbolic safety: anomaly active ({health}); forced safe mode."
            self._grounding_overrides += 1
            return {"eventsat_0": {"mode": "safe"}}

        if not state:
            self._last_rationale = "No state available; defaulting to charging."
            return {"eventsat_0": {"mode": "charging"}}

        # --- Mock mode short-circuit (0 LLM calls) ---
        if self._client.mock_mode:
            mode = self._symbolic_fallback(state)
            self._last_rationale = f"Mock agentic: symbolic fallback selected '{mode}'."
            self._last_accumulated_context = []
            self._total_decisions += 1
            mode = self._apply_grounding(mode, state)
            return {"eventsat_0": {"mode": mode}}

        # --- Agentic loop ---
        return self._run_agentic_loop(state, enrichments, memory)

    def reason(self, state: Dict[str, Any], memory: Any) -> List[Dict[str, Any]]:
        """Produce structured reasoning steps for ReAct thought phase.

        Runs a lightweight agentic step: checks battery + pipeline + constraints
        via tools, then formats the results as structured reasoning trace.
        """
        if not state:
            return [{"check": "state", "value": None, "implication": "no_state_default_charging"}]

        # In mock mode, produce tool-based reasoning without LLM
        tool_results = []

        # Quick tool queries for reasoning
        from src.representation.agentic_tools import check_battery, check_data_pipeline, check_ground_pass
        battery_result = check_battery(state=state)
        pipeline_result = check_data_pipeline(state=state)
        pass_result = check_ground_pass(state=state)

        steps = []
        steps.append({
            "check": "battery",
            "value": battery_result["soc"],
            "implication": battery_result["charging_assessment"],
        })
        steps.append({
            "check": "ground_pass",
            "value": "active" if pass_result["active"] else "inactive",
            "implication": pass_result["recommendation"],
        })
        steps.append({
            "check": "data_pipeline",
            "value": pipeline_result["bottleneck"],
            "implication": pipeline_result["pipeline_summary"],
        })

        # If not mock, also ask LLM for deeper reasoning
        if not self._client.mock_mode:
            tool_result_entries = [
                {"tool": "check_battery", "result": battery_result},
                {"tool": "check_data_pipeline", "result": pipeline_result},
                {"tool": "check_ground_pass", "result": pass_result},
            ]
            prompt = format_agentic_reasoning_prompt(state, memory, tool_result_entries)
            try:
                raw = self._client.generate(
                    system_prompt=AGENTIC_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    json_mode=True,
                )
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return [parsed]
            except Exception as e:
                logger.debug("Agentic reasoning parse failed: %s", e)

        return steps

    # ------------------------------------------------------------------
    # Optional extension points
    # ------------------------------------------------------------------

    def get_rationale(self) -> Optional[str]:
        """Return the full reasoning chain for the last decision."""
        return self._last_rationale

    def get_metrics(self) -> Dict[str, float]:
        """Return agentic + LLM metrics."""
        metrics = self._client.get_metrics()
        metrics["agentic_grounding_overrides"] = float(self._grounding_overrides)
        metrics["agentic_total_tool_calls"] = float(self._total_tool_calls)
        metrics["agentic_total_decisions"] = float(self._total_decisions)
        avg_steps = (
            self._total_agentic_steps / self._total_decisions
            if self._total_decisions > 0 else 0.0
        )
        metrics["agentic_avg_steps_per_decision"] = round(avg_steps, 2)

        # Per-tool histogram
        for tool_name, count in self._tool_call_histogram.items():
            metrics[f"agentic_tool_{tool_name}"] = float(count)

        return metrics

    def get_name(self) -> str:
        return "AgenticEventSat"

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    def _run_agentic_loop(
        self,
        state: Dict[str, Any],
        enrichments: Dict[str, Any],
        memory: Any,
    ) -> Dict[str, Any]:
        """Execute the Plan-Tool-Reflect-Decide cycle."""
        accumulated_context: List[Dict[str, Any]] = []
        raw_responses: List[str] = []
        remaining_budget = self._max_agentic_steps
        mode: Optional[str] = None
        rationale: Optional[str] = None
        steps_taken = 0

        # Step 1: PLAN
        user_prompt = format_planning_prompt(state, enrichments)
        try:
            raw = self._client.generate(
                system_prompt=AGENTIC_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_mode=True,
            )
            raw_responses.append(raw)
            parsed = self._parse_agentic_response(raw)
            steps_taken += 1
            remaining_budget -= 1

            # Check for immediate decision
            decision = parsed.get("decision")
            if decision:
                mode = decision.get("mode")
                rationale = decision.get("rationale", "")
                accumulated_context.append({
                    "step": "plan_decide",
                    "content": parsed.get("plan", parsed.get("reflection", "")),
                })
            else:
                accumulated_context.append({
                    "step": "plan",
                    "content": parsed.get("plan", ""),
                })

                # Steps 2..N: TOOL-REFLECT loop
                tool_call = parsed.get("tool_call")
                while remaining_budget > 0 and tool_call and mode is None:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})

                    # Execute tool
                    tool_result = execute_tool(tool_name, tool_args, state, memory)
                    self._total_tool_calls += 1
                    self._tool_call_histogram[tool_name] = (
                        self._tool_call_histogram.get(tool_name, 0) + 1
                    )
                    accumulated_context.append({
                        "step": "tool",
                        "name": tool_name,
                        "result": tool_result,
                    })

                    # Reflect
                    reflect_prompt = format_tool_result_prompt(
                        tool_name, tool_result, accumulated_context
                    )
                    try:
                        raw = self._client.generate(
                            system_prompt=AGENTIC_SYSTEM_PROMPT,
                            user_prompt=reflect_prompt,
                            json_mode=True,
                        )
                        raw_responses.append(raw)
                        parsed = self._parse_agentic_response(raw)
                        steps_taken += 1
                        remaining_budget -= 1

                        reflection = parsed.get("reflection", "")
                        accumulated_context.append({
                            "step": "reflect",
                            "content": reflection,
                        })

                        # Check for decision
                        decision = parsed.get("decision")
                        if decision:
                            mode = decision.get("mode")
                            rationale = decision.get("rationale", "")
                        else:
                            tool_call = parsed.get("tool_call")

                    except Exception as e:
                        logger.warning("Agentic reflect step failed: %s", e)
                        accumulated_context.append({
                            "step": "reflect_error",
                            "content": str(e),
                        })
                        break

        except Exception as e:
            logger.warning("Agentic plan step failed: %s", e)
            accumulated_context.append({
                "step": "plan_error",
                "content": str(e),
            })

        # Fallback if no decision was made
        if mode is None or mode not in VALID_MODES:
            mode = self._symbolic_fallback(state)
            rationale = (
                f"Agentic loop ended without valid decision after {steps_taken} steps; "
                f"symbolic fallback selected '{mode}'."
            )
            self._grounding_overrides += 1

        # Apply grounding
        mode = self._apply_grounding(mode, state)

        # Build rationale from full chain
        chain_parts = []
        for entry in accumulated_context:
            step_type = entry.get("step", "")
            if step_type in ("plan", "plan_decide"):
                chain_parts.append(f"Plan: {entry.get('content', '')[:100]}")
            elif step_type == "tool":
                chain_parts.append(f"Tool({entry.get('name', '?')})")
            elif step_type == "reflect":
                chain_parts.append(f"Reflect: {entry.get('content', '')[:100]}")

        chain_summary = " → ".join(chain_parts) if chain_parts else "direct"
        self._last_rationale = f"Agentic [{chain_summary}] → {mode}: {rationale or ''}"
        self._last_raw_responses = raw_responses
        self._last_accumulated_context = accumulated_context

        # Update metrics
        self._total_agentic_steps += steps_taken
        self._total_decisions += 1

        return {"eventsat_0": {"mode": mode}}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_agentic_response(self, raw: str) -> Dict[str, Any]:
        """Parse agentic LLM response as JSON, handling formatting issues."""
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Handle /think blocks from reasoning models
        if "<think>" in text:
            # Extract content after </think>
            parts = text.split("</think>")
            if len(parts) > 1:
                text = parts[-1].strip()

        return json.loads(text)

    def _symbolic_fallback(self, state: Dict[str, Any]) -> str:
        """Minimal symbolic fallback when LLM is unavailable.

        Same logic as llm_eventsat for fair comparison.
        """
        soc = state.get("battery_soc", 0.5)
        if soc < 0.35:
            return "charging"
        if state.get("ground_pass_active", False) and state.get("obc_data_mb", 0.0) > 0:
            return "communication"
        return "charging"

    def _apply_grounding(self, mode: str, state: Dict[str, Any]) -> str:
        """Apply symbolic grounding constraints.

        Same rules as llm_eventsat for fair comparison:
        - Cannot communicate without an active ground pass
        - Very low battery overrides any mode to charging
        """
        # Cannot communicate without an active ground pass
        if mode == "communication" and not state.get("ground_pass_active", False):
            logger.debug("Grounding: overriding 'communication' — no active pass.")
            self._grounding_overrides += 1
            return "charging"

        # Very low battery override (hard constraint)
        soc = state.get("battery_soc", 0.5)
        if soc < 0.20 and mode != "charging":
            logger.debug("Grounding: overriding '%s' — SoC %.2f below hard limit.", mode, soc)
            self._grounding_overrides += 1
            return "charging"

        return mode

"""
Agentic Prompt Templates for CoALA-style EventSat Representation.

Extended prompt design for multi-step agentic reasoning with tool use.
The LLM follows a Plan-Tool-Reflect-Decide protocol, using domain tools
to query satellite state before making mode selections.

Papers:
- Sumers et al. (2024) [CoALA] — agentic architecture with tool use
- Rodriguez-Fernandez et al. (2024) §3.2 — prompt engineering for sat ops
- Li (2025) — tool-augmented AI agents for satellite operations

All prompts are pure functions (no side effects) for testability.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.representation.agentic_tools import get_tool_schemas
from src.representation.llm_prompts import DEFAULT_STORAGE_CAPACITY_MB


# ======================================================================
# System prompt
# ======================================================================

def _build_tool_descriptions() -> str:
    """Build formatted tool descriptions for system prompt."""
    schemas = get_tool_schemas()
    lines = []
    for schema in schemas:
        params = ", ".join(
            f"{k}: {v}" for k, v in schema.get("parameters", {}).items()
            if k != "state"  # state is always implicit
        )
        param_str = f" ({params})" if params else ""
        lines.append(f"  - {schema['name']}{param_str}: {schema['description']}")
    return "\n".join(lines)


AGENTIC_SYSTEM_PROMPT = """\
You are an autonomous satellite operations agent managing a single Earth \
observation satellite in low Earth orbit (400 km SSO).

MISSION: Maximise observation data downlinked to ground while maintaining \
satellite health and safety.

AVAILABLE MODES (exactly one per timestep):
- charging: Recharge battery from solar panels (only effective in sunlight).
- payload_observe: Capture Earth observation imagery (consumes power, produces raw data on Jetson).
- payload_compress: Compress raw observations on Jetson (reduces size ~5:1, takes ~2x observation time).
- payload_detect: Run CV detection on compressed observations (5 min per observation).
- payload_send: Transfer compressed/detected data from Jetson to OBC via RS-485 (50 kbps).
- communication: Downlink data from OBC to ground station during a ground pass.
- safe: Minimal power mode for anomaly recovery (environment may force this).

DATA PIPELINE (3-pool):
  Jetson raw → (compress) → Jetson compressed → (send) → OBC → (communicate) → Ground

CONSTRAINTS:
- Battery SoC must stay above 0.20 (hard limit) and above 0.35 (preferred).
- Ground passes are limited windows; OBC data must be ready before pass starts.
- ADCS settling takes 135 seconds when switching to observe or communicate mode.
- Daily downlink budget is finite (configurable, typically 27 MB).
- Anomalies force safe mode; you cannot override environment-enforced safe mode.

REASONING PROTOCOL:
You make decisions using a Plan-Tool-Reflect-Decide cycle:
1. PLAN: Analyze the situation and decide which tool(s) to use.
2. TOOL: Request a tool call to gather information.
3. REFLECT: Incorporate tool results and refine your reasoning.
4. DECIDE: When you have enough information, select a mode.

You may call tools multiple times before deciding. Use tools to verify \
assumptions rather than guessing.

AVAILABLE TOOLS:
""" + _build_tool_descriptions() + """

OUTPUT FORMAT:
At each step, respond with a JSON object. The format depends on what you want to do:

To call a tool:
  {"plan": "<your reasoning>", "tool_call": {"name": "<tool_name>", "args": {<args>}}}

To make a final decision (after sufficient tool use):
  {"decision": {"mode": "<mode_name>", "rationale": "<brief explanation>"}}

To call a tool AND make a decision simultaneously:
  {"reflection": "<updated reasoning>", "decision": {"mode": "<mode_name>", "rationale": "<why>"}}

Do not include any text outside the JSON object."""


# ======================================================================
# Planning prompt (first LLM call in agentic loop)
# ======================================================================

def format_planning_prompt(
    state: Dict[str, Any],
    enrichments: Dict[str, Any] | None = None,
) -> str:
    """Format the initial planning prompt with current state.

    Args:
        state: Encoded observation dict from encode_observation().
        enrichments: Optional loop-specific enrichments (OODA/ReAct).

    Returns:
        Formatted prompt for the planning step.
    """
    if not state:
        return (
            "No satellite state available. Decide on the safest mode.\n"
            'Respond with: {"decision": {"mode": "charging", "rationale": "<why>"}}'
        )

    soc = state.get("battery_soc", 0.5)
    mode = state.get("current_mode", "unknown")
    pass_active = state.get("ground_pass_active", False)
    in_sunlight = state.get("in_sunlight", False)
    obc_mb = state.get("obc_data_mb", 0.0)
    jetson_raw = state.get("jetson_raw_mb", 0.0)
    jetson_comp = state.get("jetson_compressed_mb", 0.0)
    cap_mb = state.get("storage_capacity_mb", DEFAULT_STORAGE_CAPACITY_MB)
    uncomp = state.get("uncompressed_observations", 0)
    undetected = state.get("undetected_observations", 0)
    health = state.get("health_status", "nominal")
    budget_mb = state.get("daily_downlink_budget_mb", 27.0)

    lines = [
        "CURRENT SATELLITE STATE:",
        f"  Battery SoC: {soc:.2f} (sunlight: {'yes' if in_sunlight else 'no'})",
        f"  Current mode: {mode}",
        f"  Health: {health}",
        f"  Ground pass active: {'YES' if pass_active else 'no'}",
        "",
        "DATA PIPELINE:",
        f"  Jetson raw: {jetson_raw:.2f} MB ({uncomp} uncompressed obs)",
        f"  Jetson compressed: {jetson_comp:.2f} MB ({undetected} undetected obs)",
        f"  OBC ready for downlink: {obc_mb:.2f} / {cap_mb:.0f} MB",
        f"  Daily downlink budget: {budget_mb:.0f} MB",
    ]

    # Loop enrichments
    if enrichments:
        lines.append("")
        lines.append("SITUATION ASSESSMENT:")
        if "situation_class" in enrichments:
            lines.append(f"  Situation: {enrichments['situation_class']}")
        if "urgency" in enrichments:
            lines.append(f"  Urgency: {enrichments['urgency']:.2f}")
        if "reasoning_trace" in enrichments:
            trace = enrichments["reasoning_trace"]
            if trace:
                lines.append(f"  Prior reasoning: {len(trace)} steps")
                for step in trace[-3:]:
                    lines.append(f"    - {step.get('check', '?')}: {step.get('implication', '?')}")
        if "grounding_violations" in enrichments:
            violations = enrichments["grounding_violations"]
            if violations:
                lines.append(f"  Prior violations: {violations}")

    lines.append("")
    lines.append(
        "Analyze the state above. Use tools to check battery, pass windows, "
        "pipeline status, or constraints before deciding on a mode. "
        "Respond with JSON."
    )

    return "\n".join(lines)


# ======================================================================
# Tool result prompt (reflect step)
# ======================================================================

def format_tool_result_prompt(
    tool_name: str,
    tool_result: Dict[str, Any],
    accumulated_context: List[Dict[str, Any]],
) -> str:
    """Format tool result for the reflect step.

    Args:
        tool_name: Name of the tool that was called.
        tool_result: Structured result from the tool.
        accumulated_context: List of prior steps [{step, content/name/result}].

    Returns:
        Prompt for the LLM to reflect on tool results.
    """
    # Summarize prior context
    prior_lines = []
    for entry in accumulated_context:
        step_type = entry.get("step", "unknown")
        if step_type == "plan":
            prior_lines.append(f"  PLAN: {entry.get('content', '')[:200]}")
        elif step_type == "tool":
            prior_lines.append(f"  TOOL ({entry.get('name', '?')}): {_summarize_result(entry.get('result', {}))}")
        elif step_type == "reflect":
            prior_lines.append(f"  REFLECT: {entry.get('content', '')[:200]}")

    lines = []
    if prior_lines:
        lines.append("REASONING SO FAR:")
        lines.extend(prior_lines)
        lines.append("")

    lines.append(f"LATEST TOOL RESULT ({tool_name}):")
    lines.append(f"  {json.dumps(tool_result, indent=2)}")
    lines.append("")
    lines.append(
        "Based on this information, either:\n"
        "1. Call another tool for more information: "
        '{\"reflection\": \"<reasoning>\", \"tool_call\": {\"name\": \"<tool>\", \"args\": {}}}\n'
        "2. Make your decision: "
        '{\"reflection\": \"<reasoning>\", \"decision\": {\"mode\": \"<mode>\", \"rationale\": \"<why>\"}}'
    )

    return "\n".join(lines)


def format_forced_decision_prompt(
    accumulated_context: List[Dict[str, Any]],
) -> str:
    """Terminal Decide-phase prompt — tool budget exhausted, decision required.

    A bounded agentic loop must close with an answer-extraction step (ReAct,
    Yao et al. 2023): the reflect prompt always offers a tool option, so a
    tool-hungry model can ride the budget to exhaustion without ever deciding.
    This prompt offers no tool option.
    """
    prior_lines = []
    for entry in accumulated_context:
        step_type = entry.get("step", "unknown")
        if step_type == "plan":
            prior_lines.append(f"  PLAN: {entry.get('content', '')[:200]}")
        elif step_type == "tool":
            prior_lines.append(
                f"  TOOL ({entry.get('name', '?')}): {_summarize_result(entry.get('result', {}))}"
            )
        elif step_type == "reflect":
            prior_lines.append(f"  REFLECT: {entry.get('content', '')[:200]}")

    lines = []
    if prior_lines:
        lines.append("REASONING SO FAR:")
        lines.extend(prior_lines)
        lines.append("")
    lines.append(
        "Your tool budget is exhausted. Decide the operating mode NOW using only "
        "the information above.\n"
        "Respond with ONLY the decision JSON — tool calls are not available:\n"
        '{"decision": {"mode": "<mode>", "rationale": "<why>"}}'
    )
    return "\n".join(lines)


def _summarize_result(result: Dict[str, Any]) -> str:
    """One-line summary of a tool result for context."""
    if "error" in result:
        return f"Error: {result['error']}"
    # Pick key fields for common tools
    if "soc" in result:
        return f"SoC={result['soc']}, feasible={result.get('feasible_modes', [])}"
    if "active" in result and "obc_data_mb" in result:
        return f"pass={'active' if result['active'] else 'inactive'}, obc={result['obc_data_mb']}MB"
    if "bottleneck" in result:
        return f"bottleneck={result['bottleneck']}, obc={result.get('obc_data_mb', 0)}MB"
    if "feasible" in result and "violations" in result:
        return f"feasible={result['feasible']}, violations={len(result['violations'])}"
    if "last_modes" in result:
        return f"modes={result['last_modes'][-3:]}, trend={result.get('battery_trend', '?')}"
    if "estimated_utility" in result:
        return f"utility={result['estimated_utility']}, risks={len(result.get('risk_factors', []))}"
    # Fallback
    return json.dumps(result)[:150]


# ======================================================================
# Reasoning prompt (for ReAct thought step)
# ======================================================================

def format_agentic_reasoning_prompt(
    state: Dict[str, Any],
    memory: Optional[Any] = None,
    tool_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format reasoning prompt for ReAct thought step.

    Produces structured reasoning trace from accumulated tool results,
    in the same [{"check", "value", "implication"}] format as llm_eventsat.

    Args:
        state: Encoded observation dict.
        memory: Agent memory (used by recall_history).
        tool_results: Optional pre-computed tool results.

    Returns:
        Prompt asking the LLM to produce structured reasoning steps.
    """
    if not state:
        return "No state available. List the key factors for choosing safe mode."

    soc = state.get("battery_soc", 0.5)
    health = state.get("health_status", "nominal")
    pass_active = state.get("ground_pass_active", False)
    uncomp = state.get("uncompressed_observations", 0)
    obc_mb = state.get("obc_data_mb", 0.0)

    pass_str = "active" if pass_active else "inactive"

    lines = [
        f"Analyze the satellite state and identify key decision factors.",
        f"State summary: SoC={soc:.2f}, health={health}, pass={pass_str}, "
        f"uncompressed={uncomp}, obc_data={obc_mb:.1f}MB",
    ]

    if tool_results:
        lines.append("")
        lines.append("Tool analysis results:")
        for tr in tool_results:
            name = tr.get("tool", "unknown")
            result = tr.get("result", {})
            lines.append(f"  {name}: {_summarize_result(result)}")

    example = '[{"check": "battery", "value": 0.45, "implication": "charging_preferred"}]'
    schema = '{"check": "<what>", "value": <numeric or string>, "implication": "<conclusion>"}'

    lines.append("")
    lines.append(
        f"Respond with a JSON array of reasoning steps, each with fields:\n"
        f"  {schema}\n"
        f"Example: {example}"
    )

    return "\n".join(lines)

"""
Agentic Tools — Domain-specific tools for CoALA-style agentic reasoning.

Pure functions operating on state/memory dicts — no side effects, no LLM calls.
Tools are invoked by the agentic representation during its Plan-Tool-Reflect-Decide
loop to query satellite state, check constraints, and evaluate plans.

Papers:
- Sumers et al. (2024) [CoALA] — action decomposition into internal (reasoning,
  retrieval) and external (tool use, grounding) actions
- Li (2025) — tool-augmented AI agents for satellite operations
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ======================================================================
# Tool type definition
# ======================================================================

class ToolDef:
    """Tool definition with schema for prompt embedding."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, str],
        func: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def to_schema(self) -> Dict[str, Any]:
        """Return schema dict for prompt embedding."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ======================================================================
# Tool registry
# ======================================================================

TOOL_REGISTRY: Dict[str, ToolDef] = {}


def _register_tool(
    name: str, description: str, parameters: Dict[str, str],
):
    """Decorator to register an agentic tool."""
    def decorator(func):
        TOOL_REGISTRY[name] = ToolDef(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )
        return func
    return decorator


# ======================================================================
# Mode feasibility helper
# ======================================================================

VALID_MODES = [
    "charging", "communication", "payload_observe", "payload_compress",
    "payload_detect", "payload_send", "safe",
]

# Modes that require minimum SoC thresholds
_ENERGY_INTENSIVE_MODES = frozenset({
    "payload_observe", "payload_compress", "payload_detect",
    "payload_send", "communication",
})

_SOC_HARD_LIMIT = 0.20
_SOC_PREFERRED = 0.35


def _get_feasible_modes(state: Dict[str, Any]) -> List[str]:
    """Return list of feasible modes given current state."""
    soc = state.get("battery_soc", 0.5)
    pass_active = state.get("ground_pass_active", False)
    health = state.get("health_status", "nominal")

    if health != "nominal":
        return ["safe"]

    feasible = []
    for mode in VALID_MODES:
        if mode == "safe":
            feasible.append(mode)
            continue
        if soc < _SOC_HARD_LIMIT:
            if mode == "charging":
                feasible.append(mode)
            continue
        if mode == "communication" and not pass_active:
            continue
        if mode in _ENERGY_INTENSIVE_MODES and soc < _SOC_PREFERRED:
            # Feasible but risky
            feasible.append(mode)
            continue
        feasible.append(mode)

    if "charging" not in feasible:
        feasible.insert(0, "charging")
    return feasible


# ======================================================================
# Derived-telemetry helpers (NOT agentic tools)
#
# These summarise telemetry the planner already receives each step. They are
# FOLDED into the planning prompt (feasible modes, pipeline bottleneck) and
# reused by AgenticEventSat.reason(); they are NOT registered as agentic tools.
# Spending an LLM round-trip to re-read state the prompt already contains just
# inflates decision latency — the planner has the numbers, so the derivations
# go in the prompt, not behind a tool call. See
# agentic_prompts.format_planning_prompt / format_schedule_planning_prompt.
# ======================================================================

def _get_pipeline_bottleneck(state: Dict[str, Any]) -> str:
    """Identify the current data-pipeline bottleneck from received telemetry."""
    if state.get("uncompressed_observations", 0) > 0:
        return "compression_needed"
    if state.get("undetected_observations", 0) > 0:
        return "detection_needed"
    if state.get("jetson_compressed_mb", 0.0) > 0:
        return "send_to_obc_needed"
    if state.get("obc_data_mb", 0.0) > 0:
        return "downlink_needed"
    return "none"


def check_battery(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Battery assessment with mode feasibility analysis."""
    soc = state.get("battery_soc", 0.5)
    in_sunlight = state.get("in_sunlight", False)

    if soc >= _SOC_PREFERRED:
        charging_assessment = "good"
    elif soc >= _SOC_HARD_LIMIT:
        charging_assessment = "low"
    else:
        charging_assessment = "critical"

    return {
        "soc": round(soc, 3),
        "in_sunlight": in_sunlight,
        "charging_rate": "nominal" if in_sunlight else "none (eclipse)",
        "charging_assessment": charging_assessment,
        "below_preferred": soc < _SOC_PREFERRED,
        "below_hard_limit": soc < _SOC_HARD_LIMIT,
        "feasible_modes": _get_feasible_modes(state),
    }


def check_ground_pass(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Ground pass window assessment."""
    active = state.get("ground_pass_active", False)
    obc_data = state.get("obc_data_mb", 0.0)

    # Orbital lookahead from extended metadata (Phase 4b env extension)
    time_to_next = state.get("time_to_next_pass", None)
    remaining = state.get("remaining_pass_duration", 0)

    result: Dict[str, Any] = {
        "active": active,
        "obc_data_mb": round(obc_data, 2),
        "data_ready_for_downlink": obc_data > 0,
    }

    if time_to_next is not None:
        result["time_to_next"] = f"~{int(time_to_next)} steps"
    else:
        result["time_to_next"] = "unknown"

    if active:
        result["remaining_duration"] = int(remaining) if remaining else 0
        result["recommendation"] = (
            "communicate" if obc_data > 0 else "no data to downlink"
        )
    else:
        result["remaining_duration"] = 0
        result["recommendation"] = "pass not active — cannot communicate"

    return result


def check_data_pipeline(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Data pipeline status with bottleneck identification."""
    jetson_raw = state.get("jetson_raw_mb", 0.0)
    jetson_compressed = state.get("jetson_compressed_mb", 0.0)
    obc_data = state.get("obc_data_mb", 0.0)
    uncompressed = state.get("uncompressed_observations", 0)
    undetected = state.get("undetected_observations", 0)
    compression_progress = state.get("compression_progress", 0)
    daily_budget = state.get("daily_downlink_budget_mb", 27.0)

    bottleneck = _get_pipeline_bottleneck(state)

    # Build summary
    parts = []
    if uncompressed > 0:
        parts.append(f"{uncompressed} uncompressed obs on Jetson")
    if undetected > 0:
        parts.append(f"{undetected} undetected obs")
    if jetson_compressed > 0:
        parts.append(f"{jetson_compressed:.1f} MB compressed on Jetson")
    if obc_data > 0:
        parts.append(f"{obc_data:.1f} MB on OBC ready for downlink")
    if not parts:
        parts.append("pipeline empty")

    return {
        "jetson_raw_mb": round(jetson_raw, 2),
        "jetson_compressed_mb": round(jetson_compressed, 2),
        "obc_data_mb": round(obc_data, 2),
        "uncompressed": uncompressed,
        "undetected": undetected,
        "compression_progress": compression_progress,
        "daily_downlink_budget_mb": round(daily_budget, 2),
        "bottleneck": bottleneck,
        "pipeline_summary": "; ".join(parts),
    }


# ======================================================================
# Agentic tools (external actions advertised to the model — CoALA §3)
#
# What-if / lookup actions the planner CANNOT answer from the prompt alone:
# validate a candidate mode against the constraints (check_constraints), score
# it (evaluate_plan), or query episodic memory (recall_history — per-step core
# only; the ground scheduler has no memory, so it advertises only the first two
# via SCHEDULE_TOOL_NAMES below).
# ======================================================================

@_register_tool(
    name="check_constraints",
    description="Pre-validate whether a proposed mode is feasible given the current state. Returns violations and warnings.",
    parameters={"state": "Current satellite state dict", "proposed_mode": "Mode to check (string)"},
)
def check_constraints(
    state: Dict[str, Any], proposed_mode: str = "charging", **kwargs,
) -> Dict[str, Any]:
    """Constraint check for a proposed mode."""
    soc = state.get("battery_soc", 0.5)
    pass_active = state.get("ground_pass_active", False)
    health = state.get("health_status", "nominal")

    violations: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    # Hard constraints
    if health != "nominal" and proposed_mode != "safe":
        violations.append({
            "constraint": "anomaly",
            "reason": f"Anomaly active ({health}); only safe mode allowed.",
        })

    if soc < _SOC_HARD_LIMIT and proposed_mode != "charging":
        violations.append({
            "constraint": "battery_critical",
            "reason": f"SoC {soc:.2f} below hard limit {_SOC_HARD_LIMIT}; must charge.",
        })

    if proposed_mode == "communication" and not pass_active:
        violations.append({
            "constraint": "ground_pass",
            "reason": "No active ground pass; cannot communicate.",
        })

    # Warnings
    if soc < _SOC_PREFERRED and proposed_mode in _ENERGY_INTENSIVE_MODES:
        warnings.append({
            "constraint": "battery_low",
            "reason": f"SoC {soc:.2f} below preferred {_SOC_PREFERRED}; consider charging first.",
        })

    if proposed_mode not in VALID_MODES:
        violations.append({
            "constraint": "invalid_mode",
            "reason": f"'{proposed_mode}' is not a valid EventSat mode.",
        })

    return {
        "proposed_mode": proposed_mode,
        "feasible": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
    }


@_register_tool(
    name="recall_history",
    description="Query episodic memory: retrieve recent mode history, mode frequency counts, and battery trend.",
    parameters={"memory": "FixedMemory instance or None", "n": "Number of recent steps (default 5)"},
)
def recall_history(
    state: Dict[str, Any],
    memory: Optional[Any] = None,
    n: int = 5,
    **kwargs,
) -> Dict[str, Any]:
    """Query episodic memory for recent mode history and trends."""
    last_modes: List[str] = []
    battery_values: List[float] = []

    if memory is not None:
        try:
            history = memory.query("history") or []
        except Exception:
            history = []

        for entry in history[-n:]:
            sats = entry.get("satellites", {})
            sat = sats.get("eventsat_0", {})
            if isinstance(sat, dict):
                mode = sat.get("status", "unknown")
                soc = sat.get("resources", {}).get("battery_soc", None)
            else:
                mode = getattr(sat, "status", "unknown")
                res = getattr(sat, "resources", {}) or {}
                soc = res.get("battery_soc", None)
            last_modes.append(mode)
            if soc is not None:
                battery_values.append(soc)

    # Compute mode counts
    mode_counts: Dict[str, int] = {}
    for m in last_modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    # Battery trend
    if len(battery_values) >= 2:
        if battery_values[-1] > battery_values[0] + 0.01:
            battery_trend = "rising"
        elif battery_values[-1] < battery_values[0] - 0.01:
            battery_trend = "falling"
        else:
            battery_trend = "stable"
    else:
        battery_trend = "insufficient_data"

    return {
        "last_modes": last_modes,
        "mode_counts": mode_counts,
        "battery_trend": battery_trend,
        "history_depth": len(last_modes),
    }


@_register_tool(
    name="evaluate_plan",
    description="Heuristic evaluation of a proposed mode: estimated utility and risk factors.",
    parameters={"state": "Current satellite state dict", "proposed_mode": "Mode to evaluate (string)"},
)
def evaluate_plan(
    state: Dict[str, Any], proposed_mode: str = "charging", **kwargs,
) -> Dict[str, Any]:
    """Heuristic plan evaluation — estimated utility and risks."""
    soc = state.get("battery_soc", 0.5)
    pass_active = state.get("ground_pass_active", False)
    obc_data = state.get("obc_data_mb", 0.0)
    in_sunlight = state.get("in_sunlight", False)
    uncompressed = state.get("uncompressed_observations", 0)
    undetected = state.get("undetected_observations", 0)
    jetson_compressed = state.get("jetson_compressed_mb", 0.0)

    risk_factors: List[str] = []
    utility = 0.5  # baseline

    if proposed_mode == "charging":
        utility = 0.4 if soc > _SOC_PREFERRED else 0.7
        if not in_sunlight:
            risk_factors.append("in eclipse — charging ineffective")
            utility -= 0.2

    elif proposed_mode == "communication":
        if pass_active and obc_data > 0:
            utility = 0.9
        elif pass_active:
            utility = 0.3
            risk_factors.append("pass active but no OBC data to downlink")
        else:
            utility = 0.0
            risk_factors.append("no ground pass — communication blocked")

    elif proposed_mode == "payload_observe":
        utility = 0.7
        if soc < _SOC_PREFERRED:
            risk_factors.append("battery below preferred — observation drains power")
            utility -= 0.2

    elif proposed_mode == "payload_compress":
        if uncompressed > 0:
            utility = 0.7
        else:
            utility = 0.1
            risk_factors.append("no uncompressed observations to process")

    elif proposed_mode == "payload_detect":
        if undetected > 0:
            utility = 0.7
        else:
            utility = 0.1
            risk_factors.append("no undetected observations to process")

    elif proposed_mode == "payload_send":
        if jetson_compressed > 0:
            utility = 0.6
        else:
            utility = 0.1
            risk_factors.append("no compressed data on Jetson to send")

    elif proposed_mode == "safe":
        utility = 0.2
        risk_factors.append("safe mode — minimal operations")

    # Common risk: low battery
    if soc < _SOC_PREFERRED and proposed_mode in _ENERGY_INTENSIVE_MODES:
        risk_factors.append("SoC below preferred threshold")

    recommendation = "proceed" if utility >= 0.5 and not risk_factors else "reconsider"
    if utility >= 0.5:
        recommendation = "proceed"

    return {
        "proposed_mode": proposed_mode,
        "estimated_utility": round(max(0.0, min(1.0, utility)), 2),
        "risk_factors": risk_factors,
        "recommendation": recommendation,
    }


# ======================================================================
# CoALA memory-write tools (writable_coala mechanism only)
# ======================================================================

# These are NOT in TOOL_REGISTRY by default — they're injected at runtime
# only when behaviour_config.mechanism == "writable_coala".

_WRITABLE_TOOL_REGISTRY: Dict[str, ToolDef] = {}


def _register_writable_tool(
    name: str, description: str, parameters: Dict[str, str],
):
    """Decorator to register a writable-memory tool."""
    def decorator(func):
        _WRITABLE_TOOL_REGISTRY[name] = ToolDef(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )
        return func
    return decorator


@_register_writable_tool(
    name="memory_write_rule",
    description=(
        "Write a learned domain rule to semantic memory. Use when you discover "
        "a reliable condition-action pattern that should persist across episodes. "
        "Example: 'If battery < 20% and in eclipse, avoid payload modes.'"
    ),
    parameters={
        "rule_text": "Human-readable rule description",
        "condition": "Trigger condition (e.g. 'battery < 20%')",
        "action": "Recommended response action",
    },
)
def memory_write_rule(
    state: Dict[str, Any],
    memory: Optional[Any] = None,
    rule_text: str = "",
    condition: str = "",
    action: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """Write a domain rule to writable semantic memory."""
    if memory is None or not hasattr(memory, "write_semantic_rule"):
        return {"error": "Writable memory not available. Check behaviour_config.mechanism."}
    confirmation = memory.write_semantic_rule(
        rule_text=rule_text,
        condition=condition,
        action=action,
    )
    return {"status": "written", "message": confirmation}


@_register_writable_tool(
    name="memory_write_episode",
    description=(
        "Write a summary of this episode's experience to episodic memory. "
        "Use at the end of an episode to record key decisions and outcomes. "
        "Example summary: 'Heavy eclipse period — stayed in charging most of episode. "
        "Missed 2 observation windows.'"
    ),
    parameters={
        "summary": "Summary of what happened and key decisions made",
        "outcome": "Quantified outcome (e.g. utility=0.72, anomalies=1)",
    },
)
def memory_write_episode(
    state: Dict[str, Any],
    memory: Optional[Any] = None,
    summary: str = "",
    outcome: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """Write an episode summary to writable episodic memory."""
    if memory is None or not hasattr(memory, "write_episodic_entry"):
        return {"error": "Writable memory not available. Check behaviour_config.mechanism."}
    confirmation = memory.write_episodic_entry(summary=summary, outcome=outcome)
    return {"status": "written", "message": confirmation}


# ======================================================================
# Public API
# ======================================================================

def execute_tool(
    tool_name: str,
    args: Dict[str, Any],
    state: Dict[str, Any],
    memory: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a registered tool by name.

    Searches the standard TOOL_REGISTRY first, then _WRITABLE_TOOL_REGISTRY.
    Returns tool result dict, or error dict for unknown tools.
    """
    tool_def = TOOL_REGISTRY.get(tool_name) or _WRITABLE_TOOL_REGISTRY.get(tool_name)
    if tool_def is None:
        return {"error": f"Unknown tool '{tool_name}'", "available": list(TOOL_REGISTRY.keys())}

    return tool_def.func(state=state, memory=memory, **args)


# Tool subset advertised to the ground scheduler (hllm-a / llm-a). It plans on
# fresh telemetry with no episodic memory, so recall_history (memory=None →
# empty) is excluded; only the what-if tools remain. The per-step core keeps the
# full registry (recall_history is live there).
SCHEDULE_TOOL_NAMES: List[str] = ["check_constraints", "evaluate_plan"]


def get_tool_schemas(
    include_writable: bool = False,
    tool_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return list of tool schemas for prompt embedding.

    Args:
        include_writable: If True, include writable-memory tools (only valid
            when behaviour_config.mechanism == "writable_coala").
        tool_names: If given, restrict to these registry tools, in order (used
            by the ground scheduler to drop memory-dependent tools).
    """
    if tool_names is None:
        items = list(TOOL_REGISTRY.values())
    else:
        items = [TOOL_REGISTRY[n] for n in tool_names if n in TOOL_REGISTRY]
    schemas = [t.to_schema() for t in items]
    if include_writable:
        schemas.extend(t.to_schema() for t in _WRITABLE_TOOL_REGISTRY.values())
    return schemas


TOOL_SCHEMAS = get_tool_schemas()
TOOL_SCHEMAS_WRITABLE = get_tool_schemas(include_writable=True)

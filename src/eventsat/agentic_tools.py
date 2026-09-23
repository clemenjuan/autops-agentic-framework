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

import math
from typing import Any, Dict, List, Optional

from src.eventsat.transitions import (
    PipelineParameters,
    apply_can_transfer,
    apply_compress,
    apply_detect,
    apply_downlink,
    apply_observe,
    can_compress,
)


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
_DEFAULT_MODE_MIN_SOC = {
    "payload_observe": 0.40,
    "payload_compress": 0.30,
    "payload_detect": 0.30,
    "payload_send": 0.30,
}


def _float_state(state: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(state.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _pipeline_parameters(state: Dict[str, Any]) -> PipelineParameters:
    """Build the canonical projection constants from declared telemetry."""

    return PipelineParameters(
        observation_size_mb=_float_state(state, "observation_size_mb", 9.41),
        compression_ratio=_float_state(state, "compression_ratio", 5.11),
        jetson_capacity_mb=_float_state(
            state, "jetson_capacity_mb", 249036.8
        ),
        obc_capacity_mb=_float_state(state, "storage_capacity_mb", 4096.0),
        detection_metadata_mb=_float_state(
            state, "detection_metadata_mb", 0.01
        ),
        jetson_to_obc_rate_kbps=_float_state(
            state, "jetson_to_obc_rate_kbps", 8000.0
        ),
        downlink_rate_kbps=_float_state(state, "downlink_rate_kbps", 50.0),
        step_duration_s=_float_state(state, "step_duration_s", 60.0),
    )


def _contact_seconds(state: Dict[str, Any], params: PipelineParameters) -> float:
    if not bool(state.get("ground_pass_active", False)):
        return 0.0
    if "contact_window_seconds" in state:
        return max(0.0, _float_state(state, "contact_window_seconds", 0.0))
    # Compatibility for older/synthetic declared observations. Production
    # telemetry always carries the second-accurate contact-plan value.
    return params.step_duration_s


def _minimum_soc(state: Dict[str, Any], mode: str) -> float:
    configured = state.get("mode_min_battery_soc", {})
    if isinstance(configured, dict) and mode in configured:
        try:
            return float(configured[mode])
        except (TypeError, ValueError):
            pass
    return _DEFAULT_MODE_MIN_SOC.get(mode, _SOC_HARD_LIMIT)


def _critical_soc(state: Dict[str, Any]) -> float:
    return _float_state(state, "battery_min_soc", _SOC_HARD_LIMIT)


def _get_feasible_modes(state: Dict[str, Any]) -> List[str]:
    """Return modes admitted by the same projections used by the tool call."""

    return [
        mode for mode in VALID_MODES if check_constraints(state, mode)["feasible"]
    ]


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
    soc = _float_state(state, "battery_soc", 0.5)
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
    achievable = state.get("achievable_downlink_mb")

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
        "achievable_downlink_mb": round(float(achievable), 2) if achievable is not None else None,
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
    """Project a proposed mode through the canonical physical contract."""
    soc = _float_state(state, "battery_soc", 0.5)
    pass_active = state.get("ground_pass_active", False)
    health = state.get("health_status", "nominal")
    params = _pipeline_parameters(state)

    violations: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    pipeline_productive = True

    # Hard constraints
    if health != "nominal" and proposed_mode != "safe":
        violations.append({
            "constraint": "anomaly",
            "reason": f"Anomaly active ({health}); only safe mode allowed.",
        })

    critical_soc = _critical_soc(state)
    if soc <= critical_soc and proposed_mode != "safe":
        violations.append({
            "constraint": "battery_critical",
            "reason": (
                f"SoC {soc:.2f} at/below critical limit {critical_soc:.2f}; "
                "the environment resolves every non-safe request to safe mode."
            ),
        })
    elif (
        proposed_mode in _DEFAULT_MODE_MIN_SOC
        and soc < _minimum_soc(state, proposed_mode)
    ):
        threshold = _minimum_soc(state, proposed_mode)
        violations.append({
            "constraint": "mode_battery",
            "reason": (
                f"SoC {soc:.2f} below {proposed_mode} minimum {threshold:.2f}; "
                "the environment resolves this request to charging."
            ),
        })

    contact_seconds = _contact_seconds(state, params)
    if proposed_mode == "communication" and (not pass_active or contact_seconds <= 0.0):
        violations.append({
            "constraint": "ground_pass",
            "reason": "No positive declared contact duration; cannot communicate.",
        })

    if proposed_mode == "payload_observe":
        outcome = apply_observe(state, params)
        pipeline_productive = outcome.accepted
        if not outcome.accepted:
            violations.append({
                "constraint": "jetson_capacity",
                "reason": "The complete observation would exceed shared Jetson capacity.",
            })
    elif proposed_mode == "payload_compress":
        pipeline_productive = can_compress(state, params)
        if not pipeline_productive:
            violations.append({
                "constraint": "source_backlog",
                "reason": "No complete raw observation is available to compress.",
            })
    elif proposed_mode == "payload_detect":
        has_detection_source = _float_state(
            state, "undetected_observations", 0.0
        ) >= 1.0
        detection_steps = max(
            1, int(math.ceil(_float_state(state, "detection_steps", 5.0)))
        )
        completion_due = (
            _float_state(state, "detection_progress", 0.0) + 1.0
            >= detection_steps
        )
        outcome = apply_detect(state, params) if completion_due else None
        pipeline_productive = has_detection_source and (
            outcome is None or outcome.accepted
        )
        if not has_detection_source or (outcome is not None and not outcome.accepted):
            reason = "no_undetected_product" if not has_detection_source else outcome.reason
            constraint = (
                "source_backlog"
                if reason == "no_undetected_product"
                else "obc_capacity"
            )
            violations.append({
                "constraint": constraint,
                "reason": (
                    "No undetected product is available."
                    if constraint == "source_backlog"
                    else "Detection metadata would exceed OBC capacity."
                ),
            })
    elif proposed_mode == "payload_send":
        outcome = apply_can_transfer(state, params)
        pipeline_productive = outcome.accepted
        if not outcome.accepted:
            violations.append({
                "constraint": (
                    "source_backlog"
                    if outcome.reason == "no_source_data"
                    else "obc_capacity"
                ),
                "reason": (
                    "No compressed Jetson backlog is available."
                    if outcome.reason == "no_source_data"
                    else "The OBC has no destination headroom."
                ),
            })
    elif proposed_mode == "communication" and pass_active and contact_seconds > 0.0:
        outcome = apply_downlink(state, params, contact_seconds=contact_seconds)
        pipeline_productive = outcome.accepted
        if not outcome.accepted and outcome.reason == "no_source_data":
            # An empty mission-data queue does not make the RF link invalid:
            # telemetry/uplink and plan activation can still physically resolve.
            warnings.append({
                "constraint": "source_backlog",
                "reason": "RF link is executable, but no mission data is queued.",
            })

    # Warnings
    if (
        soc < _SOC_PREFERRED
        and proposed_mode in _ENERGY_INTENSIVE_MODES
        and not any(v["constraint"].startswith("battery") or v["constraint"] == "mode_battery" for v in violations)
    ):
        warnings.append({
            "constraint": "battery_low",
            "reason": f"SoC {soc:.2f} below preferred {_SOC_PREFERRED}; consider charging first.",
        })

    if proposed_mode not in VALID_MODES:
        violations.append({
            "constraint": "invalid_mode",
            "reason": f"'{proposed_mode}' is not a valid EventSat mode.",
        })

    # Mirror environment action resolution before claiming that the requested
    # mode is productive this step. A maneuver request may be executable and
    # necessary to start settling even though the current step resolves to
    # charging.
    resolved_mode = proposed_mode if proposed_mode in VALID_MODES else "charging"
    if health != "nominal":
        resolved_mode = "safe"
    elif soc <= critical_soc:
        resolved_mode = "safe"
    elif (
        resolved_mode in _DEFAULT_MODE_MIN_SOC
        and soc < _minimum_soc(state, resolved_mode)
    ):
        resolved_mode = "charging"
    elif resolved_mode == "communication" and (
        not pass_active or contact_seconds <= 0.0
    ):
        resolved_mode = "charging"

    settling_steps = max(
        0, int(_float_state(state, "settling_time_steps", 0.0))
    )
    transition_remaining = max(
        0, int(_float_state(state, "transition_steps_remaining", 0.0))
    )
    previous_mode = str(
        state.get("previous_mode", state.get("current_mode", "charging"))
    )
    safety_forced = health != "nominal" or soc <= critical_soc
    if safety_forced:
        # Safety interventions preempt attitude settling in the environment.
        transition_remaining = 0
    elif settling_steps > 0 and transition_remaining > 0:
        # An ongoing slew keeps its initial target: this command is ignored,
        # not queued, and therefore neither a violation nor productive.
        target = state.get("transition_target_mode") or "its initial target"
        return {
            "proposed_mode": proposed_mode,
            "feasible": proposed_mode in VALID_MODES,
            "resolved_mode_this_step": "charging",
            "productive_this_step": False,
            "transition_steps_required": transition_remaining,
            "transition_target_mode": state.get("transition_target_mode"),
            "command_ignored": True,
            "violations": [] if proposed_mode in VALID_MODES else [{
                "constraint": "invalid_mode",
                "reason": f"'{proposed_mode}' is not a valid EventSat mode.",
            }],
            "warnings": [{
                "constraint": "attitude_settling",
                "reason": (
                    f"Slew toward {target} in progress ({transition_remaining} "
                    "settling step(s) left); this command is ignored and not "
                    "queued; this step resolves to charging."
                ),
            }],
        }
    attitude_modes = set(state.get("attitude_maneuver_modes") or [])
    starts_transition = (
        not safety_forced
        and settling_steps > 0
        and previous_mode != resolved_mode
        and (resolved_mode in attitude_modes or previous_mode in attitude_modes)
    )
    transition_steps_required = (
        transition_remaining
        if transition_remaining > 0
        else (settling_steps if starts_transition else 0)
    )
    resolved_mode_this_step = (
        "charging" if transition_steps_required > 0 else resolved_mode
    )
    if transition_steps_required > 0:
        warnings.append({
            "constraint": "attitude_settling",
            "reason": (
                f"Request starts {transition_steps_required} non-productive "
                f"settling step(s) toward {resolved_mode}; this step resolves "
                "to charging and later commands cannot retarget the slew."
            ),
        })

    if proposed_mode == "communication" and resolved_mode == "communication":
        remaining_contact_s = max(
            0.0,
            _float_state(
                state, "remaining_pass_duration_s", contact_seconds
            ),
        )
        settling_duration_s = transition_steps_required * params.step_duration_s
        if (
            transition_steps_required > 0
            and remaining_contact_s <= settling_duration_s
        ):
            violations.append({
                "constraint": "contact_too_short",
                "reason": (
                    f"Only {remaining_contact_s:.1f}s contact remains, not enough "
                    f"to outlast {settling_duration_s:.1f}s of settling."
                ),
            })

    productive_this_step = (
        len(violations) == 0
        and resolved_mode_this_step == proposed_mode
        and pipeline_productive
    )

    return {
        "proposed_mode": proposed_mode,
        "feasible": len(violations) == 0,
        "resolved_mode_this_step": resolved_mode_this_step,
        "productive_this_step": productive_this_step,
        "transition_steps_required": transition_steps_required,
        "transition_target_mode": resolved_mode if starts_transition else None,
        "command_ignored": False,
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


def _future_contact_capacity_mb(state: Dict[str, Any]) -> float:
    """Declared capacity of the pass for which pipeline work is being planned."""

    for key in ("planning_downlink_capacity_mb", "achievable_downlink_mb"):
        value = state.get(key)
        if value is not None:
            return max(0.0, _float_state(state, key, 0.0))
    return 0.0


def _obc_ready_by_future_contact(
    state: Dict[str, Any],
    params: PipelineParameters,
    available_steps: int,
) -> float:
    """Optimistic OBC-ready bytes after bounded onboard processing.

    The estimate uses only declared state and rates.  It chooses how many of the
    available single-action steps to spend compressing versus sending, then
    respects raw backlog, CAN throughput, and OBC headroom.  No mode-specific
    utility constants enter the estimate.
    """

    steps = max(0, int(available_steps))
    obc_mb = max(0.0, _float_state(state, "obc_data_mb", 0.0))
    compressed_mb = max(
        0.0, _float_state(state, "jetson_compressed_mb", 0.0)
    )
    raw_mb = max(0.0, _float_state(state, "jetson_raw_mb", 0.0))
    declared_products = max(
        0, int(_float_state(state, "uncompressed_observations", 0.0))
    )
    physical_products = int(
        (raw_mb + 1e-12) / max(params.observation_size_mb, 1e-12)
    )
    raw_products = min(declared_products, physical_products)
    compression_steps = max(
        1,
        int(
            math.ceil(
                _float_state(state, "compression_time_factor", 2.0)
            )
        ),
    )
    current_progress = max(
        0.0, _float_state(state, "compression_progress", 0.0)
    )
    first_completion_steps = max(
        1, int(math.ceil(max(0.0, compression_steps - current_progress)))
    )
    can_per_step_mb = max(
        0.0,
        params.jetson_to_obc_rate_kbps
        / 8.0
        * params.step_duration_s
        / 1000.0,
    )
    obc_headroom_mb = max(0.0, params.obc_capacity_mb - obc_mb)
    best_ready_mb = min(obc_mb, params.obc_capacity_mb)

    for send_steps in range(steps + 1):
        compress_actions = steps - send_steps
        completed_products = 0
        if raw_products > 0 and compress_actions >= first_completion_steps:
            completed_products = 1 + (
                compress_actions - first_completion_steps
            ) // compression_steps
            completed_products = min(raw_products, completed_products)
        sendable_mb = (
            compressed_mb
            + completed_products * params.compressed_observation_mb
        )
        transferred_mb = min(
            sendable_mb,
            send_steps * can_per_step_mb,
            obc_headroom_mb,
        )
        best_ready_mb = max(best_ready_mb, obc_mb + transferred_mb)

    return min(best_ready_mb, params.obc_capacity_mb)


@_register_tool(
    name="evaluate_plan",
    description="Evaluate a proposed mode using incremental contact-deliverable value and physical risk factors.",
    parameters={"state": "Current satellite state dict", "proposed_mode": "Mode to evaluate (string)"},
)
def evaluate_plan(
    state: Dict[str, Any], proposed_mode: str = "charging", **kwargs,
) -> Dict[str, Any]:
    """Score only incremental value that can move toward the ground archive.

    Archived pre-fix agentic results used fixed preferred-mode bonuses and are
    not comparable.  This projection now shares the environment's admission
    helpers, multi-step processing state, ADCS resolution, and declared contact
    capacity; selecting a named mode by itself earns no utility.
    """

    params = _pipeline_parameters(state)
    constraints = check_constraints(state, proposed_mode)
    risk_factors = [
        item["reason"]
        for item in constraints["violations"] + constraints["warnings"]
    ]
    deliverable_mb = 0.0
    immediate_delivery_mb = 0.0
    pipeline_progress_mb = 0.0
    processing_progress_steps = 0
    utility = 0.0
    projected = dict(state)

    if constraints["feasible"] and constraints["productive_this_step"]:
        if proposed_mode == "communication":
            outcome = apply_downlink(
                state, params, contact_seconds=_contact_seconds(state, params)
            )
            projected = outcome.state
            immediate_delivery_mb = outcome.transferred_mb
            deliverable_mb = immediate_delivery_mb
            physical_step_capacity = (
                params.downlink_rate_kbps
                / 8.0
                * _contact_seconds(state, params)
                / 1000.0
            )
            utility = deliverable_mb / max(physical_step_capacity, 1e-12)
        elif proposed_mode == "payload_send":
            outcome = apply_can_transfer(state, params)
            projected = outcome.state
            pipeline_progress_mb = outcome.transferred_mb
        elif proposed_mode == "payload_observe":
            outcome = apply_observe(state, params)
            if outcome.accepted:
                projected = outcome.state
                pipeline_progress_mb = params.compressed_observation_mb
        elif proposed_mode == "payload_compress":
            progress = _float_state(state, "compression_progress", 0.0) + 1.0
            required = max(
                1,
                int(
                    math.ceil(
                        _float_state(state, "compression_time_factor", 2.0)
                    )
                ),
            )
            if progress >= required:
                outcome = apply_compress(state, params)
                if outcome.accepted:
                    projected = outcome.state
                    projected["compression_progress"] = 0.0
                    pipeline_progress_mb = params.compressed_observation_mb
            else:
                projected["compression_progress"] = progress
                processing_progress_steps = 1
        elif proposed_mode == "payload_detect":
            progress = _float_state(state, "detection_progress", 0.0) + 1.0
            required = max(
                1,
                int(
                    math.ceil(_float_state(state, "detection_steps", 5.0))
                ),
            )
            if progress >= required:
                outcome = apply_detect(state, params)
                if outcome.accepted:
                    projected = outcome.state
                    projected["detection_progress"] = 0.0
                    pipeline_progress_mb = params.detection_metadata_mb
            else:
                projected["detection_progress"] = progress
                processing_progress_steps = 1

        if proposed_mode != "communication":
            future_capacity_mb = _future_contact_capacity_mb(state)
            time_to_contact = max(
                0,
                int(math.floor(_float_state(state, "time_to_next_pass", 0.0))),
            )
            remaining_processing_steps = max(0, time_to_contact - 1)
            baseline_ready_mb = _obc_ready_by_future_contact(
                state, params, remaining_processing_steps
            )
            projected_ready_mb = _obc_ready_by_future_contact(
                projected, params, remaining_processing_steps
            )
            deliverable_mb = max(
                0.0,
                min(projected_ready_mb, future_capacity_mb)
                - min(baseline_ready_mb, future_capacity_mb),
            )
            utility = deliverable_mb / max(future_capacity_mb, 1e-12)

    useful_progress = (
        deliverable_mb > 0.0
        or pipeline_progress_mb > 0.0
        or processing_progress_steps > 0
    )
    if (
        constraints["feasible"]
        and constraints["transition_steps_required"] > 0
    ):
        # Starting/continuing a feasible maneuver is necessary progress, but it
        # receives no mission utility until a later step moves deliverable data.
        useful_progress = True
    if proposed_mode == "charging":
        useful_progress = _float_state(state, "battery_soc", 0.5) < _SOC_PREFERRED
    elif proposed_mode == "safe":
        useful_progress = state.get("health_status", "nominal") != "nominal"
    if constraints["command_ignored"]:
        # Every valid command is equivalent while an ongoing slew completes.
        useful_progress = True
    recommendation = (
        "proceed"
        if constraints["feasible"] and useful_progress
        else "not_proceed"
    )

    return {
        "proposed_mode": proposed_mode,
        "estimated_utility": round(max(0.0, min(1.0, utility)), 2),
        "estimated_deliverable_mb": round(deliverable_mb, 6),
        "estimated_immediate_delivery_mb": round(immediate_delivery_mb, 6),
        "pipeline_progress_mb": round(pipeline_progress_mb, 6),
        "processing_progress_steps": processing_progress_steps,
        "feasible": constraints["feasible"],
        "productive_this_step": constraints["productive_this_step"],
        "resolved_mode_this_step": constraints["resolved_mode_this_step"],
        "transition_steps_required": constraints["transition_steps_required"],
        "command_ignored": constraints["command_ignored"],
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

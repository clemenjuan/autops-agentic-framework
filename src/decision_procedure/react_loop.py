"""
ReAct (Reason-Act-Observe) Decision Loop.

Yao et al. (2023), "ReAct: Synergizing Reasoning and Acting in Language
Models", ICLR 2023. Adapted here for symbolic/rule-based agents following
the application in Li (2025) "AI Agents for Satellite Operations" (see
Zotero library for the canonical reference).

The ReAct loop implements an iterative Thought-Action-Observation cycle:

  1. Thought  — Representation.reason(state, memory) produces a structured
                reasoning trace that explains the decision factors at play.
                For symbolic representations this is a rule evaluation
                trace; for future LLM representations it would be chain-of-
                thought text.

  2. Action   — Representation.select_action(context) proposes an action,
                with the reasoning trace passed in the enrichments dict so
                the representation can use it as context.

  3. Observation — Grounding checks validate the proposed action against
                physics and operational constraints. If violations are found,
                they are fed back as the "Observation" (Yao et al.'s
                key insight: grounding replaces environment feedback in the
                reasoning loop). The cycle repeats.

Termination:
  - Stops when the action passes all grounding checks (converged=True).
  - Falls back to charging if max_iterations is reached without convergence.

Key differences from other loops:
  - vs SDA:  Adds reasoning trace and iterative refinement.
  - vs OODA: No fixed 4-phase structure; convergence-based iteration.
             OODA's situation classification enriches a single decision;
             ReAct re-reasons until constraints are satisfied.

Config:
    max_iterations (int, default 3):
        Maximum reason-act-observe cycles per step.
    grounding_checks (list[str], default ["battery_feasibility", "pass_window_timing"]):
        Constraints to validate after each action proposal.
        - "battery_feasibility": energy-intensive modes require SoC ≥ 0.30
        - "pass_window_timing": communication mode only during ground passes

Metrics:
    decision_latency_s:    Total latency for the ReAct cycle.
    reasoning_depth:       Total thought steps across all iterations.
    iterations:            Reasoning cycles to convergence (or max).
    grounding_violations:  Total constraint violations across all iterations.
    converged:             1.0 if action passed grounding, 0.0 if fallback.
    has_rationale:         1.0 if representation provides a rationale string.
    total_decisions:       Cumulative steps processed.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from src.decision_procedure.base import DecisionProcedure
from src.decision_procedure.context import DecisionContext
from src.representation.base import Representation

# Modes that require meaningful battery reserve
_ENERGY_INTENSIVE_MODES = {
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "communication",
}

# Minimum SoC for energy-intensive operations (grounding threshold)
_GROUNDING_MIN_SOC = 0.30


class ReActLoop(DecisionProcedure):
    """Reason-Act-Observe iterative decision loop (Yao et al. 2023).

    Implements the ReAct cycle: at each step the representation is asked to
    reason about the current state (Thought), propose an action (Action),
    and the loop validates that action against operational constraints
    (Observation). If grounding fails the loop re-invokes the representation
    with the constraint violations as feedback.
    """

    def __init__(self, config: Dict[str, Any], representation: Representation) -> None:
        self.config = config
        self.representation = representation

        self._max_iterations: int = config.get("max_iterations", 3)
        self._grounding_checks: List[str] = config.get(
            "grounding_checks",
            ["battery_feasibility", "pass_window_timing"],
        )

        # Metrics state
        self._last_total_latency: float = 0.0
        self._last_reasoning_depth: int = 0
        self._last_iterations: int = 0
        self._last_grounding_violations: int = 0
        self._last_converged: bool = True
        self._last_has_rationale: bool = False
        self._total_steps: int = 0

    # ------------------------------------------------------------------
    # DecisionProcedure interface
    # ------------------------------------------------------------------

    def process(
        self, observation: Any, memory: Any
    ) -> Tuple[Dict[str, Any], Any]:
        """Execute one ReAct cycle: Thought → Action → Observation (× max_iterations)."""
        t_total = time.perf_counter()

        # Unwrap AgentObservation if needed (same as SDA/OODA)
        raw_obs = observation
        if hasattr(observation, "local_state") and isinstance(
            observation.local_state, dict
        ):
            raw_obs = observation.local_state.get("full_observation", observation)

        state = self.representation.encode_observation(raw_obs)

        # Accumulated reasoning and violation state across iterations
        all_thought_steps: List[Dict[str, Any]] = []
        all_violations: List[Dict[str, Any]] = []
        action: Dict[str, Any] = {"eventsat_0": {"mode": "charging"}}
        converged = False

        for iteration in range(self._max_iterations):
            # === THOUGHT: reason about state + accumulated violations ===
            if hasattr(self.representation, "reason"):
                thought_steps = self.representation.reason(state, memory) or []
            else:
                thought_steps = []
            all_thought_steps.extend(thought_steps)

            # === ACTION: propose action with full reasoning context ===
            context = DecisionContext(
                state=state,
                loop_type="react",
                memory=memory,
                enrichments={
                    "reasoning_trace": list(all_thought_steps),
                    "iteration": iteration,
                    "grounding_violations": list(all_violations),
                },
                loop_metadata={
                    "max_iterations": self._max_iterations,
                    "grounding_checks": self._grounding_checks,
                },
            )
            action = self.representation.select_action(context)

            # === OBSERVATION: validate action against constraints ===
            violations = self._check_grounding(action, state)

            if not violations:
                converged = True
                break

            # Feed violations back into next iteration's context
            all_violations.extend(violations)

        # Fallback if all iterations failed grounding
        if not converged:
            action = {"eventsat_0": {"mode": "charging"}}

        # Bookkeeping
        self._last_total_latency = time.perf_counter() - t_total
        self._last_reasoning_depth = len(all_thought_steps)
        self._last_iterations = (
            self._max_iterations if not converged else
            len(all_thought_steps) // max(1, self._max_iterations) + 1
        )
        self._last_grounding_violations = len(all_violations)
        self._last_converged = converged
        self._last_has_rationale = (
            hasattr(self.representation, "get_rationale")
            and self.representation.get_rationale() is not None
        )
        self._total_steps += 1

        return action, memory

    def get_metrics(self) -> Dict[str, Any]:
        """Return ReAct-specific metrics (Yao et al. framework)."""
        rationale = ""
        if hasattr(self.representation, "get_rationale"):
            rationale = self.representation.get_rationale() or ""
        return {
            "decision_latency_s": self._last_total_latency,
            "reasoning_depth": float(self._last_reasoning_depth),
            "iterations": float(self._last_iterations),
            "grounding_violations": float(self._last_grounding_violations),
            "converged": float(self._last_converged),
            "has_rationale": float(self._last_has_rationale),
            "total_decisions": float(self._total_steps),
            "rationale": rationale,
        }

    def reset(self) -> None:
        """Reset internal state at episode start."""
        self._last_total_latency = 0.0
        self._last_reasoning_depth = 0
        self._last_iterations = 0
        self._last_grounding_violations = 0
        self._last_converged = True
        self._last_has_rationale = False
        self._total_steps = 0

    def get_name(self) -> str:
        return "ReActLoop"

    # ------------------------------------------------------------------
    # Grounding: Observation phase of ReAct
    # ------------------------------------------------------------------

    def _check_grounding(
        self,
        action: Dict[str, Any],
        state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Validate action against operational constraints.

        Returns a list of violation dicts. Empty list means action is valid.
        Each violation has: check, mode, reason.
        """
        violations: List[Dict[str, Any]] = []
        if not action or not state:
            return violations

        sat_action = action.get("eventsat_0", {})
        mode = sat_action.get("mode", "charging")
        soc = state.get("battery_soc", 0.5)
        pass_active = state.get("ground_pass_active", False)

        if "battery_feasibility" in self._grounding_checks:
            if mode in _ENERGY_INTENSIVE_MODES and soc < _GROUNDING_MIN_SOC:
                violations.append({
                    "check": "battery_feasibility",
                    "mode": mode,
                    "reason": (
                        f"SoC={soc:.2f} < {_GROUNDING_MIN_SOC} threshold "
                        f"for energy-intensive mode '{mode}'"
                    ),
                })

        if "pass_window_timing" in self._grounding_checks:
            if mode == "communication" and not pass_active:
                violations.append({
                    "check": "pass_window_timing",
                    "mode": mode,
                    "reason": "communication mode requires an active ground pass",
                })

        return violations

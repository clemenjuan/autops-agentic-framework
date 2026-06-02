"""Autonomous Hybrid operations paradigm.

Dual onboard/ground operations. Onboard autonomy (rules, small DNNs)
operates with full real-time state every step. Ground systems (LLM,
planners) prepare detailed analysis and plans between passes using
last-received telemetry, uplinked during next contact. Onboard can
override ground plans for fault detection or opportunistic events.

In EventSat, no LLM runs onboard — only small DNNs are feasible.
The 122B Qwen model represents ground-side reasoning capability.

Known simplification: the current implementation gives all representations
real-time state every step (filter_observation returns full state). In
reality, ground-based representations (LLM, agentic) would only see stale
telemetry between passes. A future improvement would add dual observation
paths (onboard=real-time, ground=stale).
"""

from __future__ import annotations

from typing import Any, Dict

from src.operations.base import OperationsParadigm


class AutonomousHybrid(OperationsParadigm):
    """Dual onboard/ground operations — onboard real-time, ground between passes."""

    def filter_observation(self, full_observation: Any, step: int) -> Any:
        return full_observation

    def can_act(self, step: int, ground_pass_active: bool) -> bool:
        return True

    def should_allow_inference(self, step: int, ground_pass_active: bool) -> bool:
        """Allow inference every step (simulation simplification).

        In reality, ground-based representations (LLM) would only compute
        between passes with stale telemetry. Onboard representations (rules,
        small DNNs) can compute every step. Current implementation does not
        distinguish — see module docstring for known simplification.
        """
        return True

    def can_self_recover_anomaly(self) -> bool:
        """Onboard FDIR can clear anomalies without ground contact."""
        return True

    def has_onboard_autonomy(self) -> bool:
        """AH runs the onboard per-step core — Jetson powered every step."""
        return True

    def process_action(
        self,
        action: Dict[str, Any],
        step: int,
        ground_pass_active: bool,
    ) -> Dict[str, Any]:
        return action

"""Autonomous Onboard operations paradigm.

Pure onboard autonomy: the satellite runs a single per-step core (rules or a
small DNN/RL policy) closed-loop on full real-time state every step. There is
**no ground planner and no uplinked schedule** — every decision is made onboard.

This is the clean "onboard only" primitive of the operations ladder:
    autonomous_onboard (onboard only)  →  autonomous_hybrid (onboard + ground)
    →  autonomous_ground (ground only)  →  conventional_ground (human ground)

Contrast:
- vs AutonomousHybrid: AH additionally runs a ground planner (uplinked plan) that
  the onboard core can override; AO has no ground plan at all.
- vs AutonomousGround: AG executes a ground-produced schedule open-loop with no
  onboard autonomy; AO is the opposite (fully onboard, no schedule).

Meaningful only for substrates with an onboard core (symbolic rules, subsymbolic
RL). A bare hybrid/LLM has no onboard core (its LLM is a ground component), so
hybrid + autonomous_onboard is excluded at config validation.
"""

from __future__ import annotations

from typing import Any, Dict

from src.core.operations.base import OperationsParadigm


class AutonomousOnboard(OperationsParadigm):
    """Onboard-only operations — one per-step core, real-time, closed-loop."""

    def filter_observation(self, full_observation: Any, step: int) -> Any:
        return full_observation

    def can_act(self, step: int, ground_pass_active: bool) -> bool:
        return True

    def should_allow_inference(self, step: int, ground_pass_active: bool) -> bool:
        """Onboard inference runs every step on real-time state."""
        return True

    def can_self_recover_anomaly(self) -> bool:
        """Onboard FDIR can clear anomalies without ground contact."""
        return True

    def has_onboard_autonomy(self) -> bool:
        """Decisions run onboard — Jetson powered every step."""
        return True

    def process_action(
        self,
        action: Dict[str, Any],
        step: int,
        ground_pass_active: bool,
    ) -> Dict[str, Any]:
        return action

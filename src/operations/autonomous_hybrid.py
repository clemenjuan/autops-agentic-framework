"""Autonomous Hybrid operations paradigm.

The agent operates autonomously with full real-time access to all
satellite state, whether running onboard or on the ground. Actions
are applied immediately every timestep. This is the default paradigm.
"""

from __future__ import annotations

from typing import Any, Dict

from src.operations.base import OperationsParadigm


class AutonomousHybrid(OperationsParadigm):
    """Autonomous operations (onboard or ground) — real-time state, immediate actions."""

    def filter_observation(self, full_observation: Any, step: int) -> Any:
        return full_observation

    def can_act(self, step: int, ground_pass_active: bool) -> bool:
        return True

    def process_action(
        self,
        action: Dict[str, Any],
        step: int,
        ground_pass_active: bool,
    ) -> Dict[str, Any]:
        return action

"""Reward function for the SSA constellation scenario."""
from __future__ import annotations

from typing import Any, Dict

from src.eventsat.rewards import MultiEventsatRewardFunction


class SSARewardFunction(MultiEventsatRewardFunction):
    """Collective-negative SSA reward using delivered ground coverage.

    The inherited local/team blend preserves the per-satellite reward contract.
    SSA adds a shared mission term based on delivered-to-ground coverage, so
    onboard-only hoarding does not earn M-01 credit.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.collective_weight = float(cfg.get("collective_weight", 1.0))
        self.mission_scale = float(cfg.get("mission_scale", 1.0))
        self.negative = bool(cfg.get("collective_negative", True))

    def compute_rewards(
        self,
        individual_rewards: Dict[str, float],
        per_satellite_inputs: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, float]:
        rewards = super().compute_rewards(individual_rewards, per_satellite_inputs)
        global_inputs = (per_satellite_inputs or {}).get("_global", {})
        delivered = float(global_inputs.get("delivered_coverage", 0.0))
        if self.negative:
            mission_term = -self.mission_scale * (1.0 - delivered)
        else:
            mission_term = self.mission_scale * delivered
        return {
            sat_id: value + self.collective_weight * mission_term
            for sat_id, value in rewards.items()
        }

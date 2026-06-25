"""SSA metrics collector."""
from __future__ import annotations

from typing import Any, Dict, List

from src.core.metrics_collector import EpisodeMetrics, StepMetrics
from src.eventsat.metrics import EventSatMetricsCollector


class SSAMetricsCollector(EventSatMetricsCollector):
    """EventSat metrics plus SSA coverage, duplication, ISL, and M-10 hooks."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._baseline_utility_n1 = float(self.config.get("baseline_utility_n1", 0.0))
        self._constellation_size = int(self.config.get("constellation_size", 1))

    def collect_step_metrics(
        self,
        timestep: int,
        wall_clock_seconds: float,
        env_state: Any,
        actions: Any,
        rewards: Dict[str, float],
        info: Dict[str, Any],
        decision_metrics: Dict[str, Any],
    ) -> StepMetrics:
        step = super().collect_step_metrics(
            timestep=timestep,
            wall_clock_seconds=wall_clock_seconds,
            env_state=env_state,
            actions=actions,
            rewards=rewards,
            info=info,
            decision_metrics=decision_metrics,
        )
        step.metrics.update({
            "ssa_onboard_coverage": float(info.get("ssa_onboard_coverage", 0.0)),
            "ssa_delivered_coverage": float(info.get("ssa_delivered_coverage", 0.0)),
            "duplicate_observation_rate": float(info.get("duplicate_observation_rate", 0.0)),
            "mean_revisit_steps": float(info.get("mean_revisit_steps", 0.0)),
            "isl_connectivity": float(info.get("isl_connectivity", 0.0)),
            "ssa_delivered_objects": float(info.get("ssa_delivered_objects", 0.0)),
            "ssa_known_objects": float(info.get("ssa_known_objects", 0.0)),
        })
        return step

    def aggregate_episode_metrics(self, step_metrics: List[StepMetrics]) -> EpisodeMetrics:
        episode = super().aggregate_episode_metrics(step_metrics)
        if not step_metrics:
            return episode
        last = step_metrics[-1].metrics
        delivered_coverage = float(last.get("ssa_delivered_coverage", 0.0))
        utility = delivered_coverage
        if self._baseline_utility_n1 > 0.0 and self._constellation_size > 0:
            eta_scale = (utility / self._constellation_size) / self._baseline_utility_n1
        else:
            eta_scale = 0.0
        episode.aggregated.update({
            "utility": utility,
            "ssa_onboard_coverage": float(last.get("ssa_onboard_coverage", 0.0)),
            "ssa_delivered_coverage": delivered_coverage,
            "duplicate_observation_rate": float(last.get("duplicate_observation_rate", 0.0)),
            "mean_revisit_steps": float(last.get("mean_revisit_steps", 0.0)),
            "isl_connectivity": float(last.get("isl_connectivity", 0.0)),
            "ssa_delivered_objects": float(last.get("ssa_delivered_objects", 0.0)),
            "ssa_known_objects": float(last.get("ssa_known_objects", 0.0)),
            "eta_scale": eta_scale,
        })
        return episode

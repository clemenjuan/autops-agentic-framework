"""
EventSat-specific Metrics Collector.

Tracks satellite operations metrics: observation time, downlink volume,
battery health, forced mode changes, anomalies.
"""
from __future__ import annotations
from typing import Any, Dict, List
from src.orchestration.metrics_collector import (
    EpisodeMetrics,
    ExperimentStatistics,
    MetricsCollector,
    StepMetrics,
)


class EventSatMetricsCollector(MetricsCollector):
    """Collects and aggregates EventSat-specific metrics."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def collect_step_metrics(
        self,
        step: int,
        observation: Any,
        actions: Dict[str, Any],
        result: Any,
        decision_metrics: Dict[str, Any],
    ) -> StepMetrics:
        env_metrics = {}
        if hasattr(result, "info"):
            env_metrics = result.info or {}
        reward = sum(result.rewards.values()) if hasattr(result, "rewards") and result.rewards else 0.0
        return StepMetrics(
            step=step,
            reward=reward,
            environment_metrics=env_metrics,
            decision_metrics=decision_metrics,
        )

    def aggregate_episode_metrics(
        self, episode_id: int, step_metrics: List[StepMetrics]
    ) -> EpisodeMetrics:
        total_reward = sum(s.reward for s in step_metrics)
        n = len(step_metrics)

        forced_count = sum(
            1 for s in step_metrics
            if s.environment_metrics.get("forced", False)
        )
        anomaly_count = sum(
            1 for s in step_metrics
            if s.environment_metrics.get("anomaly") is not None
        )
        avg_latency = 0.0
        if n > 0:
            latencies = [
                s.decision_metrics.get("decision_latency_s", 0.0)
                for s in step_metrics
            ]
            avg_latency = sum(latencies) / n

        last = step_metrics[-1] if step_metrics else None
        obs_hours = 0.0
        dl_mb = 0.0
        final_soc = 0.0
        if last and last.environment_metrics:
            obs_hours = last.environment_metrics.get("observation_hours", 0.0)
            dl_mb = last.environment_metrics.get("data_downlinked_mb", 0.0)
            final_soc = last.environment_metrics.get("battery_soc", 0.0)

        return EpisodeMetrics(
            episode_id=episode_id,
            total_reward=total_reward,
            steps=n,
            custom_metrics={
                "observation_hours": obs_hours,
                "downlinked_mb": dl_mb,
                "final_battery_soc": final_soc,
                "forced_mode_changes": forced_count,
                "anomaly_events": anomaly_count,
                "avg_decision_latency_s": avg_latency,
            },
        )

    def compute_statistics(
        self, episode_metrics: List[EpisodeMetrics]
    ) -> ExperimentStatistics:
        if not episode_metrics:
            return ExperimentStatistics(
                mean_reward=0.0,
                std_reward=0.0,
                mean_steps=0.0,
                custom_statistics={},
            )
        rewards = [e.total_reward for e in episode_metrics]
        steps = [e.steps for e in episode_metrics]
        n = len(rewards)
        mean_r = sum(rewards) / n
        var_r = sum((r - mean_r) ** 2 for r in rewards) / max(1, n - 1)
        std_r = var_r ** 0.5

        def avg_custom(key: str) -> float:
            vals = [e.custom_metrics.get(key, 0.0) for e in episode_metrics]
            return sum(vals) / len(vals) if vals else 0.0

        return ExperimentStatistics(
            mean_reward=mean_r,
            std_reward=std_r,
            mean_steps=sum(steps) / n,
            custom_statistics={
                "mean_observation_hours": avg_custom("observation_hours"),
                "mean_downlinked_mb": avg_custom("downlinked_mb"),
                "mean_final_soc": avg_custom("final_battery_soc"),
                "mean_forced_modes": avg_custom("forced_mode_changes"),
                "mean_anomalies": avg_custom("anomaly_events"),
                "mean_decision_latency_s": avg_custom("avg_decision_latency_s"),
            },
        )

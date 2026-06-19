"""
Flamingo-lite metrics collector.

This collector keeps the board-facing result shape close to EventSat while
adding the SSA coordination metrics needed for the organization sweep.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.orchestration.metrics_collector import (
    EpisodeMetrics,
    ExperimentStatistics,
    MetricsCollector,
    StepMetrics,
)


class FlamingoMetricsCollector(MetricsCollector):
    """Collect and aggregate Flamingo-lite SSA metrics."""

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
        reward = sum(rewards.values()) if rewards else 0.0
        snapshot = dict(info.get("metrics", {}))
        metrics = {
            **snapshot,
            "decision_latency_s": decision_metrics.get("decision_latency_s", 0.0),
            "inference_allowed": float(decision_metrics.get("inference_allowed", True)),
            "has_rationale": float(decision_metrics.get("has_rationale", False)),
            "step_successful_observations": float(
                info.get("step_successful_observations", 0.0)
            ),
            "step_duplicate_observations": float(
                info.get("step_duplicate_observations", 0.0)
            ),
            "step_constraint_violations": float(
                info.get("step_constraint_violations", 0.0)
            ),
            "coordination_messages": float(
                decision_metrics.get("coordination_messages", 0.0)
            ),
        }
        return StepMetrics(
            timestep=timestep,
            wall_clock_seconds=wall_clock_seconds,
            reward=reward,
            metrics=metrics,
            metadata={"actions": actions},
        )

    def aggregate_episode_metrics(
        self, step_metrics: List[StepMetrics]
    ) -> EpisodeMetrics:
        n = len(step_metrics)
        if n == 0:
            return EpisodeMetrics()

        total_reward = sum(s.reward for s in step_metrics)
        total_wall = sum(s.wall_clock_seconds for s in step_metrics)
        last = step_metrics[-1].metrics
        decision_steps = [
            s for s in step_metrics if s.metrics.get("inference_allowed", 1.0) > 0
        ]
        mean_latency = (
            sum(s.metrics.get("decision_latency_s", 0.0) for s in decision_steps)
            / len(decision_steps)
            if decision_steps else 0.0
        )
        explainability_score = (
            sum(1 for s in decision_steps if s.metrics.get("has_rationale", 0.0) > 0)
            / len(decision_steps)
            if decision_steps else 0.0
        )

        utility = float(last.get("utility", total_reward))
        aggregated = {
            "utility": utility,
            "mean_latency_s": mean_latency,
            "max_latency_s": max(
                s.metrics.get("decision_latency_s", 0.0) for s in step_metrics
            ),
            "explainability_score": explainability_score,
            "coverage_rate": float(last.get("coverage_rate", 0.0)),
            "successful_observations": float(
                last.get("successful_observations", 0.0)
            ),
            "duplicate_observation_rate": float(
                last.get("duplicate_observation_rate", 0.0)
            ),
            "constraint_violation_rate": float(
                last.get("constraint_violation_rate", 0.0)
            ),
            "mean_revisit_steps": float(last.get("mean_revisit_steps", 0.0)),
            "resource_efficiency": (
                utility / self.config.get("constellation_size", 1)
                if self.config.get("constellation_size", 1) > 0 else 0.0
            ),
            "operator_load": float(last.get("duplicate_observation_rate", 0.0)),
            "coordination_messages": (
                sum(s.metrics.get("coordination_messages", 0.0) for s in step_metrics)
                / n
            ),
        }
        return EpisodeMetrics(
            num_steps=n,
            total_wall_clock_seconds=total_wall,
            total_reward=total_reward,
            aggregated=aggregated,
            step_metrics=step_metrics,
        )

    def compute_statistics(
        self, episode_metrics: List[EpisodeMetrics]
    ) -> ExperimentStatistics:
        if not episode_metrics:
            return ExperimentStatistics()

        n = len(episode_metrics)
        all_keys: set[str] = {"total_reward"}
        for episode in episode_metrics:
            all_keys.update(episode.aggregated.keys())

        mean: Dict[str, float] = {}
        std: Dict[str, float] = {}
        min_val: Dict[str, float] = {}
        max_val: Dict[str, float] = {}

        for key in sorted(all_keys):
            vals = (
                [episode.total_reward for episode in episode_metrics]
                if key == "total_reward"
                else [episode.aggregated.get(key, 0.0) for episode in episode_metrics]
            )
            avg = sum(vals) / n
            var = sum((value - avg) ** 2 for value in vals) / max(1, n - 1)
            mean[key] = avg
            std[key] = var ** 0.5
            min_val[key] = min(vals)
            max_val[key] = max(vals)

        utility_mean = mean.get("utility", 0.0)
        utility_std = std.get("utility", 0.0)
        mean["robustness_cv"] = (
            utility_std / utility_mean if utility_mean > 0 else 0.0
        )

        return ExperimentStatistics(
            num_episodes=n,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            raw_episodes=episode_metrics,
        )


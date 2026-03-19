"""
EventSat-specific Metrics Collector.

Implements the research metrics for the EventSat scenario.

Research metrics (per episode):
  utility            — mission objective achievement ratio
                       = w_obs x (obs_h / scaled_obs_target)
                       + w_dl  x (dl_mb / scaled_dl_target)
                       - w_anomaly x anomaly_rate
                       Targets scaled from 90-day mission to episode length.
  data_downlink_efficiency — fraction of available downlink capacity actually used
                        = data_downlinked_mb / max_achievable_downlink_mb
                        where max_achievable = total_pass_duration_s x S-band rate
                        Source: Proposal — "useful obs time limited by downlink capacity"
  mean_latency_s      — mean decision latency per step
  robustness_mean_recovery_steps — mean steps to recover after anomaly onset
  resource_efficiency — utility / total_energy_consumed_wh
  operator_load       — fraction of steps with environment safety overrides
  explainability_score — fraction of steps with a rationale string

Step-level metrics tracked:
  battery_soc, data_stored_mb, data_downlinked_mb, observation_hours,
  total_detections, max_achievable_downlink_mb,
  jetson_raw_mb, jetson_compressed_mb, obc_data_mb,
  in_sunlight, ground_pass_active, in_transition,
  forced, anomaly, safety_override, energy_consumed_wh,
  decision_latency_s, has_rationale
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
    """Collects and aggregates EventSat-specific + research metrics."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        # Utility weights (configurable via experiment YAML metrics section)
        uw = self.config.get("utility_weights", {})
        self._w_obs = uw.get("observation", 0.5)
        self._w_dl = uw.get("downlink", 0.4)
        self._w_anomaly = uw.get("anomaly_penalty", 0.1)

        # Targets from scenario (can be overridden in config)
        targets = self.config.get("utility_targets", {})
        self._obs_target_h = targets.get("observation_hours", 2.0)
        self._dl_target_mb = targets.get("downlinked_mb", 240.0)

        # Episode length for target scaling
        self._episode_steps = self.config.get("max_steps", 10080)
        self._step_duration_s = self.config.get("step_duration_s", 60.0)
        # Mission duration from scenario is 90 days; scale targets to episode
        mission_days = targets.get("mission_duration_days", 90.0)
        episode_days = (self._episode_steps * self._step_duration_s) / 86400.0
        self._target_scale = episode_days / mission_days if mission_days > 0 else 1.0
        self._scaled_obs_target = self._obs_target_h * self._target_scale
        self._scaled_dl_target = self._dl_target_mb * self._target_scale

        # Battery capacity for energy computation
        self._battery_capacity_wh = self.config.get("battery_capacity_wh", 84.0)

    # ------------------------------------------------------------------
    # Per-step collection
    # ------------------------------------------------------------------

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

        # Energy consumed this step (estimated from SoC delta)
        prev_soc = info.get("prev_battery_soc", info.get("battery_soc", 0.0))
        curr_soc = info.get("battery_soc", prev_soc)
        soc_delta = prev_soc - curr_soc  # positive = consumed
        energy_consumed_wh = max(0.0, soc_delta * self._battery_capacity_wh)

        metrics = {
            # Environment telemetry
            "battery_soc": info.get("battery_soc", 0.0),
            "data_stored_mb": info.get("data_stored_mb", 0.0),
            "data_downlinked_mb": info.get("data_downlinked_mb", 0.0),
            "observation_hours": info.get("observation_hours", 0.0),
            "total_detections": info.get("total_detections", 0),
            "max_achievable_downlink_mb": info.get("max_achievable_downlink_mb", 0.0),
            "in_sunlight": info.get("in_sunlight", 0.0),
            "ground_pass_active": info.get("ground_pass_active", 0.0),
            # Safety & anomaly
            "forced": float(info.get("forced", False)),
            "anomaly": float(info.get("anomaly") is not None and info.get("anomaly") is not False),
            "safety_override": float(info.get("forced", False)),
            # 3-pool data pipeline telemetry
            "jetson_raw_mb": info.get("jetson_raw_mb", 0.0),
            "jetson_compressed_mb": info.get("jetson_compressed_mb", 0.0),
            "obc_data_mb": info.get("obc_data_mb", 0.0),
            "in_transition": info.get("in_transition", 0.0),
            # Energy
            "energy_consumed_wh": energy_consumed_wh,
            # Decision loop metrics
            "decision_latency_s": decision_metrics.get("decision_latency_s", 0.0),
            "has_rationale": float(decision_metrics.get("has_rationale", False)),
        }

        return StepMetrics(
            timestep=timestep,
            wall_clock_seconds=wall_clock_seconds,
            reward=reward,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Episode aggregation (includes all 7 research metrics)
    # ------------------------------------------------------------------

    def aggregate_episode_metrics(
        self, step_metrics: List[StepMetrics]
    ) -> EpisodeMetrics:
        n = len(step_metrics)
        if n == 0:
            return EpisodeMetrics()

        total_reward = sum(s.reward for s in step_metrics)
        total_wall = sum(s.wall_clock_seconds for s in step_metrics)

        last = step_metrics[-1]
        obs_hours = last.metrics.get("observation_hours", 0.0)
        dl_mb = last.metrics.get("data_downlinked_mb", 0.0)
        final_soc = last.metrics.get("battery_soc", 0.0)

        # --- Research Metric 1: Utility ---
        obs_ratio = obs_hours / self._scaled_obs_target if self._scaled_obs_target > 0 else 0.0
        dl_ratio = dl_mb / self._scaled_dl_target if self._scaled_dl_target > 0 else 0.0
        anomaly_count = sum(1 for s in step_metrics if s.metrics.get("anomaly", 0.0) > 0)
        anomaly_rate = anomaly_count / n
        utility = (
            self._w_obs * obs_ratio
            + self._w_dl * dl_ratio
            - self._w_anomaly * anomaly_rate
        )

        # --- Research Metric 2: Latency ---
        latencies = [s.metrics.get("decision_latency_s", 0.0) for s in step_metrics]
        mean_latency = sum(latencies) / n
        max_latency = max(latencies)

        # --- Research Metric 3: Robustness (within-episode) ---
        # Recovery steps: after each anomaly onset, count steps until
        # no anomaly is active.  Average across anomaly events.
        recovery_steps_list: List[int] = []
        in_anomaly = False
        anomaly_start = 0
        for s in step_metrics:
            if s.metrics.get("anomaly", 0.0) > 0 and not in_anomaly:
                in_anomaly = True
                anomaly_start = s.timestep
            elif s.metrics.get("anomaly", 0.0) == 0 and in_anomaly:
                in_anomaly = False
                recovery_steps_list.append(s.timestep - anomaly_start)
        mean_recovery_steps = (
            sum(recovery_steps_list) / len(recovery_steps_list)
            if recovery_steps_list else 0.0
        )

        # --- Research Metric 4: Resource Efficiency ---
        total_energy = sum(s.metrics.get("energy_consumed_wh", 0.0) for s in step_metrics)
        resource_efficiency = utility / total_energy if total_energy > 0 else 0.0

        # --- Research Metric 5: Operator Load ---
        safety_overrides = sum(
            1 for s in step_metrics if s.metrics.get("safety_override", 0.0) > 0
        )
        operator_load = safety_overrides / n  # fraction of steps with overrides

        # --- Research Metric 8: Explainability ---
        decisions_with_rationale = sum(
            1 for s in step_metrics if s.metrics.get("has_rationale", 0.0) > 0
        )
        explainability_score = decisions_with_rationale / n

        # --- Pipeline Efficiency (C4) ---
        max_dl_mb = last.metrics.get("max_achievable_downlink_mb", 0.0)
        data_dl_efficiency = dl_mb / max_dl_mb if max_dl_mb > 0 else 0.0
        total_detections = last.metrics.get("total_detections", 0)

        aggregated = {
            # Research metrics
            "utility": utility,
            "data_downlink_efficiency": data_dl_efficiency,
            "mean_latency_s": mean_latency,
            "max_latency_s": max_latency,
            "robustness_mean_recovery_steps": mean_recovery_steps,
            "resource_efficiency": resource_efficiency,
            "operator_load": operator_load,
            "explainability_score": explainability_score,
            # Scenario telemetry
            "observation_hours": obs_hours,
            "downlinked_mb": dl_mb,
            "final_battery_soc": final_soc,
            "total_energy_consumed_wh": total_energy,
            "safety_overrides": float(safety_overrides),
            "anomaly_events": float(anomaly_count),
            "total_detections": float(total_detections),
            "max_achievable_downlink_mb": max_dl_mb,
        }

        return EpisodeMetrics(
            num_steps=n,
            total_wall_clock_seconds=total_wall,
            total_reward=total_reward,
            aggregated=aggregated,
            step_metrics=step_metrics,
        )

    # ------------------------------------------------------------------
    # Cross-episode statistics
    # ------------------------------------------------------------------

    def compute_statistics(
        self, episode_metrics: List[EpisodeMetrics]
    ) -> ExperimentStatistics:
        if not episode_metrics:
            return ExperimentStatistics()

        n = len(episode_metrics)

        # Collect all aggregated keys present in any episode
        all_keys: set[str] = set()
        for em in episode_metrics:
            all_keys.update(em.aggregated.keys())
        # Always include total_reward
        all_keys.add("total_reward")

        mean: Dict[str, float] = {}
        std: Dict[str, float] = {}
        min_val: Dict[str, float] = {}
        max_val: Dict[str, float] = {}

        for key in sorted(all_keys):
            if key == "total_reward":
                vals = [em.total_reward for em in episode_metrics]
            else:
                vals = [em.aggregated.get(key, 0.0) for em in episode_metrics]
            m = sum(vals) / n
            var = sum((v - m) ** 2 for v in vals) / max(1, n - 1)
            mean[key] = m
            std[key] = var ** 0.5
            min_val[key] = min(vals)
            max_val[key] = max(vals)

        # Robustness CV (cross-episode): std(utility) / mean(utility)
        u_mean = mean.get("utility", 0.0)
        u_std = std.get("utility", 0.0)
        mean["robustness_cv"] = u_std / u_mean if u_mean > 0 else 0.0

        return ExperimentStatistics(
            num_episodes=n,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            raw_episodes=episode_metrics,
        )

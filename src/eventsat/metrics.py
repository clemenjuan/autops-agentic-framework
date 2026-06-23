"""
EventSat-specific Metrics Collector.

Implements the research metrics for the EventSat scenario.

Research metrics (per episode):
  utility (M-01)     — mission objective achievement, on DELIVERED information.
                       = w_obs x (obs_h / scaled_obs_target)       [w_obs=0.0 default]
                       + w_dl  x (dl_mb / scaled_dl_target)        [w_dl =1.0 default]
                       - w_anomaly x anomaly_rate                  [w_anomaly=0.1]
                       Only data actually downlinked to the ground counts. Raw
                       observation hours are NOT rewarded by default: observation
                       that never downlinks has no mission value, and crediting it
                       lets a planner inflate utility by hoarding undeliverable data
                       (observed empirically: an LLM ground planner over-observed
                       ~22x at equal downlink and scored 18.8 vs 1.3 under the old
                       obs-weighted formula). w_obs is retained as an ablation knob.
                       Targets scaled from 90-day mission to episode length.
  data_downlink_efficiency — fraction of available downlink capacity actually used
                        = data_downlinked_mb / max_achievable_downlink_mb
                        where max_achievable = total_pass_duration_s x S-band rate
                        Source: Proposal — "useful obs time limited by downlink capacity"
  mean_latency_s      — mean decision latency per step
  robustness_mean_recovery_steps — mean steps to recover after anomaly onset
  resource_efficiency — utility / total_energy_consumed_wh
  operator_load       — fraction of steps with environment safety overrides
  explainability_score — fraction of decision cycles with a rationale string

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

from src.core.metrics_collector import (
    EpisodeMetrics,
    ExperimentStatistics,
    MetricsCollector,
    StepMetrics,
)


PLANNER_DIAGNOSTIC_KEYS = (
    "planner_latency_s",
    "orin_planner_latency_ms",
    "planner_rollouts_per_s",
    "candidate_count",
    "cem_iterations",
    "model_size_mb",
    "peak_memory_mb",
    "probe_validation_error",
    "train_dataset_steps",
    "artifact_loaded",
    "artifact_fallback",
    "policy_loaded",
)


class EventSatMetricsCollector(MetricsCollector):
    """Collects and aggregates EventSat-specific + research metrics."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        # Utility weights (configurable via experiment YAML metrics section)
        # M-01 rewards DELIVERED information only (data downlinked to the ground).
        # Raw observation hours are not credited by default (w_obs=0.0): observation
        # that is never downlinked has no mission value, and rewarding it lets a
        # planner inflate utility by over-observing into undeliverable storage.
        # w_obs is retained as a (default-zero) ablation knob.
        uw = self.config.get("utility_weights", {})
        self._w_obs = uw.get("observation", 0.0)
        self._w_dl = uw.get("downlink", 1.0)
        self._w_anomaly = uw.get("anomaly_penalty", 0.1)

        # Targets from scenario (can be overridden in config)
        targets = self.config.get("utility_targets", {})
        self._obs_target_h = targets.get("observation_hours", 2.0)
        # Two hours of 60 s observations, stored/downlinked compressed: 120 * (9.41 / 5.11) ~= 221 MB.
        self._dl_target_mb = targets.get("downlinked_mb", 221.0)

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
        self._battery_capacity_wh = self.config.get("battery_capacity_wh", 70.0)
        self._manual_command_weight = self.config.get("manual_command_weight", 10.0)

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

        # Pre-transition safety classification (anomaly OR critical battery → "safe"),
        # emitted by the env. Immune to the settling mask that reports
        # effective_mode="charging" during a forced-safe step. Fall back to the
        # executed resolved_mode for traces predating the safety_safe field.
        safety_safe = info.get("safety_safe")
        if safety_safe is None:
            safety_safe = 1.0 if info.get("resolved_mode", "") == "safe" else 0.0
        safety_safe = float(safety_safe)

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
            # M-05 Safety-Override: satellite held in protective safe mode (anomaly OR
            # critical battery), from the env's pre-transition safety_safe — immune to
            # both the anomaly-clear timing skew and the settling mask. DISJOINT from
            # M-13: a forced step is safety_safe (M-05) or a charging clamp (M-13).
            "safety_override": safety_safe,
            # Active-state anomaly: true for the full forced-safe duration (unlike
            # "anomaly", which only marks onset events). Drives M-04 recovery timing.
            "anomaly_active": float(info.get("anomaly_forced_safe", 0.0)),
            "in_safe_mode": safety_safe,
            # M-13 Constraint-Violation: agent requested an operational mode whose
            # preconditions failed, so the env clamped it to charging (forced but NOT
            # safety_safe — the safe path is the M-05 safety override).
            "constraint_violation": float(
                bool(info.get("forced", False)) and safety_safe == 0.0
            ),
            # M-02/M-03 Age-of-Information: per-step fresh delivery to the ground
            # (a downlink step resets AoI; gaps between deliveries grow it).
            "step_downlinked_mb": float(info.get("step_downlinked_mb", 0.0)),
            # raw-equivalent MB (emitted by the env; absent in pre-M-12 traces).
            "total_raw_captured_mb": float(info.get("total_raw_captured_mb", 0.0)),
            "downlink_raw_equivalent_mb": float(info.get("downlink_raw_equivalent_mb", 0.0)),
            # 3-pool data pipeline telemetry
            "jetson_raw_mb": info.get("jetson_raw_mb", 0.0),
            "jetson_compressed_mb": info.get("jetson_compressed_mb", 0.0),
            "obc_data_mb": info.get("obc_data_mb", 0.0),
            "in_transition": info.get("in_transition", 0.0),
            # Energy
            "energy_consumed_wh": energy_consumed_wh,
            # Decision loop metrics
            "decision_latency_s": decision_metrics.get("decision_latency_s", 0.0),
            # Whether the primary core actually ran inference this step (ground
            # paradigms decide at passes, not every step) — denominator for M-07.
            "inference_allowed": float(decision_metrics.get("inference_allowed", True)),
            # AH dual-core: wall-clock of the ground-planner decision at this step.
            "ground_decision_latency_s": decision_metrics.get("ground_decision_latency_s", 0.0),
            "has_rationale": float(decision_metrics.get("has_rationale", False)),
        }

        for key in PLANNER_DIAGNOSTIC_KEYS:
            value = decision_metrics.get(key)
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

        return StepMetrics(
            timestep=timestep,
            wall_clock_seconds=wall_clock_seconds,
            reward=reward,
            metrics=metrics,
            metadata={
                "requested_mode": info.get("requested_mode", ""),
                "resolved_mode": info.get("resolved_mode", ""),
            },
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

        # --- Research Metrics 2-3: Age of Information ---
        current_age_s = 0.0
        aoi_integral_s = 0.0
        peak_aoi_s = 0.0
        for s in step_metrics:
            if s.metrics.get("step_downlinked_mb", 0.0) > 0:
                current_age_s = 0.0
            else:
                current_age_s += self._step_duration_s
            aoi_integral_s += current_age_s
            peak_aoi_s = max(peak_aoi_s, current_age_s)
        mean_aoi_s = aoi_integral_s / n


        # --- Research Metric 7: Decision Latency (M-07 = mean wall-clock per *decision cycle*) ---
        # Average over steps where the primary core actually ran inference, not over
        # all steps: ground paradigms decide at passes (~once/orbit), so dividing by
        # every step would amortise a multi-second LLM call down to ~ms. AO/AH onboard
        # decide every step → denominator is all steps (unchanged).
        latencies = [s.metrics.get("decision_latency_s", 0.0) for s in step_metrics]
        decided = [
            l for s, l in zip(step_metrics, latencies)
            if s.metrics.get("inference_allowed", 1.0) > 0
        ]
        mean_latency = (sum(decided) / len(decided)) if decided else 0.0
        max_latency = max(latencies) if latencies else 0.0
        # AH dual-core: ground-planner decision latency, per ground-planning event.
        gp_latencies = [
            s.metrics.get("ground_decision_latency_s", 0.0) for s in step_metrics
        ]
        gp_events = [x for x in gp_latencies if x > 0]
        mean_ground_latency = (sum(gp_events) / len(gp_events)) if gp_events else 0.0

        # --- Research Metric 4: Autonomous Recovery Efficiency ---
        # Count from anomaly onset until the anomaly is cleared and the spacecraft
        # has left safe mode. Incomplete recoveries are horizon-censored.
        recovery_steps_list: List[int] = []
        recovery_start: int | None = None
        for s in step_metrics:
            if s.metrics.get("anomaly", 0.0) > 0 and recovery_start is None:
                recovery_start = s.timestep
            recovered = (
                recovery_start is not None
                and s.timestep > recovery_start
                and s.metrics.get("anomaly_active", 0.0) == 0
                and s.metrics.get("in_safe_mode", 0.0) == 0
            )
            if recovered:
                recovery_steps_list.append(s.timestep - recovery_start)
                recovery_start = None
        unrecovered_anomaly_events = 0.0
        if recovery_start is not None:
            recovery_steps_list.append(step_metrics[-1].timestep + 1 - recovery_start)
            unrecovered_anomaly_events = 1.0
        mean_recovery_steps = (
            sum(recovery_steps_list) / len(recovery_steps_list)
            if recovery_steps_list else 0.0
        )

        # --- Research Metric 6: Resource Efficiency ---
        total_energy = sum(s.metrics.get("energy_consumed_wh", 0.0) for s in step_metrics)
        resource_efficiency = utility / total_energy if total_energy > 0 else 0.0

        # --- Research Metric 5: Safety-Override Rate ---
        # Fraction of steps held in protective safe mode — anomaly or critical
        # battery (disjoint from the M-13 agent constraint-violation rate).
        safety_overrides = sum(
            1 for s in step_metrics if s.metrics.get("safety_override", 0.0) > 0
        )
        operator_load = safety_overrides / n

        # --- Research Metric 8: Explainability ---
        # Coverage is over decision cycles, not episode steps. Ground paradigms only
        # decide at contact opportunities; schedule playback between contacts should
        # not dilute an otherwise explained ground decision down to ~1%.
        decision_cycle_count = sum(
            1 for s in step_metrics if s.metrics.get("inference_allowed", 1.0) > 0
        )
        decisions_with_rationale = sum(
            1
            for s in step_metrics
            if s.metrics.get("inference_allowed", 1.0) > 0
            and s.metrics.get("has_rationale", 0.0) > 0
        )
        explainability_score = (
            decisions_with_rationale / decision_cycle_count
            if decision_cycle_count > 0 else 0.0
        )

        # --- Pipeline Efficiency (C4) ---
        max_dl_mb = last.metrics.get("max_achievable_downlink_mb", 0.0)
        data_dl_efficiency = dl_mb / max_dl_mb if max_dl_mb > 0 else 0.0
        total_detections = last.metrics.get("total_detections", 0)

        # --- Research Metric 12: Value of Information ---
        raw_captured_mb = last.metrics.get("total_raw_captured_mb", 0.0)
        raw_delivered_mb = last.metrics.get("downlink_raw_equivalent_mb", 0.0)
        value_of_information = (
            raw_delivered_mb / raw_captured_mb if raw_captured_mb > 0 else 0.0
        )

        # --- Research Metric 13: Constraint-Violation Rate ---
        constraint_violations = sum(
            1 for s in step_metrics if s.metrics.get("constraint_violation", 0.0) > 0
        )
        constraint_violation_rate = constraint_violations / n

        # --- Research Metric 14: Commanding Effort ---
        command_count = 0
        previous_command = None
        for s in step_metrics:
            command = s.metadata.get("requested_mode", "")
            if command and command != previous_command:
                command_count += 1
                previous_command = command
        episode_days = (n * self._step_duration_s) / 86400.0
        # N_manual = manual ground interventions, counted as anomaly-recovery EVENTS
        # (one ground command per anomaly onset), not per-step safe-mode dwell. Using
        # the forced-step count here let safe-mode dwell dominate M-14 by ~230x.
        manual_intervention_count = float(anomaly_count)
        commanding_effort = (
            (command_count + self._manual_command_weight * manual_intervention_count) / episode_days
            if episode_days > 0 else 0.0
        )

        planner_diagnostics: Dict[str, float] = {}
        for key in PLANNER_DIAGNOSTIC_KEYS:
            values = [
                s.metrics[key]
                for s in step_metrics
                if key in s.metrics and isinstance(s.metrics[key], (int, float))
            ]
            if values:
                planner_diagnostics[key] = sum(values) / len(values)

        aggregated = {
            # Research metrics
            "utility": utility,
            "mean_aoi_s": mean_aoi_s,
            "peak_aoi_s": peak_aoi_s,
            "value_of_information": value_of_information,
            "constraint_violation_rate": constraint_violation_rate,
            "commanding_effort": commanding_effort,
            "data_downlink_efficiency": data_dl_efficiency,
            "mean_latency_s": mean_latency,
            "max_latency_s": max_latency,
            "mean_ground_latency_s": mean_ground_latency,
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
            "unrecovered_anomaly_events": unrecovered_anomaly_events,
            "constraint_violations": float(constraint_violations),
            "constraint_violations_per_episode": float(constraint_violations),
            "command_count": float(command_count),
            "manual_intervention_count": manual_intervention_count,
            "total_raw_captured_mb": raw_captured_mb,
            "downlink_raw_equivalent_mb": raw_delivered_mb,
            "total_detections": float(total_detections),
            "max_achievable_downlink_mb": max_dl_mb,
        }
        aggregated.update(planner_diagnostics)

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

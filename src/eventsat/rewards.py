"""
Reward Functions for Satellite Operations Environments.

Adapted from autops-rl reward modelling (Juan Oliver et al., EUCASS 2025).
Decomposes reward into three components:
  R_total = alpha * [R_resource + R_action + R_mission]

Diagnostic RL runs may add policy-invariant potential shaping inside the same
scale: alpha * [R_resource + R_action + R_mission + k*(gamma*Phi(s') - Phi(s))].

Current implementation: delivery-aligned Individual Negative (Case 2).
Resource and failed-action penalties provide shaping, while successful
pipeline actions are neutral by default and mission progress is measured from
data delivered to the ground.  This avoids teaching a policy the benchmark's
hand-written operations sequence.

Future multi-satellite (Vyoma 12-sat): extend to Collective Negative
(Case 4) where R_mission uses constellation-wide metrics.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict


class EventSatRewardFunction:
    """Individual Negative reward function for EventSat.

    Components (following autops-rl Eq. 3-5, 7):
      R_resource: proportional penalty for low battery and high storage usage
      R_action:   outcome penalty; successful pipeline stages are neutral by default
      R_mission:  penalty proportional to the unmet delivered-data target
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        *,
        discount_factor: float = 1.0,
    ) -> None:
        cfg = config or {}
        self.reward_scale = cfg.get("reward_scale", 0.01)

        self.discount_factor = float(discount_factor)
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be between 0 and 1")

        # Resource penalty parameters (Eq. 3)
        self.resource_penalty_factor = cfg.get("resource_penalty_factor", 1.0)
        self.battery_low_threshold = cfg.get("battery_low_threshold", 0.3)
        self.storage_high_threshold = cfg.get("storage_high_threshold", 0.8)

        # Action reward parameters (Eq. 4)
        self.standby_penalty = cfg.get("standby_penalty", 0.0)
        self.safe_penalty = cfg.get("safe_penalty", 0.3)
        # Retain the historical knobs for explicit ablations, but do not reward
        # the handcrafted observe→compress→send sequence in benchmark defaults.
        self.observe_reward = cfg.get("observe_reward", 0.0)
        self.compress_reward = cfg.get("compress_reward", 0.0)
        self.failed_action_penalty = cfg.get("failed_action_penalty", 0.1)
        self.comm_reward_factor = cfg.get("comm_reward_factor", 1.0)
        self.comm_reward_cap = cfg.get("comm_reward_cap", 5.0)
        # Diagnostic ablation: an in-contact radio attempt with an empty OBC is
        # neutral when False. Out-of-contact attempts always remain failures.
        self.empty_downlink_is_failure = bool(
            cfg.get("empty_downlink_is_failure", True)
        )

        # Optional potential-based shaping for RL learnability experiments.
        # Stage weights are fixed by the three transitions from compressed
        # data to delivery (1/3, 2/3, 1), rather than tuned reward fractions.
        pipeline_shaping = cfg.get("pipeline_shaping", {})
        if not isinstance(pipeline_shaping, dict):
            raise ValueError("pipeline_shaping must be a mapping")
        self.pipeline_shaping_enabled = bool(
            pipeline_shaping.get("enabled", False)
        )
        self.pipeline_potential_type = pipeline_shaping.get("potential", "delivery")
        if self.pipeline_potential_type not in {"delivery", "raw_progress"}:
            raise ValueError("pipeline_shaping.potential must be delivery or raw_progress")
        # k scales the potential itself (Phi' = k*Phi), so shaping stays
        # potential-based (Ng et al., 1999) while matching penalty magnitudes.
        self.pipeline_shaping_scale = float(pipeline_shaping.get("scale", 1.0))
        if not math.isfinite(self.pipeline_shaping_scale) or self.pipeline_shaping_scale < 0.0:
            raise ValueError("pipeline_shaping.scale must be finite and non-negative")

        # Mission penalty parameters (Eq. 7 -- Individual Negative)
        self.mission_scale = cfg.get("mission_scale", 1.0)
        # Raw observations that never reach the ground are not mission utility.
        # Observation credit remains an explicit ablation knob.
        mission_weights = cfg.get("mission_weights", {})
        self.mission_observation_weight = float(
            cfg.get(
                "mission_observation_weight",
                mission_weights.get("observation", 0.0),
            )
        )
        self.mission_downlink_weight = float(
            cfg.get(
                "mission_downlink_weight",
                mission_weights.get("downlink", 1.0),
            )
        )
        if self.mission_observation_weight < 0 or self.mission_downlink_weight < 0:
            raise ValueError("mission reward weights must be non-negative")
        self._mission_weight_sum = (
            self.mission_observation_weight + self.mission_downlink_weight
        )

    def resource_penalty(
        self, battery_soc: float, data_stored_mb: float, storage_capacity_mb: float
    ) -> float:
        """Penalize low battery and high storage usage (Eq. 3).

        Penalty is proportional to how far resources are from ideal:
        - Battery: penalty when SoC drops below threshold
        - Storage: penalty when usage exceeds threshold
        """
        penalty = 0.0
        if battery_soc < self.battery_low_threshold:
            penalty += (self.battery_low_threshold - battery_soc) / self.battery_low_threshold
        storage_ratio = data_stored_mb / storage_capacity_mb if storage_capacity_mb > 0 else 0.0
        if storage_ratio > self.storage_high_threshold:
            penalty += (storage_ratio - self.storage_high_threshold) / (
                1.0 - self.storage_high_threshold
            )
        return -self.resource_penalty_factor * penalty

    def action_reward(self, mode: str, action_info: Dict[str, Any]) -> float:
        """Reward or penalize based on action taken and its outcome (Eq. 4).

        Args:
            mode: The resolved mode that was executed.
            action_info: Dict with outcome details:
                - constraint_violation: requested mode was operationally invalid
                  and the environment clamped it to a non-safety fallback
                - storage_overflow: bool, observation caused storage cap
                - had_data_to_compress: bool
                - data_sent_mb: float, MB transferred from Jetson to OBC
                - pass_active: bool, ground pass available for comm
                - data_downlinked_mb: float, MB actually downlinked this step
                - communication_failure: optional reason for zero downlink
        """
        if self.is_failed_action(mode, action_info):
            return -self.failed_action_penalty

        if mode == "payload_observe":
            return self.observe_reward

        if mode in {"payload_compress", "payload_detect"}:
            return self.compress_reward

        if mode == "payload_send":
            return self.compress_reward * 0.5  # Moving data, less value than processing

        if mode == "communication":
            dl = action_info.get("data_downlinked_mb", 0.0)
            return min(self.comm_reward_factor * dl, self.comm_reward_cap)

        if mode == "charging":
            return -self.standby_penalty

        if mode == "safe":
            return -self.safe_penalty

        if mode == "transitioning":
            return -self.standby_penalty

        return 0.0

    def is_failed_action(self, mode: str, action_info: Dict[str, Any]) -> bool:
        """Shared failure classification for the reward and diagnostic logging."""
        if action_info.get("constraint_violation", False):
            return True
        if mode == "payload_observe":
            return bool(action_info.get("storage_overflow", False))
        if mode == "payload_compress":
            return not action_info.get("had_data_to_compress", False)
        if mode == "payload_detect":
            return not action_info.get("had_data_to_detect", False)
        if mode == "payload_send":
            return not action_info.get("had_data_to_send", False)
        if mode == "communication":
            if not action_info.get("pass_active", False):
                return True
            if action_info.get("data_downlinked_mb", 0.0) > 0.0:
                return False
            if action_info.get("communication_failure") == "no_contact":
                return True
            return self.empty_downlink_is_failure
        return False

    def pipeline_potential(
        self,
        state: Mapping[str, Any],
        *,
        compression_ratio: float,
        downlink_target_mb: float,
    ) -> float:
        """Return normalized progress through the physical data pipeline.

        Ng, Harada & Russell (1999): use this state potential only through
        gamma * Phi(next) - Phi(previous). The legacy delivery potential gives
        compressed/OBC/ground credits of 1/3, 2/3, 1. raw_progress gives
        raw/compressed/OBC/ground credits of 1/4, 1/2, 3/4, 1 and interpolates
        compression progress for ONE raw product. All stages share a single
        raw-equivalent mission-target cap, with furthest-stage priority.
        """
        if not self.pipeline_shaping_enabled:
            return 0.0

        ratio = float(compression_ratio)
        if ratio <= 0.0:
            raise ValueError("compression_ratio must be positive")

        target_raw_mb = max(0.0, float(downlink_target_mb)) * ratio
        if target_raw_mb == 0.0:
            return 0.0

        compressed_raw_mb = max(
            0.0, float(state.get("jetson_compressed_mb", 0.0))
        ) * ratio
        obc_raw_mb = max(0.0, float(state.get("obc_raw_equivalent_mb", 0.0)))
        ground_raw_mb = max(
            0.0, float(state.get("downlink_raw_equivalent_mb", 0.0))
        )

        # Credit at most one target's worth of data across all stages. Assign
        # eligibility from the furthest stage backwards to avoid rewarding a
        # backlog once the corresponding mission target is already delivered.
        remaining_raw_mb = target_raw_mb
        credited_ground_mb = min(ground_raw_mb, remaining_raw_mb)
        remaining_raw_mb -= credited_ground_mb
        credited_obc_mb = min(obc_raw_mb, remaining_raw_mb)
        remaining_raw_mb -= credited_obc_mb
        credited_compressed_mb = min(compressed_raw_mb, remaining_raw_mb)
        remaining_raw_mb -= credited_compressed_mb

        if self.pipeline_potential_type == "raw_progress":
            raw_mb = max(0.0, float(state.get("jetson_raw_mb", 0.0)))
            credited_raw_mb = min(raw_mb, remaining_raw_mb)
            # Progress belongs to one product, never the entire raw backlog.
            product_mb = max(0.0, float(state.get("observation_size_mb", 0.0)))
            progress = min(1.0, max(
                0.0, float(state.get("compression_progress_fraction", 0.0))
            ))
            processing_mb = (
                min(product_mb, credited_raw_mb)
                if float(state.get("uncompressed_observations", 0.0)) >= 1.0
                and raw_mb >= product_mb > 0.0
                else 0.0
            )
            return min(1.0, max(0.0, (
                credited_ground_mb
                + 0.75 * credited_obc_mb
                + 0.5 * credited_compressed_mb
                + 0.25 * credited_raw_mb
                + 0.25 * progress * processing_mb
            ) / target_raw_mb))

        weighted_progress_mb = (
            credited_ground_mb
            + (2.0 / 3.0) * credited_obc_mb
            + (1.0 / 3.0) * credited_compressed_mb
        )
        return min(1.0, max(0.0, weighted_progress_mb / target_raw_mb))

    def pipeline_shaping_reward(
        self,
        previous_state: Mapping[str, Any],
        next_state: Mapping[str, Any],
        *,
        compression_ratio: float,
        downlink_target_mb: float,
        is_final_step: bool,
    ) -> float:
        """Compute policy-invariant shaping ``k * (gamma * Phi(s') - Phi(s))``."""
        if not self.pipeline_shaping_enabled:
            return 0.0

        previous_potential = self.pipeline_potential(
            previous_state,
            compression_ratio=compression_ratio,
            downlink_target_mb=downlink_target_mb,
        )
        next_potential = (
            0.0
            if is_final_step
            else self.pipeline_potential(
                next_state,
                compression_ratio=compression_ratio,
                downlink_target_mb=downlink_target_mb,
            )
        )
        return self.pipeline_shaping_scale * (
            self.discount_factor * next_potential - previous_potential
        )

    def mission_penalty(
        self,
        is_final_step: bool,
        obs_hours: float,
        downlinked_mb: float,
        obs_target_hours: float,
        downlink_target_mb: float,
        episode_steps: int,
        max_mission_steps: int,
    ) -> float:
        """Individual Negative mission term (Eq. 7).

        Applied every step as a continuous signal: penalizes the weighted
        fraction of configured targets not yet achieved, scaled by episode
        progress. Benchmark defaults put all weight on ground-delivered data;
        raw observation hours are available only as an explicit ablation.

        For future Vyoma 12-sat: override this method in a CollectiveNegative
        subclass to use constellation-wide metrics.
        """
        # Fraction of targets not yet met (both clamped to [0, 1])
        obs_gap = max(0.0, 1.0 - obs_hours / obs_target_hours) if obs_target_hours > 0 else 0.0
        dl_gap = (
            max(0.0, 1.0 - downlinked_mb / downlink_target_mb) if downlink_target_mb > 0 else 0.0
        )

        unmet_fraction = (
            (
                self.mission_observation_weight * obs_gap
                + self.mission_downlink_weight * dl_gap
            )
            / self._mission_weight_sum
            if self._mission_weight_sum > 0.0
            else 0.0
        )

        # Scale penalty by episode progress (larger penalty as time runs out)
        progress = episode_steps / max_mission_steps if max_mission_steps > 0 else 1.0

        # At final step, apply full penalty; during episode, scale by progress
        if is_final_step:
            return -self.mission_scale * unmet_fraction
        return -self.mission_scale * unmet_fraction * progress

    def compute(
        self,
        mode: str,
        battery_soc: float,
        data_stored_mb: float,
        storage_capacity_mb: float,
        action_info: Dict[str, Any],
        obs_hours: float,
        downlinked_mb: float,
        obs_target_hours: float,
        downlink_target_mb: float,
        episode_step: int,
        max_steps: int,
        is_final_step: bool,
        pipeline_state_before: Mapping[str, Any] | None = None,
        pipeline_state_after: Mapping[str, Any] | None = None,
        compression_ratio: float = 1.0,
    ) -> float:
        """Compute task reward plus optional potential-based shaping."""
        r_resource = self.resource_penalty(battery_soc, data_stored_mb, storage_capacity_mb)
        r_action = self.action_reward(mode, action_info)
        r_mission = self.mission_penalty(
            is_final_step=is_final_step,
            obs_hours=obs_hours,
            downlinked_mb=downlinked_mb,
            obs_target_hours=obs_target_hours,
            downlink_target_mb=downlink_target_mb,
            episode_steps=episode_step,
            max_mission_steps=max_steps,
        )
        r_shaping = 0.0
        if self.pipeline_shaping_enabled:
            if pipeline_state_before is None or pipeline_state_after is None:
                raise ValueError(
                    "pipeline shaping requires both previous and next pipeline state"
                )
            r_shaping = self.pipeline_shaping_reward(
                pipeline_state_before,
                pipeline_state_after,
                compression_ratio=compression_ratio,
                downlink_target_mb=downlink_target_mb,
                is_final_step=is_final_step,
            )
        return self.reward_scale * (
            r_resource + r_action + r_mission + r_shaping
        )


class MultiEventsatRewardFunction:
    """Per-satellite reward for the ``multieventsat`` constellation scenario.

    Single locus of reward freedom for multi-agent scenarios. Its contract
    returns a dict keyed by satellite_id, so a future scenario-specific reward
    class is a drop-in replacement that can compute a collective/shared term
    internally with no changes to the environment or the RLlib bridge.

    v1: individual per-satellite rewards are produced upstream by each
    satellite's sub-environment (:class:`EventSatRewardFunction`, Individual
    Negative, Case 2); :meth:`compute_rewards` scales them and optionally adds a
    team term: ``local_weight * r_i + team_weight * team(r)``.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.local_weight = float(cfg.get("local_weight", 1.0))
        self.team_weight = float(cfg.get("team_weight", 0.0))
        self.team_reducer = str(cfg.get("team_reducer", "mean"))  # mean | sum | min

    def team_term(
        self,
        individual_rewards: Dict[str, float],
        per_satellite_inputs: Dict[str, Dict[str, Any]],
    ) -> float:
        """Collective reward term shared by all satellites (override for Case 4)."""
        values = list(individual_rewards.values())
        if not values:
            return 0.0
        if self.team_reducer == "sum":
            return float(sum(values))
        if self.team_reducer == "min":
            return float(min(values))
        return float(sum(values) / len(values))  # mean

    def compute_rewards(
        self,
        individual_rewards: Dict[str, float],
        per_satellite_inputs: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, float]:
        """Final per-satellite rewards from precomputed individual rewards.

        Each satellite's individual reward (already computed upstream by its
        sub-environment via :class:`EventSatRewardFunction`) is scaled by
        ``local_weight`` and, when ``team_weight > 0``, combined with a shared
        team term: ``local_weight * r_i + team_weight * team(r)``. Returns a dict
        keyed by satellite_id.

        ``per_satellite_inputs`` is accepted and forwarded to :meth:`team_term`
        so an override (Case 4) can build a richer collective term from raw
        per-satellite state; the default mean/sum/min term ignores it.
        """
        if self.team_weight == 0.0 or not individual_rewards:
            return {
                sat_id: self.local_weight * r
                for sat_id, r in individual_rewards.items()
            }
        team = self.team_term(individual_rewards, per_satellite_inputs or {})
        return {
            sat_id: self.local_weight * r + self.team_weight * team
            for sat_id, r in individual_rewards.items()
        }

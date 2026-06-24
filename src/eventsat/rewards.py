"""
Reward Functions for Satellite Operations Environments.

Adapted from autops-rl reward modelling (Juan Oliver et al., EUCASS 2025).
Decomposes reward into three components:
  R_total = alpha * [R_resource + R_action + R_mission]

Current implementation: Individual Negative (Case 2) -- penalizes
unmet targets rather than rewarding achieved ones. Paper findings
show negative reward functions produce more stable and robust policies
across scaling factors and coordination topologies.

Future multi-satellite (Vyoma 12-sat): extend to Collective Negative
(Case 4) where R_mission uses constellation-wide metrics.
"""
from __future__ import annotations

from typing import Any, Dict


class EventSatRewardFunction:
    """Individual Negative reward function for EventSat.

    Components (following autops-rl Eq. 3-5, 7):
      R_resource: proportional penalty for low battery and high storage usage
      R_action:   reward/penalty per mode based on outcome
      R_mission:  penalty proportional to unmet observation & downlink targets
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.reward_scale = cfg.get("reward_scale", 0.01)

        # Resource penalty parameters (Eq. 3)
        self.resource_penalty_factor = cfg.get("resource_penalty_factor", 1.0)
        self.battery_low_threshold = cfg.get("battery_low_threshold", 0.3)
        self.storage_high_threshold = cfg.get("storage_high_threshold", 0.8)

        # Action reward parameters (Eq. 4)
        self.standby_penalty = cfg.get("standby_penalty", 0.05)
        self.safe_penalty = cfg.get("safe_penalty", 0.3)
        self.observe_reward = cfg.get("observe_reward", 1.0)
        self.compress_reward = cfg.get("compress_reward", 0.5)
        self.failed_action_penalty = cfg.get("failed_action_penalty", 0.1)
        self.comm_reward_factor = cfg.get("comm_reward_factor", 1.0)
        self.comm_reward_cap = cfg.get("comm_reward_cap", 5.0)

        # Mission penalty parameters (Eq. 7 -- Individual Negative)
        self.mission_scale = cfg.get("mission_scale", 1.0)

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
                - storage_overflow: bool, observation caused storage cap
                - had_data_to_compress: bool
                - pass_active: bool, ground pass available for comm
                - data_downlinked_mb: float, MB actually downlinked this step
        """
        if mode == "payload_observe":
            r = self.observe_reward
            if action_info.get("storage_overflow", False):
                r -= self.observe_reward * 0.5
            return r

        if mode == "payload_compress":
            if action_info.get("had_data_to_compress", False):
                return self.compress_reward
            return -self.failed_action_penalty

        if mode == "payload_detect":
            if action_info.get("had_data_to_detect", False):
                return self.compress_reward  # Same as compress — productive Jetson work
            return -self.failed_action_penalty

        if mode == "payload_send":
            if action_info.get("had_data_to_send", False):
                return self.compress_reward * 0.5  # Moving data, less value than processing
            return -self.failed_action_penalty

        if mode == "communication":
            if action_info.get("pass_active", False):
                dl = action_info.get("data_downlinked_mb", 0.0)
                return min(self.comm_reward_factor * dl, self.comm_reward_cap)
            return -self.failed_action_penalty

        if mode == "charging":
            return -self.standby_penalty

        if mode == "safe":
            return -self.safe_penalty

        if mode == "transitioning":
            return -self.standby_penalty

        return 0.0

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

        Applied every step as a continuous signal: penalizes the fraction
        of each target NOT yet achieved, scaled by episode progress.

        For future Vyoma 12-sat: override this method in a CollectiveNegative
        subclass to use constellation-wide metrics.
        """
        # Fraction of targets not yet met (both clamped to [0, 1])
        obs_gap = max(0.0, 1.0 - obs_hours / obs_target_hours) if obs_target_hours > 0 else 0.0
        dl_gap = (
            max(0.0, 1.0 - downlinked_mb / downlink_target_mb) if downlink_target_mb > 0 else 0.0
        )

        # Weight: observation and downlink equally important
        unmet_fraction = 0.5 * obs_gap + 0.5 * dl_gap

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
    ) -> float:
        """Compute total reward: R = alpha * [R_resource + R_action + R_mission]."""
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
        return self.reward_scale * (r_resource + r_action + r_mission)


class BaseMultiSatRewardFunction:
    """Per-satellite reward for the ``basemultisat`` constellation scenario.

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

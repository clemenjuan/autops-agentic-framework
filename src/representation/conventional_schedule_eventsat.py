"""
Conventional Schedule Representation for EventSat Human Ground Operations.

Models a human flight dynamics team planning a time-tagged command schedule.
Subclasses ScheduleBasedEventSat and overrides _generate_schedule() to add
cognitive constraints that degrade schedule quality relative to the optimal
algorithmic planner.

Cognitive constraint parameters (Sellmaier et al. 2022 [SGJTLF4D],
ECSS-E-ST-70C [CIYT2V68], Endsley 1995 [46MUS93H]):

    conservative_margin (float, default 1.3):
        Multiplier on charge duration estimates. Humans use worst-case power
        models from their ops handbook, adding 20-40% margin to charge blocks.
        A value of 1.3 means "charge for 1.3x the computed time."

    planning_horizon_discount (float, default 0.85):
        Fraction of the inter-pass gap actually planned. Human teams cannot
        reliably plan the full gap — cognitive load and shift handover limits
        cause them to leave ~10-20% of the gap as a charging buffer at the end.
        active_steps = int(gap_steps * planning_horizon_discount)

    max_observations_per_gap (int, default 2):
        Maximum observation blocks per schedule. Each observation creates a
        cascade of compression → detection → send steps. Human planners limit
        observations to avoid complex multi-step reasoning. Real LEO missions
        typically schedule 1-3 observations per gap.

    shift_handover_probability (float, default 0.10):
        Probability per schedule-generation call that a shift handover occurred
        since the last pass. On handover, the incoming team has degraded context
        (Endsley 1995 SA Level-3 projection loss) and applies a higher min_soc
        threshold as a safety margin.

    shift_handover_soc_penalty (float, default 0.10):
        Additional SoC added to min_soc_for_operations when a shift handover
        is detected. E.g., normal min_soc=0.40 becomes 0.50 during handovers.

Key context: The schedule generated here will be uploaded one pass later
(per ConventionalGround paradigm). So it plans based on telemetry that will
be ~1 orbit stale by execution time. The conservative parameters account for
this temporal uncertainty.

Registered as "conventional_schedule_eventsat" in the emergence controller.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from src.emergence.controller import register
from src.representation.schedule_based_eventsat import ScheduleBasedEventSat, _merge_schedule


@register("conventional_schedule_eventsat")
class ConventionalScheduleEventSat(ScheduleBasedEventSat):
    """Human-modeled schedule planner with cognitive constraints.

    Inherits all physics modeling from ScheduleBasedEventSat and overrides
    schedule generation to introduce human planning limitations.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)

        # Human cognitive constraint parameters
        self._conservative_margin: float = self.config.get("conservative_margin", 1.3)
        self._planning_horizon_discount: float = self.config.get(
            "planning_horizon_discount", 0.85
        )
        self._max_observations_per_gap: int = self.config.get("max_observations_per_gap", 2)
        self._shift_handover_probability: float = self.config.get(
            "shift_handover_probability", 0.10
        )
        self._shift_handover_soc_penalty: float = self.config.get(
            "shift_handover_soc_penalty", 0.10
        )

        # Random state for shift handover stochasticity (seeded by experiment runner)
        self._rng = random.Random()

    def _generate_schedule(
        self,
        state: Dict[str, Any],
        gap_steps: int,
        orient_urgency: float = 0.0,
    ) -> List[Tuple[str, int]]:
        """Generate a schedule with human planning cognitive constraints.

        Modifications vs. algorithmic ScheduleBasedEventSat:
          1. planning_horizon_discount: plan only a fraction of the gap
          2. conservative_margin: inflate charge duration estimates
          3. max_observations_per_gap: cap total observation blocks
          4. shift_handover: stochastically raise min_soc_for_operations
        """
        # --- Apply planning horizon discount ---
        effective_gap = max(10, int(gap_steps * self._planning_horizon_discount))
        remaining_as_charging = gap_steps - effective_gap  # appended at the end

        # --- Determine effective min_soc (shift handover penalty) ---
        effective_min_soc = self._min_soc_for_operations
        handover_occurred = self._rng.random() < self._shift_handover_probability
        if handover_occurred:
            effective_min_soc = min(
                effective_min_soc + self._shift_handover_soc_penalty, 0.80
            )

        # --- Run parent schedule generation with modified parameters ---
        # Temporarily patch the relevant attributes on self so the parent
        # _generate_schedule logic uses our degraded values, then restore them.
        original_min_soc = self._min_soc_for_operations
        self._min_soc_for_operations = effective_min_soc

        # Generate schedule using the effective (discounted) gap
        raw_schedule = self._generate_schedule_inner(
            state=state,
            gap_steps=effective_gap,
            orient_urgency=orient_urgency,
            conservative_margin=self._conservative_margin,
            max_observations=self._max_observations_per_gap,
        )

        # Restore original min_soc
        self._min_soc_for_operations = original_min_soc

        # Append the un-planned tail as charging (human buffer)
        if remaining_as_charging > 0:
            raw_schedule.append(("charging", remaining_as_charging))

        return _merge_schedule(raw_schedule)

    def _generate_schedule_inner(
        self,
        state: Dict[str, Any],
        gap_steps: int,
        orient_urgency: float,
        conservative_margin: float,
        max_observations: int,
    ) -> List[Tuple[str, int]]:
        """Core schedule generation with cognitive constraint hooks.

        This replicates the parent logic but adds:
          - conservative_margin applied to charge blocks
          - max_observations cap on observation blocks
        """
        # Simulated state
        sim_soc = state.get("battery_soc", 0.5)
        sim_uncomp = state.get("uncompressed_observations", 0)
        sim_undetected = state.get("undetected_observations", 0)
        sim_jetson_compressed_mb = state.get("jetson_compressed_mb", 0.0)
        sim_jetson_raw_mb = state.get("jetson_raw_mb", 0.0)
        sim_obc_mb = state.get("obc_data_mb", 0.0)
        daily_budget_mb = state.get("daily_downlink_budget_mb", self._daily_downlink_budget_mb)

        # Reserve the last chunk for charging (pre-pass battery buffer)
        # OODA: high urgency → reduce reserve fraction
        reserve_fraction = self._charge_reserve_fraction
        if orient_urgency > 0.6:
            reserve_fraction = max(0.06, reserve_fraction * 0.5)
        reserve_steps = max(5, int(gap_steps * reserve_fraction))
        active_steps = gap_steps - reserve_steps

        schedule: List[Tuple[str, int]] = []
        remaining = active_steps
        obs_count = 0  # Track observations for cap

        # Precompute pipeline constants
        obs_compressed_mb = self._observation_size_mb / self._compression_ratio
        compress_steps = int(self._compression_time_factor) + self._settling_time_steps
        jetson_send_rate_mbs = self._jetson_to_obc_rate_kbps / 8.0 / 1000.0
        obs_schedule_steps = self._settling_time_steps + 1

        while remaining > 0:
            # ---- Battery critically low: charge first ----
            if sim_soc < self._min_soc_for_operations:
                target_soc = min(self._min_soc_for_operations + 0.15, 0.85)
                base_charge_dur = self._steps_to_reach_soc(sim_soc, target_soc)
                # Apply conservative margin: humans over-estimate charge duration
                charge_dur = min(
                    max(1, int(base_charge_dur * conservative_margin)), remaining
                )
                schedule.append(("charging", charge_dur))
                for _ in range(charge_dur):
                    sim_soc = min(1.0, sim_soc + self._soc_delta_per_step("charging"))
                remaining -= charge_dur
                continue

            # ---- Compression backlog ----
            if sim_uncomp >= 1 and sim_soc >= 0.35:
                dur = min(compress_steps, remaining)
                schedule.append(("payload_compress", dur))
                for _ in range(dur):
                    sim_soc = min(1.0, sim_soc + self._soc_delta_per_step("payload_compress"))
                if dur >= compress_steps:
                    sim_uncomp -= 1
                    sim_jetson_raw_mb = max(
                        0.0, sim_jetson_raw_mb - self._observation_size_mb
                    )
                    sim_jetson_compressed_mb += obs_compressed_mb
                    sim_undetected += 1
                remaining -= dur
                continue

            # ---- Detection backlog ----
            if sim_undetected > 0 and sim_soc >= 0.35:
                dur = min(self._detection_steps, remaining)
                schedule.append(("payload_detect", dur))
                for _ in range(dur):
                    sim_soc = min(1.0, sim_soc + self._soc_delta_per_step("payload_detect"))
                if dur >= self._detection_steps:
                    sim_undetected -= 1
                remaining -= dur
                continue

            # ---- Send compressed data to OBC ----
            if sim_jetson_compressed_mb > 0.01 and sim_soc >= 0.35:
                steps_to_drain = max(
                    1,
                    int(sim_jetson_compressed_mb / (jetson_send_rate_mbs * self._step_duration_s)) + 1,
                )
                dur = min(steps_to_drain, remaining)
                schedule.append(("payload_send", dur))
                transferred = jetson_send_rate_mbs * self._step_duration_s * dur
                for _ in range(dur):
                    sim_soc = min(1.0, sim_soc + self._soc_delta_per_step("payload_send"))
                sim_jetson_compressed_mb = max(0.0, sim_jetson_compressed_mb - transferred)
                sim_obc_mb = min(self._obc_capacity_mb, sim_obc_mb + transferred)
                remaining -= dur
                continue

            # ---- Observe (subject to max_observations cap) ----
            pipeline_mb = sim_obc_mb + sim_jetson_compressed_mb
            if (
                obs_count < max_observations
                and sim_soc > 0.60
                and pipeline_mb < daily_budget_mb
                and sim_obc_mb < self._obc_capacity_mb * 0.8
                and remaining >= obs_schedule_steps
            ):
                schedule.append(("payload_observe", obs_schedule_steps))
                for _ in range(self._settling_time_steps):
                    sim_soc = min(1.0, sim_soc + self._soc_delta_per_step("charging"))
                sim_soc = min(1.0, sim_soc + self._soc_delta_per_step("payload_observe"))
                sim_jetson_raw_mb += self._observation_size_mb
                sim_uncomp += 1
                obs_count += 1
                remaining -= obs_schedule_steps
                continue

            # ---- Default: charge ----
            dur = min(10, remaining)
            schedule.append(("charging", dur))
            for _ in range(dur):
                sim_soc = min(1.0, sim_soc + self._soc_delta_per_step("charging"))
            remaining -= dur

        # Reserve block at end: charging (pre-pass battery buffer for comms)
        if reserve_steps > 0:
            schedule.append(("charging", reserve_steps))

        return schedule

    def get_name(self) -> str:
        return "ConventionalScheduleEventSat"

    def seed(self, seed_value: int) -> None:
        """Seed the internal RNG for reproducible shift handover simulation."""
        self._rng.seed(seed_value)

"""
Schedule-Based Representation for EventSat Conventional Ground Operations.

Ground schedule planner that generates time-tagged command schedules during
ground passes for execution between passes (zero onboard autonomy).

Telemetry-first sequencing:
  1. Pass starts → staleness_steps is high (stale from previous pass)
  2. First step(s): staleness high → return communication only (downlink HK first)
  3. After downlink: staleness_steps drops to ~1 → generate schedule from fresh data
  4. Return {"mode": "communication", "schedule": [...]} so paradigm stores it

Registered as "schedule_based_eventsat" in the representation factory.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.core.representation import Representation
from src.core.behaviour.controller import register

if TYPE_CHECKING:
    from src.core.decision_procedure.context import DecisionContext


@register("schedule_based_eventsat")
class ScheduleBasedEventSat(Representation):
    """Ground schedule planner for EventSat conventional operations.

    Generates a full time-tagged schedule during each ground pass based on
    freshly downlinked telemetry. The schedule covers the gap until the
    estimated next pass using a greedy cyclic battery-aware planner.
    """

    # Default power consumption (eclipse rates, from eventsat.yaml PDR Table 6.2)
    _DEFAULT_POWER = {
        "charging":         {"sun_w": 4.72,  "eclipse_w": 4.32},
        "communication":    {"sun_w": 33.65, "eclipse_w": 33.24},
        "payload_observe":  {"sun_w": 17.94, "eclipse_w": 17.55},
        "payload_compress": {"sun_w": 12.77, "eclipse_w": 12.37},
        "payload_detect":   {"sun_w": 12.77, "eclipse_w": 12.37},
        "payload_send":     {"sun_w": 12.77, "eclipse_w": 12.37},
        "safe":             {"sun_w": 9.58,  "eclipse_w": 9.58},
    }

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._last_rationale: Optional[str] = None
        self._schedule_generated_this_pass: bool = False
        self._last_pass_active: bool = False

        # Power system parameters
        self._solar_generation_w: float = self.config.get("solar_generation_w", 24.0)
        self._battery_capacity_wh: float = self.config.get("battery_capacity_wh", 70.0)
        self._eclipse_fraction: float = self.config.get("eclipse_fraction", 0.36)
        self._step_duration_s: float = self.config.get("step_duration_s", 60.0)
        self._charge_efficiency: float = self.config.get("charge_efficiency", 0.9)

        # Payload pipeline parameters
        self._compression_time_factor: float = self.config.get("compression_time_factor", 2.0)
        self._detection_steps: int = self.config.get("detection_steps", 5)
        self._observation_size_mb: float = self.config.get("observation_size_mb", 9.41)
        self._compression_ratio: float = self.config.get("compression_ratio", 5.11)
        self._jetson_to_obc_rate_kbps: float = self.config.get("jetson_to_obc_rate_kbps", 50.0)
        self._daily_downlink_budget_mb: float = self.config.get("daily_downlink_budget_mb", 27.0)
        self._obc_capacity_mb: float = self.config.get("obc_capacity_mb", 4096.0)

        # Planning parameters
        self._charge_reserve_fraction: float = self.config.get("charge_reserve_fraction", 0.12)
        self._min_soc_for_operations: float = self.config.get("min_soc_for_operations", 0.40)
        self._staleness_threshold: int = self.config.get("staleness_threshold", 5)
        # Settling time for attitude-maneuver modes (payload_observe, communication).
        # Must match environment modes.transition_overhead.settling_time_s / step_duration_s.
        # Default: ceil(135s / 60s) = 3 steps (conservative; env uses floor = 2).
        self._settling_time_steps: int = self.config.get("settling_time_steps", 2)

        # Power consumption override from config
        self._power_consumption = dict(self._DEFAULT_POWER)
        cfg_power = self.config.get("power_consumption", {})
        for mode, rates in cfg_power.items():
            self._power_consumption[mode] = rates

    # ------------------------------------------------------------------
    # Representation interface
    # ------------------------------------------------------------------

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        """Extract decision-relevant state from the (stale) observation."""
        if hasattr(observation, "constellation_state"):
            sat = observation.constellation_state.satellites.get("eventsat_0")
            if sat is None:
                return {}
            res = sat.resources or {}
            meta = sat.metadata or {}
            return {
                "battery_soc": res.get("battery_soc", 0.5),
                "current_mode": sat.status,
                "in_sunlight": meta.get("in_sunlight", False),
                "ground_pass_active": meta.get("ground_pass_active", False),
                "data_stored_mb": res.get("data_stored_mb", 0.0),
                "obc_data_mb": res.get("obc_data_mb", meta.get("obc_data_mb", 0.0)),
                "jetson_raw_mb": meta.get("jetson_raw_mb", 0.0),
                "jetson_compressed_mb": meta.get("jetson_compressed_mb", 0.0),
                "storage_capacity_mb": meta.get("storage_capacity_mb", 4096.0),
                "uncompressed_observations": meta.get("uncompressed_observations", 0),
                "undetected_observations": meta.get("undetected_observations", 0),
                "health_status": meta.get("health_status", "nominal"),
                "staleness_steps": meta.get("staleness_steps", 0),
                "estimated_gap_steps": meta.get("estimated_gap_steps", 93),
                "daily_downlink_budget_mb": meta.get(
                    "daily_downlink_budget_mb", self._daily_downlink_budget_mb
                ),
                # Physical downlink achievable at the next pass (50 kbps × contact).
                "achievable_downlink_mb": meta.get("achievable_downlink_mb"),
                "settling_time_steps": self._settling_time_steps,
            }
        return {}

    def select_action(self, context: DecisionContext) -> Dict[str, Any]:
        """Select action based on current state.

        During a ground pass:
          - If telemetry is stale → communicate first (downlink HK packet)
          - If telemetry is fresh → generate schedule and attach to action

        Between passes → return charging (ignored; paradigm plays back schedule).

        """
        from src.core.decision_procedure.context import DecisionContext  # runtime import

        state = context.state
        if not state:
            self._last_rationale = "No state; defaulting to charging."
            return {"eventsat_0": {"mode": "charging"}}

        pass_active = state.get("ground_pass_active", False)
        staleness = state.get("staleness_steps", 999)

        # Detect pass transitions for schedule tracking
        if not pass_active and self._last_pass_active:
            self._schedule_generated_this_pass = False
        self._last_pass_active = pass_active

        if not pass_active:
            # Between passes — schedule is being executed by the paradigm
            self._last_rationale = "Between passes; schedule executing autonomously."
            return {"eventsat_0": {"mode": "charging"}}

        # During a ground pass
        if staleness > self._staleness_threshold:
            # Stale telemetry at a pass implies a NEW contact: under AG/AH the
            # planner is only invoked during passes, so the pass-transition reset
            # above never fires (it needs a between-pass call). Without this
            # reset the planner generated exactly one schedule per episode and
            # idled thereafter (full-length trace, 2026-06-11).
            self._schedule_generated_this_pass = False
            self._last_rationale = (
                f"Pass active but telemetry stale ({staleness} steps); "
                f"communicating to receive fresh HK."
            )
            return {"eventsat_0": {"mode": "communication"}}

        if self._schedule_generated_this_pass:
            # Schedule already uploaded this pass — keep communicating to downlink data
            self._last_rationale = "Schedule uploaded; continuing communication for data downlink."
            return {"eventsat_0": {"mode": "communication"}}

        # Fresh telemetry received — generate and upload schedule
        gap_steps = state.get("estimated_gap_steps", 93)
        schedule = self._generate_schedule(state, gap_steps)
        self._schedule_generated_this_pass = True
        self._last_rationale = (
            f"Fresh telemetry (staleness={staleness}); generated schedule "
            f"of {len(schedule)} entries for {gap_steps}-step gap."
        )
        return {"eventsat_0": {"mode": "communication", "schedule": schedule}}

    def get_rationale(self) -> Optional[str]:
        return self._last_rationale

    def reason(self, state: Dict[str, Any], memory: Any) -> List[Dict[str, Any]]:
        """Summarise schedule planning intent for explanations/debugging."""
        if not state:
            return [{"phase": "schedule_planning", "intent": "no_state_default_charging"}]

        pass_active = state.get("ground_pass_active", False)
        soc = state.get("battery_soc", 0.5)
        gap_steps = state.get("estimated_gap_steps", 93)  # 93 ≈ 5554s / 60s (LEO default)
        uncomp = state.get("uncompressed_observations", 0)
        undetected = state.get("undetected_observations", 0)
        obc_mb = state.get("obc_data_mb", 0.0)

        trace = [{
            "phase": "schedule_planning",
            "pass_active": pass_active,
            "battery_soc": soc,
            "gap_steps": gap_steps,
            "compression_backlog": uncomp,
            "detection_backlog": undetected,
            "obc_data_mb": obc_mb,
        }]

        if soc < self._min_soc_for_operations:
            trace.append({"implication": "charge_first", "reason": f"SoC={soc:.2f} below min_soc={self._min_soc_for_operations}"})
        if uncomp > 0:
            trace.append({"implication": "pipeline_work_pending", "reason": f"{uncomp} uncompressed obs"})
        if pass_active and obc_mb > 0:
            trace.append({"implication": "downlink_opportunity", "reason": f"{obc_mb:.1f} MB on OBC"})

        return trace

    # ------------------------------------------------------------------
    # Schedule generation
    # ------------------------------------------------------------------

    def _avg_net_power_w(self, mode: str) -> float:
        """Average net power for a mode, accounting for eclipse fraction."""
        rates = self._power_consumption.get(mode, {"sun_w": 10.0, "eclipse_w": 10.0})
        sun_fraction = 1.0 - self._eclipse_fraction
        # Average consumption
        avg_consumption = (
            sun_fraction * rates["sun_w"] + self._eclipse_fraction * rates["eclipse_w"]
        )
        # Average generation (solar only when in sun)
        avg_generation = sun_fraction * self._solar_generation_w
        net = avg_generation - avg_consumption
        return net

    def _soc_delta_per_step(self, mode: str) -> float:
        """SOC change per step for a given mode (fraction of battery capacity)."""
        net_power_w = self._avg_net_power_w(mode)
        step_duration_h = self._step_duration_s / 3600.0
        if net_power_w >= 0:
            # Charging: apply efficiency
            energy_wh = net_power_w * step_duration_h * self._charge_efficiency
        else:
            energy_wh = net_power_w * step_duration_h
        return energy_wh / self._battery_capacity_wh

    def _steps_to_reach_soc(self, current_soc: float, target_soc: float) -> int:
        """Estimate steps to charge from current_soc to target_soc."""
        delta_per_step = self._soc_delta_per_step("charging")
        if delta_per_step <= 0:
            return 999  # Can't charge (shouldn't happen in practice)
        steps_needed = (target_soc - current_soc) / delta_per_step
        return max(1, int(steps_needed) + 1)

    def _generate_schedule(
        self, state: Dict[str, Any], gap_steps: int,
    ) -> List[Tuple[str, int]]:
        """Generate a greedy time-tagged schedule for the upcoming inter-pass gap.

        Uses average power model (eclipse_fraction-weighted) for conservative
        but realistic battery estimation.
        """
        # Simulated state
        sim_soc = state.get("battery_soc", 0.5)
        sim_uncomp = state.get("uncompressed_observations", 0)
        sim_undetected = state.get("undetected_observations", 0)
        sim_jetson_compressed_mb = state.get("jetson_compressed_mb", 0.0)
        sim_jetson_raw_mb = state.get("jetson_raw_mb", 0.0)
        sim_obc_mb = state.get("obc_data_mb", 0.0)
        # Cap observation at what we can physically downlink at the next pass
        # (50 kbps × contact); fall back to the daily-budget heuristic if unknown.
        achievable = state.get("achievable_downlink_mb")
        daily_budget_mb = achievable if achievable else state.get(
            "daily_downlink_budget_mb", self._daily_downlink_budget_mb
        )

        # Reserve the last chunk for charging (pre-pass battery buffer)
        reserve_fraction = self._charge_reserve_fraction
        reserve_steps = max(5, int(gap_steps * reserve_fraction))
        active_steps = gap_steps - reserve_steps

        schedule: List[Tuple[str, int]] = []
        remaining = active_steps

        # Precompute pipeline constants
        obs_compressed_mb = self._observation_size_mb / self._compression_ratio
        # compress_steps = actual env steps (compression_time_factor) + settling overhead
        # when transitioning from payload_observe → payload_compress (attitude maneuver out).
        compress_steps = int(self._compression_time_factor) + self._settling_time_steps
        # RS-485: 50 kbps = 50/8 kB/s = 6.25 kB/s = 0.00610 MB/s
        jetson_send_rate_mbs = self._jetson_to_obc_rate_kbps / 8.0 / 1000.0
        # Steps to allocate per observation: settling (non-productive) + 1 actual step.
        obs_schedule_steps = self._settling_time_steps + 1

        while remaining > 0:
            # ---- Battery critically low: charge first ----
            if sim_soc < self._min_soc_for_operations:
                target_soc = min(self._min_soc_for_operations + 0.15, 0.85)
                charge_dur = min(
                    self._steps_to_reach_soc(sim_soc, target_soc), remaining
                )
                charge_dur = max(1, charge_dur)
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

            # ---- Observe if pipeline is not saturated ----
            # Allocate obs_schedule_steps = settling_time_steps + 1 so that the
            # attitude-maneuver settling period is consumed before the 1 actual
            # observation step executes. Without this, the single scheduled step
            # is swallowed by settling and no data is ever collected.
            pipeline_mb = sim_obc_mb + sim_jetson_compressed_mb
            if (
                sim_soc > 0.60
                and pipeline_mb < daily_budget_mb
                and sim_obc_mb < self._obc_capacity_mb * 0.8
                and remaining >= obs_schedule_steps
            ):
                schedule.append(("payload_observe", obs_schedule_steps))
                # Settling steps execute as charging in the environment
                for _ in range(self._settling_time_steps):
                    sim_soc = min(1.0, sim_soc + self._soc_delta_per_step("charging"))
                # One actual observation step
                sim_soc = min(1.0, sim_soc + self._soc_delta_per_step("payload_observe"))
                sim_jetson_raw_mb += self._observation_size_mb
                sim_uncomp += 1
                remaining -= obs_schedule_steps
                continue

            # ---- Default: charge ----
            # Charge for up to 10 steps then re-evaluate
            dur = min(10, remaining)
            schedule.append(("charging", dur))
            for _ in range(dur):
                sim_soc = min(1.0, sim_soc + self._soc_delta_per_step("charging"))
            remaining -= dur

        # Reserve block at end: charging (pre-pass battery buffer for comms)
        if reserve_steps > 0:
            schedule.append(("charging", reserve_steps))

        # Merge consecutive identical modes for cleaner schedule
        return _merge_schedule(schedule)


def _merge_schedule(
    schedule: List[Tuple[str, int]]
) -> List[Tuple[str, int]]:
    """Merge consecutive entries with the same mode."""
    if not schedule:
        return []
    merged: List[Tuple[str, int]] = []
    cur_mode, cur_steps = schedule[0]
    for mode, steps in schedule[1:]:
        if mode == cur_mode:
            cur_steps += steps
        else:
            merged.append((cur_mode, cur_steps))
            cur_mode, cur_steps = mode, steps
    merged.append((cur_mode, cur_steps))
    return merged

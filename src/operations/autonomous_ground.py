"""Autonomous Ground operations paradigm.

Algorithmic ground-based operations where:
- Ground only sees telemetry that was downlinked during the last pass
- Commands can only be uplinked during ground passes
- During a pass: the agent instantly generates an optimal time-tagged schedule
  from fresh telemetry and uploads it to the satellite
- Between passes: the satellite executes the pre-uploaded schedule step-by-step
- If the schedule is exhausted, the satellite falls back to the default mode

This paradigm models a ground system where the planning is done by an
algorithmic scheduler (not human operators), so schedule generation is
instantaneous and optimal given the telemetry. Contrast with ConventionalGround,
which models realistic human operations with one-pass planning delay and
cognitive constraints.

Based on the information constraints described in:
  Sellmaier, Uhlig & Schmidhuber (2022) "Spacecraft Operations"
  Castano et al. (2022) "Operations for Autonomous Spacecraft"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.environment.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteState,
)
from src.operations.base import GroundKnowledge, OperationsParadigm


class AutonomousGround(OperationsParadigm):
    """Algorithmic ground-based operations with stale telemetry and schedule upload.

    The agent represents an algorithmic scheduler that:
    1. Only sees the satellite state as of the last downlink
    2. Can only send commands during ground passes
    3. During a pass: receives fresh telemetry, then instantly generates and
       uploads a complete time-tagged schedule covering the gap to the next pass
    4. Between passes: the satellite executes the schedule with zero onboard autonomy

    Unlike ConventionalGround, schedule generation is instantaneous and optimal —
    there is no planning latency or cognitive degradation. This is the algorithmic
    ideal of ground-based operations.

    Config options:
        default_mode: Mode to use when schedule is exhausted (default: "charging")
        orbital_period_steps: Estimated steps between passes (default: 93 = 5554s/60s)
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._default_mode = self.config.get("default_mode", "charging")
        self._orbital_period_steps = self.config.get("orbital_period_steps", 93)
        # Schedule: list of [mode, remaining_steps] pairs (mutable for countdown)
        self._schedule: List[List] = []
        self._schedule_index: int = 0

    def filter_observation(self, full_observation: Any, step: int) -> Any:
        """Return observation based on ground knowledge.

        During a ground pass, ground_pass_active is set to True (the ground
        operator knows they have a link). estimated_gap_steps is added to
        metadata so the schedule planner knows how long to plan for.

        Between passes, the agent sees only stale downlinked telemetry.
        """
        if full_observation is None:
            return None

        # Determine real-time ground pass status from the full observation
        real_ground_pass_active = False
        real_in_sunlight = False
        for sat in full_observation.constellation_state.satellites.values():
            if sat.metadata.get("ground_pass_active", False):
                real_ground_pass_active = True
            if sat.metadata.get("in_sunlight", False):
                real_in_sunlight = True

        self._ground_knowledge.staleness_steps = (
            step - self._ground_knowledge.last_update_step
        )

        gk = self._ground_knowledge
        metadata: Dict[str, Any] = {
            "in_sunlight": real_in_sunlight,
            "ground_pass_active": real_ground_pass_active,
            "uncompressed_observations": gk.uncompressed_observations,
            "total_observation_s": gk.observation_hours * 3600.0,
            "storage_capacity_mb": 1 * 1024 * 1024,  # 1 TB in MB
            "health_status": gk.health_status,
            "staleness_steps": gk.staleness_steps,
            "last_update_step": gk.last_update_step,
            "jetson_raw_mb": gk.jetson_raw_mb,
            "jetson_compressed_mb": gk.jetson_compressed_mb,
            "obc_data_mb": gk.obc_data_mb,
            "undetected_observations": gk.undetected_observations,
        }

        if real_ground_pass_active:
            # Provide the schedule planner with the estimated gap length
            metadata["estimated_gap_steps"] = self._orbital_period_steps

        stale_sat = SatelliteState(
            satellite_id="eventsat_0",
            position=[0.0, 0.0, 500.0],
            velocity=[0.0, 0.0, 0.0],
            resources={
                "battery_soc": gk.battery_soc,
                "data_stored_mb": gk.data_stored_mb,
                "obc_data_mb": gk.obc_data_mb,
                "data_downlinked_mb": 0.0,
            },
            status=gk.current_mode,
            metadata=metadata,
        )
        stale_constellation = ConstellationState(
            timestep=gk.last_update_step,
            epoch_seconds=gk.last_update_step * 60.0,
            satellites={"eventsat_0": stale_sat},
            global_info=full_observation.constellation_state.global_info,
        )
        return EnvironmentObservation(
            constellation_state=stale_constellation,
            tasks=full_observation.tasks,
            events=[],
        )

    def should_allow_inference(self, step: int, ground_pass_active: bool) -> bool:
        """Ground inference only during passes when fresh telemetry is available.

        Between passes, the satellite executes the pre-uploaded schedule and
        the ground has no new data to plan with (Rossi et al. 2023).
        """
        return ground_pass_active

    def can_act(self, step: int, ground_pass_active: bool) -> bool:
        """Commands can only be uplinked during ground passes."""
        return ground_pass_active

    def process_action(
        self,
        action: Dict[str, Any],
        step: int,
        ground_pass_active: bool,
    ) -> Dict[str, Any]:
        """During pass: extract and store schedule, execute immediate mode.
        Between passes: consume the schedule step by step.
        """
        if ground_pass_active:
            # Extract and store schedule if provided by the representation
            sat_action = action.get("eventsat_0", {})
            if "schedule" in sat_action:
                new_schedule = sat_action["schedule"]
                if new_schedule:
                    # Convert to mutable list of [mode, remaining_steps]
                    self._schedule = [
                        [mode, steps] for mode, steps in new_schedule
                    ]
                    self._schedule_index = 0

            # Strip schedule from action before passing to environment
            immediate_mode = sat_action.get("mode", self._default_mode)
            return {"eventsat_0": {"mode": immediate_mode}}

        # Between passes: consume schedule sequentially
        return self._consume_schedule()

    def _consume_schedule(self) -> Dict[str, Any]:
        """Pop the next mode from the schedule, advancing past exhausted entries."""
        while self._schedule_index < len(self._schedule):
            entry = self._schedule[self._schedule_index]
            mode, remaining = entry[0], entry[1]
            if remaining > 0:
                entry[1] -= 1
                if entry[1] == 0:
                    self._schedule_index += 1
                return {"eventsat_0": {"mode": mode}}
            self._schedule_index += 1

        # Schedule exhausted — fallback
        return {"eventsat_0": {"mode": self._default_mode}}

    def reset(self) -> None:
        super().reset()
        self._schedule = []
        self._schedule_index = 0

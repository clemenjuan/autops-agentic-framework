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

from typing import Any, Dict, List

from src.core.operations.base import OperationsParadigm


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
        # Schedule: list of [mode, remaining_steps] pairs (mutable for countdown)
        self._schedule: List[List] = []
        self._schedule_index: int = 0
        self._pass_through_observation = bool(
            self.config.get("pass_through_observation", False)
        )

    def filter_observation(self, full_observation: Any, step: int) -> Any:
        if self._pass_through_observation:
            return full_observation
        return self._stale_ground_observation(full_observation, step)

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
            # Non-EventSat scenarios, such as Flamingo-lite, use native action
            # dictionaries keyed by satellite id rather than EventSat mode
            # schedules. Keep the EventSat schedule path below untouched.
            if "eventsat_0" not in action:
                return action

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
        action, self._schedule_index = self._consume_schedule_list(
            self._schedule, self._schedule_index
        )
        return action

    def reset(self) -> None:
        super().reset()
        self._schedule = []
        self._schedule_index = 0

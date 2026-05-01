"""Conventional Ground operations paradigm — human-realistic.

Models real spacecraft ground operations as described in:
  Sellmaier, Uhlig & Schmidhuber (2022) "Spacecraft Operations" [SGJTLF4D]
  ECSS-E-ST-70C Ground Systems and Operations (2008) [CIYT2V68]
  Castano et al. (2022) "Operations for Autonomous Spacecraft" [2IJJ7ILS]
  Endsley (1995) "Situation Awareness in Dynamic Systems" [46MUS93H]

Real ground operations planning cycle:
  Pass N:   Downlink telemetry. Upload schedule S(N-1) planned after previous pass.
  Between:  Ground team analyses pass-N telemetry and plans schedule S(N).
            This takes the full inter-pass gap (hours for LEO missions).
  Pass N+1: Upload S(N). Downlink new telemetry. Start planning S(N+1).

Key consequence — ONE-PASS DELAY:
  The schedule executing between passes N and N+1 was planned based on
  telemetry from pass N-1 (two states ago). This is a fundamental
  constraint of conventional ground operations, not a tunable parameter.

Compare with AutonomousGround, where schedule generation is instant and
optimal (no planning delay). The difference isolates the cost of human
planning overhead and temporal knowledge staleness.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.operations.base import OperationsParadigm


class ConventionalGround(OperationsParadigm):
    """Human ground operations with realistic one-pass planning delay.

    The paradigm models a human flight dynamics team that:
    1. Receives fresh telemetry during each pass (ground knowledge updates)
    2. Plans the next schedule between passes — NOT during the pass
    3. Uploads the pre-planned schedule at the start of the next pass
    4. The satellite executes that schedule with zero onboard autonomy

    Unlike AutonomousGround, there is always a one-pass delay between
    receiving telemetry and the schedule based on it being executed.
    On cold start (first pass), no prior schedule exists — the satellite
    remains in default_mode until the second pass.

    Two internal schedule buffers:
        _active_schedule:  Currently being executed by the satellite.
                           Loaded from _planned_schedule at each pass start.
        _planned_schedule: Prepared by the representation from the latest
                           telemetry. Promoted to active at the next pass.

    Config options:
        default_mode: Mode when no active schedule exists (default: "charging")
        orbital_period_steps: Estimated steps between passes (default: 93)
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)

        # Active schedule: currently executing on the satellite
        self._active_schedule: List[List] = []
        self._active_index: int = 0

        # Planned schedule: prepared by representation, waiting for next pass
        self._planned_schedule: Optional[List[List]] = None

        # Track pass transitions to detect pass start
        self._last_pass_active: bool = False
        self._pass_upload_done: bool = False

    def filter_observation(self, full_observation: Any, step: int) -> Any:
        return self._stale_ground_observation(full_observation, step)

    def should_allow_inference(self, step: int, ground_pass_active: bool) -> bool:
        """Ground inference only during passes when fresh telemetry is available.

        Between passes, the ground has no new data to plan with. The one-pass
        planning delay is modeled by the two-buffer schedule system, not by
        inference timing (Rossi et al. 2023; Sellmaier et al. 2022).
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
        """Implement the one-pass planning delay.

        During pass:
          1. On pass start: promote _planned_schedule → _active_schedule (upload).
          2. Store any newly generated schedule as _planned_schedule (for next pass).
          3. Execute communication mode during the pass (downlinking + HK).

        Between passes:
          Consume _active_schedule step by step. If exhausted, use default_mode.

        Cold start (first pass, no prior _planned_schedule):
          No schedule to upload. Satellite stays in default_mode between first
          and second pass. This correctly models that the first schedule cannot
          be uploaded until the second pass.
        """
        if ground_pass_active:
            # Detect pass start (transition from no-pass → pass)
            if not self._last_pass_active:
                # Pass just started: promote planned → active (upload to satellite)
                if self._planned_schedule is not None:
                    self._active_schedule = self._planned_schedule
                    self._active_index = 0
                    self._planned_schedule = None
                # else: cold start, no schedule to upload; active remains as-is
                self._pass_upload_done = False

            self._last_pass_active = True

            # Capture newly generated schedule from the representation
            sat_action = action.get("eventsat_0", {})
            if "schedule" in sat_action and not self._pass_upload_done:
                new_schedule = sat_action["schedule"]
                if new_schedule:
                    # Store for upload at NEXT pass (the planning delay)
                    self._planned_schedule = [
                        [mode, steps] for mode, steps in new_schedule
                    ]
                self._pass_upload_done = True

            # During the pass: always communicate (downlink data + HK)
            return {"eventsat_0": {"mode": "communication"}}

        # Between passes
        self._last_pass_active = False
        return self._consume_active_schedule()

    def _consume_active_schedule(self) -> Dict[str, Any]:
        action, self._active_index = self._consume_schedule_list(
            self._active_schedule, self._active_index
        )
        return action

    def reset(self) -> None:
        super().reset()
        self._active_schedule = []
        self._active_index = 0
        self._planned_schedule = None
        self._last_pass_active = False
        self._pass_upload_done = False

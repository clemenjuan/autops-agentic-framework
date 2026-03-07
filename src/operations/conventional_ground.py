"""Conventional Ground operations paradigm.

Traditional ground-based operations where:
- Ground only sees telemetry that was downlinked during the last pass
- Commands can only be uplinked during ground passes
- Between passes, the satellite follows the last uploaded command sequence
- Information staleness increases between passes
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.environment.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteState,
)
from src.operations.base import GroundKnowledge, OperationsParadigm


class ConventionalGround(OperationsParadigm):
    """Traditional ground-based operations with stale telemetry.

    The agent represents a ground operator who:
    1. Only sees the satellite state as of the last downlink
    2. Can only send commands during ground passes
    3. Must pre-plan command sequences for autonomous execution between passes

    Config options:
        default_mode: Mode to use when no commands can be sent (default: "charging")
        command_sequence_horizon: Steps ahead to plan (default: 100)
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._default_mode = self.config.get("default_mode", "charging")
        self._command_sequence_horizon = self.config.get(
            "command_sequence_horizon", 100
        )
        self._last_commanded_action: Dict[str, Any] = {}

    def filter_observation(self, full_observation: Any, step: int) -> Any:
        """Return observation based on stale ground knowledge.

        Instead of real-time state, the agent sees a reconstructed
        observation from the last downlinked telemetry.
        """
        if full_observation is None:
            return None

        self._ground_knowledge.staleness_steps = (
            step - self._ground_knowledge.last_update_step
        )

        gk = self._ground_knowledge
        stale_sat = SatelliteState(
            satellite_id="eventsat_0",
            position=[0.0, 0.0, 500.0],
            velocity=[0.0, 0.0, 0.0],
            resources={
                "battery_soc": gk.battery_soc,
                "data_stored_mb": gk.data_stored_mb,
                "data_downlinked_mb": 0.0,
            },
            status=gk.current_mode,
            metadata={
                "in_sunlight": False,
                "ground_pass_active": False,
                "uncompressed_observations": 0,
                "total_observation_s": gk.observation_hours * 3600.0,
                "storage_capacity_mb": 512.0,
                "health_status": gk.health_status,
                "staleness_steps": gk.staleness_steps,
                "last_update_step": gk.last_update_step,
            },
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

    def can_act(self, step: int, ground_pass_active: bool) -> bool:
        """Commands can only be uplinked during ground passes."""
        return ground_pass_active

    def process_action(
        self,
        action: Dict[str, Any],
        step: int,
        ground_pass_active: bool,
    ) -> Dict[str, Any]:
        """Buffer actions during passes; replay last command otherwise."""
        if ground_pass_active:
            self._last_commanded_action = copy.deepcopy(action)
            return action

        if self._last_commanded_action:
            return self._last_commanded_action

        return {"eventsat_0": {"mode": self._default_mode}}

    def reset(self) -> None:
        super().reset()
        self._last_commanded_action = {}

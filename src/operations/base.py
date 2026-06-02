"""Operations Paradigm — Abstract Base Class.

The operations paradigm sits between the agent organization and the
environment, controlling information flow and action timing. It models
the fundamental difference between ground-based and onboard operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.environment.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteState,
)


@dataclass
class GroundKnowledge:
    """What ground operators know about the satellite.

    This represents the ground segment's view of the satellite state,
    which may be stale (only updated during downlink passes).
    """

    last_update_step: int = 0
    battery_soc: float = 0.8
    data_stored_mb: float = 0.0
    current_mode: str = "charging"
    health_status: str = "nominal"
    observation_hours: float = 0.0
    staleness_steps: int = 0
    # 3-pool pipeline fields (needed for schedule planning)
    obc_data_mb: float = 0.0
    jetson_raw_mb: float = 0.0
    jetson_compressed_mb: float = 0.0
    uncompressed_observations: int = 0
    undetected_observations: int = 0
    in_sunlight: bool = True


class OperationsParadigm(ABC):
    """Abstract base class for operations paradigms.

    An operations paradigm controls:
    - What the agent can observe (full state vs. stale ground knowledge)
    - When the agent can act (every step vs. uplink windows only)
    - How actions are buffered and delivered to the environment
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._ground_knowledge = GroundKnowledge()
        self._action_buffer: List[Dict[str, Any]] = []
        self._default_mode: str = self.config.get("default_mode", "charging")
        self._orbital_period_steps: int = self.config.get("orbital_period_steps", 93)

    @abstractmethod
    def filter_observation(
        self,
        full_observation: Any,
        step: int,
    ) -> Any:
        """Filter the full environment observation based on this paradigm.

        For onboard autonomy, this is a pass-through. For ground-based ops,
        this returns a stale snapshot based on last downlinked telemetry.

        Args:
            full_observation: The complete environment observation.
            step: Current simulation step.

        Returns:
            The observation the agent is allowed to see.
        """
        ...

    @abstractmethod
    def can_act(self, step: int, ground_pass_active: bool) -> bool:
        """Whether the agent can issue commands at this step.

        Args:
            step: Current simulation step.
            ground_pass_active: Whether a ground pass is currently active.

        Returns:
            True if the agent may issue an action this step.
        """
        ...

    @abstractmethod
    def process_action(
        self,
        action: Dict[str, Any],
        step: int,
        ground_pass_active: bool,
    ) -> Dict[str, Any]:
        """Process an agent action through the operations paradigm.

        May buffer, delay, or pass through depending on the paradigm.

        Args:
            action: The action the agent wants to execute.
            step: Current simulation step.
            ground_pass_active: Whether a ground pass is currently active.

        Returns:
            The action to actually execute (may be from buffer or no-op).
        """
        ...

    def _stale_ground_observation(
        self, full_observation: Any, step: int
    ) -> Any:
        """Return a stale ground-segment view of the satellite state.

        Builds an observation from the last downlinked telemetry.  Real-time
        pass and sunlight status are taken from the full observation so the
        ground segment knows when a link exists; all other fields come from
        the cached ground knowledge snapshot.
        """
        if full_observation is None:
            return None

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

    def _consume_schedule_list(
        self, schedule: List[List], idx: int
    ) -> Tuple[Dict[str, Any], int]:
        """Pop the next mode from a schedule list, advancing past exhausted entries.

        Args:
            schedule: List of [mode, remaining_steps] pairs (mutated in place).
            idx: Current position in the schedule.

        Returns:
            Tuple of (action_dict, new_index).
        """
        while idx < len(schedule):
            entry = schedule[idx]
            mode, remaining = entry[0], entry[1]
            if remaining > 0:
                entry[1] -= 1
                if entry[1] == 0:
                    idx += 1
                return {"eventsat_0": {"mode": mode}}, idx
            idx += 1
        return {"eventsat_0": {"mode": self._default_mode}}, idx

    def update_ground_knowledge(
        self,
        full_observation: Any,
        step: int,
    ) -> None:
        """Update ground knowledge from a successful telemetry downlink.

        Called by the environment/runner when telemetry is downlinked
        during a communication pass.

        Args:
            full_observation: Current full satellite state.
            step: Current simulation step.
        """
        obs = full_observation
        if obs is None:
            return
        cs = obs.constellation_state
        for sat_id, sat in cs.satellites.items():
            gk = self._ground_knowledge
            gk.last_update_step = step
            gk.battery_soc = sat.resources.get("battery_soc", gk.battery_soc)
            gk.data_stored_mb = sat.resources.get("data_stored_mb", gk.data_stored_mb)
            gk.current_mode = sat.status
            gk.health_status = sat.metadata.get("health_status", "nominal")
            gk.observation_hours = sat.metadata.get("total_observation_s", 0.0) / 3600.0
            gk.staleness_steps = 0
            # 3-pool pipeline fields
            gk.obc_data_mb = sat.resources.get(
                "obc_data_mb", sat.metadata.get("obc_data_mb", gk.obc_data_mb)
            )
            gk.jetson_raw_mb = sat.metadata.get("jetson_raw_mb", gk.jetson_raw_mb)
            gk.jetson_compressed_mb = sat.metadata.get(
                "jetson_compressed_mb", gk.jetson_compressed_mb
            )
            gk.uncompressed_observations = sat.metadata.get(
                "uncompressed_observations", gk.uncompressed_observations
            )
            gk.undetected_observations = sat.metadata.get(
                "undetected_observations", gk.undetected_observations
            )
            gk.in_sunlight = sat.metadata.get("in_sunlight", gk.in_sunlight)

    def get_ground_knowledge(self) -> GroundKnowledge:
        """Return the current ground knowledge snapshot."""
        return self._ground_knowledge

    def reset(self) -> None:
        """Reset paradigm state for a new episode."""
        self._ground_knowledge = GroundKnowledge()
        self._action_buffer.clear()

    def can_self_recover_anomaly(self) -> bool:
        """Whether onboard autonomy can clear anomalies without ground contact.

        Autonomous paradigms return True (onboard FDIR clears anomalies once
        the minimum safe-mode countdown expires). Ground-based paradigms return
        False (ground must send a resume command during a pass).
        """
        return False

    def should_allow_inference(self, step: int, ground_pass_active: bool) -> bool:
        """Whether the representation should run full inference this step.

        For ground-based paradigms, inference is only meaningful when fresh
        telemetry is available (during ground passes). Between passes, the
        satellite executes the pre-uploaded schedule and ground has no new
        data to act on (Rossi et al. 2023: ground computation triggered by
        "data collected in prior downlinks").

        For onboard paradigms, inference runs every step.

        Args:
            step: Current simulation step.
            ground_pass_active: Whether a ground pass is currently active.

        Returns:
            True if the representation should run full inference.
        """
        return True  # Default: always allowed (backward compatible)

    def has_onboard_autonomy(self) -> bool:
        """Whether per-step decision-making runs onboard.

        True for onboard / hybrid paradigms — the onboard compute (Jetson) is
        kept powered every step, adding a continuous power draw (see
        ``EventSatEnvironment.onboard_autonomy_active``). False for ground
        paradigms, where decisions are made on the ground.
        """
        return False

    def get_metrics(self) -> Dict[str, float]:
        """Paradigm-level metrics for the episode (default: none)."""
        return {}

    def get_name(self) -> str:
        """Return the paradigm name."""
        return self.__class__.__name__

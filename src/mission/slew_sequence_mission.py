"""ADCS slew mission.

Setup the slew mission class as child of abstract mission class

Here the structure and functionality of the specific slew mission is defined.
The core of the mission is to track a list of specific generated attitudes one by one.
This presents the simplest mission and is intended to check the functionality ofg the steup and control algorithm

The target attitudes are inertially fixed, so the setpoint rate is always zero
and the mission never needs the environment: every decision it makes comes from
the pointing error and the clock.

Per-episode state lives entirely in MissionState:
    setpoint    the setpoint currently being tracked
    target_idx  how many targets have been cleared
    hold_timer  seconds spent continuously inside the tolerance
    extra       (targets,) -- the tuple of Setpoints sampled in reset()

The targets are stored as Setpoints rather than as bare quaternions: it is what
MissionState.setpoint already holds, so advancing a target is an index rather
than a construction. Their target_omega_body is zero throughout, which is not
waste but the type stating that this mission's targets do not move.

The mission object itself holds only its config, so one instance can be shared
across vectorised environments without them corrupting each other.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Tuple

import numpy as np

from src.mission.adcs_mission_base import (
    Mission,
    MissionEvents,
    MissionState,
)
from src.environment.orbital.adcs.configs import SlewSequenceConfig
from src.environment.orbital.adcs.control import Setpoint
from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.propagator import EnvironmentData


class SlewSequenceMission(Mission):
    """Visit a sequence of inertially fixed attitudes, one at a time.

    A target counts as cleared once the pointing error has stayed inside
    ``cfg.tolerance_deg`` for ``cfg.hold_time`` consecutive seconds; the hold
    timer resets the moment the error leaves the tolerance, so a target cannot
    be cleared by slewing through it at rate. The mission is done once every
    target has been cleared.
    """

    def __init__(self, cfg: SlewSequenceConfig) -> None:
        """Store the config. Narrows the base annotation to the config type
        this mission actually reads."""
        super().__init__(cfg)
        self.cfg: SlewSequenceConfig = cfg

    # -------------------------------------------------------------------------
    # Mission interface
    # -------------------------------------------------------------------------

    def reset(self, state: SatState, rng: np.random.Generator) -> MissionState:
        """Sample this episode's target attitudes and open on the first one.

        Args:
            state: True state at the start of the episode; the attitude the
                first target is sampled against.
            rng: The episode's seeded generator. The sole source of randomness,
                so the same seed reproduces the same target sequence.

        Returns:
            The opening MissionState: target 0's setpoint, a zeroed hold timer,
            and the sampled target setpoints in ``extra``.
        """
        targets = self._sample_targets(state,rng)
        return MissionState(
            setpoint=targets[0],
            target_idx=0,
            hold_timer=0.0,
            extra=(targets,)
        )

    def update(
        self,
        ms: MissionState,
        state: SatState,
        env_data: EnvironmentData,
        pointing_error_deg: float,
        dt: float,
    ) -> Tuple[MissionState, MissionEvents]:
        """Advance the hold timer and, on a clear, move to the next target.

        Args:
            ms: The mission state from the previous step.
            state: True state after this step's integration.
            env_data: Unused by this mission -- the targets are inertially fixed,
                so no orbital or environmental geometry is involved.
            pointing_error_deg: Angle between the current and target attitude
                [deg], computed by the env so there is one definition of it.
            dt: Timestep [s]; what the hold timer accumulates.

        Returns:
            The new mission state, and the events that fired on this step.
            ``target_cleared`` fires on the step a target's hold completes;
            ``mission_done`` fires on the step the last one does, so both fire
            together on the final target.
        """
        me = MissionEvents()
        targets = self._targets(ms)

        # Chceck if one target is cleared and weather the mission is accomplished 
        if pointing_error_deg <= self.cfg.tolerance_deg:
                ms.hold_timer += dt
        
                if ms.hold_timer >= self.cfg.hold_time:
                    me.target_cleared = True
                    ms.target_idx += 1
                    ms.hold_timer = 0.0

                    #If all targets are done we mark thge mission as done
                    if ms.target_idx >= len(targets):
                         me.mission_done=True
                    else:
                        ms.setpoint = targets[ms.target_idx]
        
        else:
             ms.hold_timer = 0.0
        
        return ms, me

    def info(self, ms: MissionState) -> Dict[str, Any]:
        """Diagnostics for the env's info dict.

        Extends the base fields with the hold timer, which is what separates a
        policy that is genuinely settling on a target from one that keeps
        drifting in and out of tolerance. Both look identical in ``target_idx``
        and in the pointing error alone, since the error dips inside tolerance
        either way; a sawtooth hold timer that never reaches ``cfg.hold_time``
        is the only visible signature of the second.

        ``targets_total`` is carried so ``target_idx`` is interpretable without
        joining a rollout back against the config.
        """
        info = super().info(ms)
        info.update(
            {
                "mission/hold_timer": float(ms.hold_timer),
                "mission/targets_total": len(self._targets(ms)),
            }
        )
        return info

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def max_target_events(self) -> int:
        """Target clears attainable in one episode.

        Read by adcs_train_ppo.estimate_return_bound to size vf_clip_param.
        """
        return self.cfg.num_targets

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _sample_targets(
        self, state: SatState, rng: np.random.Generator
    ) -> Tuple[Setpoint, ...]:
        """Draw the episode's target setpoints.

        Args:
            state: Start-of-episode state, so targets can be sampled relative
                to the attitude the satellite actually starts in.
            rng: The episode's seeded generator.

        Returns:
            ``cfg.num_targets`` Setpoints, each holding a normalised attitude
            quaternion (ECI to body, scalar-first) at zero target rate.
        """
        q = rng.normal(size=(self.cfg.num_targets, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return tuple(
        Setpoint(target_q_eci_body=qi, target_omega_body=np.zeros(3))
        for qi in q
        )

    @staticmethod
    def _targets(ms: MissionState) -> Tuple[Setpoint, ...]:
        """Unpack the target setpoints out of ``ms.extra``.

        Keeps the ``extra = (targets,)`` convention in one place rather than
        spreading ``ms.extra[0]`` through the class.
        """
        return ms.extra[0]
    
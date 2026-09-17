"""ADCS target track mission.

In this mission the satellite is tasked to find and track moving targets.
The targets in this case are on actual orbits and also get propagated.
The satellite can only select a target which is within its field of view whenever
it is searching for one.

The transition between targets and their selection is handled by a
separate algorithm and not the agent itself.

For now to find a target the satellite targets a given setpoint and waits to see if targets cross.
This might have to be changed to a more sophisticated method later on.

The targets are generated in dependence of the satellite's orbit such that they are
for sure visible to the satellite. If the number of targets increase in later generations
this might no longer be necessary.

Per-episode state
-----------------
All of it lives in MissionState, none of it on the mission object, so one
instance can be shared across vectorised environments::

    setpoint    where to point right now, rebuilt every step (see below)
    target_idx  how many targets have been cleared == len(cleared)
    hold_timer  seconds spent continuously inside the tolerance
    extra       (targets, cleared, current)

``targets`` is the episode's tuple of Target objects and a target's position in
it is its identity for the episode -- no separate id field is needed, and an int
logs cleanly. ``cleared`` is the set of those positions already tracked to
completion; ``current`` is the one being tracked now, or None while searching.

Targets are not cleared in order here -- whatever crosses the field of view gets
picked -- so ``target_idx`` cannot double as a pointer to the current target the
way it does in SlewSequenceMission. It stays a count; ``current`` is the pointer.
``current is None`` also *is* the search condition, so no separate search flag is
kept: two variables encoding one fact can disagree, one cannot.

The cleared set deliberately sits here rather than as a flag on Target. Target
objects are rebuilt every reset today, but building one constructs an Orekit
propagator, so they are a natural thing to cache across episodes later. A flag on
a cached Target would survive into the next episode and silently make that target
unselectable forever; a set in MissionState cannot outlive its episode.

Pointing rate
-------------
Setpoints carry a zero target rate even though the targets move. The rate is
withheld on purpose: handing it over as feedforward would solve part of the
control problem inside the environment, and the agent is meant to derive it.

That choice makes the task partially observable as the observation currently
stands. The observation is (q_error, omega_body, wheel_speeds), and the target's
rate appears in none of them -- it is only recoverable from how q_error changes
between steps. Two situations produce an identical observation and want
different actions: a stationary target with the satellite at rest and a small
error, versus a moving target with the satellite at rest and the same error. The
first wants a small correction, the second wants the satellite brought up to the
target's rate. A memoryless policy cannot tell them apart, so it settles at
whatever steady-state offset its feedback balances the unmodelled motion at.

Closing that gap means giving the agent the missing quantity -- the target rate,
or better the error rate (omega_body - omega_target), which is what the
controller actually has to null -- or giving it memory, through frame stacking
or a recurrent policy. Adding an observation channel keeps the policy
feedforward and does not favour one architecture over another, which matters
when architectures are what the experiment compares.

Layout
------
Module level holds the target-generation parameters, then the attitude helpers,
then the Target wrapper. Inside the mission the methods follow the order
``update`` calls them -- visibility, then selection and the hold, then the
setpoint -- with episode setup (target generation) last. Only the three
Mission interface methods are public; everything else is a private helper.
"""
from __future__ import annotations

import numpy as np
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

from src.environment.orbital.propagator import EnvironmentData, create_orbit_track

from src.mission.adcs_mission_base import Mission, MissionState, MissionEvents
from src.environment.orbital.adcs.configs import TargetTrackConfig, OrbitConfig
from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.adcs.control import Setpoint
from src.environment.orbital.adcs.dynamics import dcm_eci_to_body
from src.environment.orbital.adcs.constants import R_EARTH, MU_EARTH


# =============================================================================
# Target generation parameters
# =============================================================================

# Targets sit ABOVE the satellite, never below: the search attitude points
# radially outward, so a lower target is in the nadir direction and could never
# enter the field of view.
#
# The altitude separation is large because it sets how wide the search cone is
# where the target actually flies: the cone subtends only altitude*tan(fov) at
# the target's shell, so at 20 km a target would have to pass within 0.06 deg of
# straight overhead. At 1000-2000 km that tolerance is a few degrees, which is
# what makes the angular spread below usable.
ALTITUDE_MIN_KM = 1000.0
ALTITUDE_MAX_KM = 2000.0

# Spread applied to inclination, argument of perigee and the node, symmetric
# about the satellite's own orbit. Kept small deliberately: the current purpose
# is to exercise tracking, not acquisition, and the search and selection
# algorithms are not written yet.
ANGULAR_SPREAD_DEG = 0.5

# Whether each target is placed so that it ALREADY sits in the field of view at
# episode start.
#
# True while this mission is used to exercise TRACKING: search and selection are
# not implemented, so an episode that opens with nothing in view has nothing to
# test. Set False for the intended behaviour -- true anomaly drawn at random, so
# a target enters the field of view on its own or not at all, and acquisition
# becomes the search algorithm's problem. Targets sit high enough to have a
# noticeably longer period than the satellite, so they drift in and out of view
# rather than staying put: expect only about 5% of an orbit to have any target
# visible once this is off.
PHASE_TARGETS_INTO_VIEW = True

# Steps used to search for the phase that puts a target in view. 72 is 5 deg of
# true anomaly, well inside the field of view at these altitudes. Unused when
# PHASE_TARGETS_INTO_VIEW is False.
PHASE_SAMPLES = 72


# =============================================================================
# Attitude helpers
# =============================================================================

def _triad_dcm(
    primary_a: np.ndarray,
    secondary_a: np.ndarray,
    primary_b: np.ndarray,
    secondary_b: np.ndarray,
) -> np.ndarray:
    """Rotation taking frame B to frame A, from two vector pairs (TRIAD).

    ``primary_a`` is matched to ``primary_b`` exactly; the secondary pair only
    resolves the rotation about that axis, so it need not be perpendicular to
    the primary and is used only through its component across it.

    Args:
        primary_a: Primary direction in frame A.
        secondary_a: Secondary direction in frame A, pinning the roll.
        primary_b: The same primary direction expressed in frame B.
        secondary_b: The same secondary direction expressed in frame B.

    Returns:
        The 3x3 matrix C with ``v_a = C @ v_b``.

    Raises:
        ValueError: If either pair is parallel, which leaves the rotation about
            the primary axis undetermined.
    """
    def _triad(primary: np.ndarray, secondary: np.ndarray, name: str) -> np.ndarray:
        e1 = primary / np.linalg.norm(primary)
        cross = np.cross(e1, secondary)
        norm = np.linalg.norm(cross)
        if norm < 1e-9:
            raise ValueError(
                f"{name} vectors are parallel, so the roll about the primary "
                f"axis is undetermined: primary={primary}, secondary={secondary}"
            )
        e2 = cross / norm
        return np.column_stack([e1, e2, np.cross(e1, e2)])

    return _triad(primary_a, secondary_a, "body") @ _triad(
        primary_b, secondary_b, "reference"
    ).T


def _quat_from_dcm(dcm: np.ndarray) -> np.ndarray:
    """Scalar-first quaternion for an ECI->body DCM.

    Inverse of ``dcm_eci_to_body``: that builds the Hamilton body->ECI matrix
    and transposes it, so the extraction here runs on the transpose to match.
    Branches on the largest diagonal term, the standard guard against dividing
    by a near-zero square root when the rotation approaches 180 degrees.

    Args:
        dcm: 3x3 rotation with ``v_body = dcm @ v_eci``.

    Returns:
        The quaternion [w, x, y, z], normalised, with a non-negative scalar part.
    """
    m = np.asarray(dcm, dtype=float).T          # body->ECI, the Hamilton form
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        s = 2.0 * np.sqrt(1.0 + trace)
        q = np.array([0.25 * s,
                      (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([(m[2, 1] - m[1, 2]) / s,
                      0.25 * s,
                      (m[0, 1] + m[1, 0]) / s,
                      (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 2] - m[2, 0]) / s,
                      (m[0, 1] + m[1, 0]) / s,
                      0.25 * s,
                      (m[1, 2] + m[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[1, 0] - m[0, 1]) / s,
                      (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s,
                      0.25 * s])

    q = q / np.linalg.norm(q)
    # q and -q are the same rotation; fixing the sign keeps the setpoint from
    # flipping representation between steps, which the error quaternion would
    # otherwise have to undo every time.
    return -q if q[0] < 0.0 else q


# =============================================================================
# Targets
# =============================================================================

class Target:
    """One tracked object, on its own propagated orbit.

    A thin wrapper turning an OrbitConfig into something that can be asked where
    it is at a given mission time. Holds no mission state: whether a target has
    been tracked lives in MissionState, for the reason given in the module
    docstring.

    Attributes:
        orbit: The orbit this target flies.
        orbit_track: The Orekit propagator built from it, whose t=0 is the
            orbit's epoch and so the mission clock's zero.
    """

    def __init__(self, orbit: OrbitConfig) -> None:
        """Build the propagator for ``orbit``.

        Args:
            orbit: The target's orbit. Construction is not cheap -- it builds an
                Orekit analytical propagator -- so targets are made once per
                episode rather than per step.
        """
        self.orbit = orbit
        self.orbit_track = create_orbit_track(orbit)

    def r_eci(self, t: float) -> np.ndarray:
        """Position at mission time ``t`` [s], in ECI [m], shape (3,)."""
        return self.orbit_track.state_at(t)[0]

    def v_eci(self, t: float) -> np.ndarray:
        """Velocity at mission time ``t`` [s], in ECI [m/s], shape (3,)."""
        return self.orbit_track.state_at(t)[1]


# =============================================================================
# Mission
# =============================================================================

class TargetTrackMission(Mission):
    """Search for moving targets and track each one in turn.

    A target counts as cleared once the pointing error has stayed inside
    ``cfg.tolerance_deg`` for ``cfg.hold_time`` consecutive seconds, and no
    target is ever cleared twice. A target that leaves the field of view before
    its hold completes is released rather than held on to. The mission is done
    once every target has been cleared.
    """

    def __init__(self, cfg: TargetTrackConfig, sat_orbit: OrbitConfig) -> None:
        """Store the config and the orbit targets are generated against.

        Narrows the base annotation to the config type this mission actually
        reads. Nothing episode-specific is stored, so one instance can be shared
        across vectorised environments.

        Args:
            cfg: This mission's config.
            sat_orbit: The satellite's own orbit. Target orbits are drawn as
                offsets from it, so it is a property of the scenario rather than
                of any one episode.
        """
        super().__init__(cfg)
        self.cfg: TargetTrackConfig = cfg
        self.sat_orbit = sat_orbit

    # -------------------------------------------------------------------------
    # Mission interface
    # -------------------------------------------------------------------------

    def reset(self, state: SatState, rng: np.random.Generator) -> MissionState:
        """Draw this episode's targets and open on whatever is already in view.

        Args:
            state: True state at the start of the episode.
            rng: The episode's seeded generator, the sole source of randomness,
                so the same seed reproduces the same targets.

        Returns:
            The opening MissionState. ``current`` is None, and the setpoint is
            the search attitude, whenever nothing is in the field of view.
        """
        targets = self._generate_targets(self.sat_orbit, rng, state)
        cleared: Set[int] = set()

        visible = self._check_visibility(state, targets)
        current = self._select_target(state, visible, cleared)

        return MissionState(
            setpoint=self._setpoint_for(state, targets, current),
            target_idx=0,
            hold_timer=0.0,
            extra=(targets, cleared, current),
        )

    def update(
        self,
        ms: MissionState,
        state: SatState,
        env_data: EnvironmentData,
        pointing_error_deg: float,
        dt: float,
    ) -> Tuple[MissionState, MissionEvents]:
        """Advance the hold, release or clear the tracked target, re-aim.

        Args:
            ms: The mission state from the previous step.
            state: True state after this step's integration. Supplies the
                attitude the field-of-view test runs against, and the position
                every setpoint is built from.
            env_data: Unused. The geometry this mission needs comes from the
                targets' own propagated orbits, not from the satellite's
                environment sample.
            pointing_error_deg: Angle between current and target attitude [deg],
                computed by the env so there is one definition of it.
            dt: Timestep [s]; what the hold timer accumulates.

        Returns:
            The new mission state and the events that fired this step.
            ``target_cleared`` fires on the step a target's hold completes;
            ``mission_done`` fires on the step the last one does, so both fire
            together on the final target.
        """
        me = MissionEvents()
        targets, cleared, current = ms.extra

        visible = self._check_visibility(state, targets)

        # A target that leaves the field of view before its hold completes is
        # released rather than kept. Holding on would pin the mission to
        # something the camera can no longer see.
        if current is not None and current not in visible:
            current = None
            ms.hold_timer = 0.0

        if current is not None:
            current = self._check_tracking(ms, me, current, pointing_error_deg, dt)

        # Re-select in the same step the previous target was cleared or lost.
        if current is None and not me.mission_done:
            current = self._select_target(state, visible, cleared)

        # Rebuilt every step, never cached. The targets move and the setpoint
        # carries no rate by design, so a setpoint fixed at acquisition would aim
        # at where the target was.
        ms.setpoint = self._setpoint_for(state, targets, current)
        ms.extra = (targets, cleared, current)
        return ms, me

    def info(self, ms: MissionState) -> Dict[str, Any]:
        """Diagnostics for the env's info dict.

        ``hold_timer`` separates a policy genuinely settling on a target from one
        drifting in and out of tolerance; both look identical in ``target_idx``
        alone. ``tracking`` distinguishes an episode that never acquired anything
        from one that acquired and kept losing it -- again indistinguishable from
        the cleared count.

        ``targets_total`` is carried so ``target_idx`` is interpretable without
        joining a rollout back against the config.
        """
        targets, _, current = ms.extra
        info = super().info(ms)
        info.update(
            {
                "mission/hold_timer": float(ms.hold_timer),
                "mission/targets_total": len(targets),
                "mission/tracking": current is not None,
            }
        )
        return info

    # -------------------------------------------------------------------------
    # Visibility
    # -------------------------------------------------------------------------

    def _check_visibility(
        self, state: SatState, targets: List[Target]
    ) -> Tuple[int, ...]:
        """Which targets are inside the field of view right now.

        A geometric question only: already-cleared targets are still reported,
        because "in view" and "worth pointing at" are different facts and the
        second belongs to ``_select_target``. Keeping them apart is what lets
        info report how many targets are in view independently of how many are
        left.

        Args:
            state: Current state; supplies the attitude and position the cone
                test runs against.
            targets: This episode's targets.

        Returns:
            Positions in ``targets``, ascending. Positions rather than Target
            objects because the position is the target's identity for the
            episode -- see the module docstring.
        """
        return tuple(
            idx
            for idx, target in enumerate(targets)
            if self._check_cone(state, target)
        )

    def _check_cone(self, state: SatState, target: Target) -> bool:
        """Whether one target is inside the camera's cone and not occulted.

        Args:
            state: Current state; supplies the attitude and position.
            target: The target to test.

        Returns:
            True if the target lies within ``cfg.fov_half_angle`` of the
            boresight AND Earth does not block the line of sight.
        """
        t = state.t

        # Determine the direction the body is in body coordinates
        los_eci = target.r_eci(t) - state.r_eci
        range_m = np.linalg.norm(los_eci)
        los_body = dcm_eci_to_body(state.q_eci_body) @ (los_eci / range_m)

        # Check offset angle. Works since both are normed vectors.
        cos_off = float(los_body @ self.cfg.boresight_body)

        if self._check_line_of_sight(state.r_eci, los_eci):
            return cos_off >= np.cos(self.cfg.fov_half_angle)

        return False

    def _check_line_of_sight(self, r_sat: np.ndarray, los: np.ndarray) -> bool:
        """Whether Earth blocks the segment from the satellite to the target.

        Closest approach of the segment to Earth's centre, clamped to the
        segment itself so a target behind the satellite is not mistaken for an
        occultation.

        Args:
            r_sat: Satellite position in ECI [m], shape (3,).
            los: Vector from satellite to target in ECI [m], shape (3,); not
                normalised, since its length defines the segment.

        Returns:
            True if the line of sight clears Earth's surface.
        """
        s_prime = -np.dot(r_sat, los) / np.dot(los, los)
        s_prime = np.clip(s_prime, 0, 1)

        distance = np.linalg.norm(r_sat + s_prime * los)
        return distance >= R_EARTH

    # -------------------------------------------------------------------------
    # Target lifecycle: pick one, hold it, clear it
    # -------------------------------------------------------------------------

    def _select_target(
        self, state: SatState, visible: Tuple[int, ...], cleared: Set[int]
    ) -> Optional[int]:
        """Pick the next target to track, or None to keep searching.

        For now the lowest-numbered target in view that has not been cleared.
        Later this wants an actual algorithm -- or the agent itself, if target
        choice is ever handed to it.

        Args:
            state: Current state. Unused by this rule, but every non-trivial
                selection rule needs it (range, closing rate, how long the target
                will stay in view), so it stays in the signature.
            visible: Positions currently in the field of view.
            cleared: Positions already tracked to completion; skipped, so no
                target is tracked twice.

        Returns:
            The chosen position in ``targets``, or None if nothing is selectable.
        """
        for idx in visible:
            if idx not in cleared:
                return idx
        return None

    def _check_tracking(
        self,
        ms: MissionState,
        me: MissionEvents,
        current: int,
        pointing_error_deg: float,
        dt: float,
    ) -> Optional[int]:
        """Advance the hold on the tracked target and clear it once it completes.

        The hold timer resets the moment the error leaves the tolerance, so a
        target cannot be cleared by slewing through it at rate.

        Mutates ``ms.hold_timer`` and ``ms.target_idx``, adds to the cleared set
        in ``ms.extra``, and sets the events on ``me``.

        Args:
            ms: Mission state being advanced.
            me: Events for this step; ``target_cleared`` and ``mission_done``
                fire together on the last target.
            current: Position of the target being tracked.
            pointing_error_deg: Angle to the setpoint [deg].
            dt: Timestep [s].

        Returns:
            ``current`` while the hold is still running, None once the target has
            been cleared and the mission needs a new one.
        """
        targets, cleared, _ = ms.extra

        if pointing_error_deg > self.cfg.tolerance_deg:
            ms.hold_timer = 0.0
            return current

        ms.hold_timer += dt
        if ms.hold_timer < self.cfg.hold_time:
            return current

        cleared.add(current)
        me.target_cleared = True
        # A count, not a pointer: targets are cleared in whatever order they
        # cross the field of view. See the module docstring.
        ms.target_idx = len(cleared)
        ms.hold_timer = 0.0

        if len(cleared) >= len(targets):
            me.mission_done = True

        return None

    # -------------------------------------------------------------------------
    # Pointing
    # -------------------------------------------------------------------------

    def _setpoint_for(
        self, state: SatState, targets: List[Target], current: Optional[int]
    ) -> Setpoint:
        """The setpoint for the target being tracked, or the search attitude.

        One place deciding what "no target" points at, so the search and track
        cases cannot drift apart.

        Args:
            state: Current state.
            targets: This episode's targets.
            current: Position of the target being tracked, or None to search.

        Returns:
            The setpoint to hold this step.
        """
        if current is None:
            return self._search_setpoint(state)
        return self._generate_setpoint(state, targets[current])

    def _search_setpoint(self, state: SatState) -> Setpoint:
        """Point the camera radially outward, to wait for a target to cross.

        Zenith is not a fixed direction: it sweeps through a full turn once per
        orbit, so this cannot be a constant attitude and the setpoint has to be
        rebuilt each step. The rate is still left at zero, for the reason given
        in ``_pointing_setpoint``.

        Args:
            state: Current state; supplies the position zenith is taken from.

        Returns:
            The zenith-pointing setpoint, at zero target rate.
        """
        zenith_eci = state.r_eci / np.linalg.norm(state.r_eci)
        return self._pointing_setpoint(state, zenith_eci)

    def _generate_setpoint(self, state: SatState, target: Target) -> Setpoint:
        """Point the camera at one target, from the current relative geometry.

        Args:
            state: Current state; supplies the satellite position and the time
                the target's position is evaluated at.
            target: The target to aim at.

        Returns:
            The target-pointing setpoint, at zero target rate.
        """
        los_eci = target.r_eci(state.t) - state.r_eci
        los_hat = los_eci / np.linalg.norm(los_eci)
        return self._pointing_setpoint(state, los_hat)

    def _pointing_setpoint(
        self, state: SatState, direction_eci: np.ndarray
    ) -> Setpoint:
        """Setpoint putting the boresight on ``direction_eci``.

        Aiming the boresight leaves roll about it free, and the reward reads the
        full attitude error, so the roll is pinned by aligning
        ``cfg.reference_body`` with the orbit normal -- an orbit-frame-like
        attitude that varies smoothly rather than an arbitrary constant.

        The target rate is left at zero deliberately, and unlike
        SlewSequenceMission that is not because the target holds still. These
        targets move! Handing the rate over as feedforward
        would be solving part of the control problem in the environment. The
        agent is asked to work it out instead, so the setpoint states where to
        point and nothing about how to follow it.

        Note this leaves the observation non-Markov for tracking: see the
        pointing-rate note in the module docstring.

        Args:
            state: Current state; supplies position and velocity for the orbit
                normal.
            direction_eci: Unit vector the boresight should point along, ECI.

        Returns:
            The setpoint attitude, at zero target rate.
        """
        orbit_normal_eci = np.cross(state.r_eci, state.v_eci)
        orbit_normal_eci = orbit_normal_eci / np.linalg.norm(orbit_normal_eci)

        eci_to_body = _triad_dcm(
            primary_a=self.cfg.boresight_body,
            secondary_a=self.cfg.reference_body,
            primary_b=direction_eci,
            secondary_b=orbit_normal_eci,
        )

        return Setpoint(
            target_q_eci_body=_quat_from_dcm(eci_to_body),
            target_omega_body=np.zeros(3),
        )

    # -------------------------------------------------------------------------
    # Target generation (episode setup)
    # -------------------------------------------------------------------------

    def _generate_targets(
        self, sat_orbit: OrbitConfig, rng: np.random.Generator, state: SatState
    ) -> List[Target]:
        """Draw this episode's targets.

        Each target is a complete random orbit on its own. When
        PHASE_TARGETS_INTO_VIEW is set, one extra step then rewrites its true
        anomaly so it opens inside the field of view; dropping that step is all
        it takes to go back to freely phased targets.

        Args:
            sat_orbit: The satellite's orbit; targets are generated relative to it.
            rng: The episode's seeded generator, the sole source of randomness.
            state: Start-of-episode state. Only read when phasing, to supply
                the position the targets are placed against.

        Returns:
            ``cfg.num_targets`` targets, each on its own propagated orbit.
        """
        targets = []
        for _ in range(self.cfg.num_targets):
            orbit = self._generate_random_orbit(sat_orbit, rng)

            if PHASE_TARGETS_INTO_VIEW:
                orbit = replace(
                    orbit, true_anomaly_deg=self._phase_into_view(orbit, state)
                )

            targets.append(Target(orbit))

        return targets

    def _generate_random_orbit(
        self, sat_orbit: OrbitConfig, rng: np.random.Generator
    ) -> OrbitConfig:
        """One target orbit, offset from the satellite's.

        Complete on its own: the true anomaly is drawn across the whole orbit, so
        where the target starts is left to chance. ``_generate_targets``
        overrides that only while PHASE_TARGETS_INTO_VIEW is set.

        Args:
            sat_orbit: The orbit the offsets are applied to.
            rng: The episode's seeded generator.

        Returns:
            The target's orbit config.
        """
        spread = ANGULAR_SPREAD_DEG
        node_offset_deg = rng.uniform(-spread, spread)

        if sat_orbit.ltan_hours is not None:
            # Offsetting the node through LTAN resolves to exactly that offset in
            # RAAN: both orbits share an epoch, so the Sun's right ascension is
            # common to both conversions and cancels. Setting raan_deg instead
            # would read the satellite's field, which is a dead placeholder
            # whenever LTAN is set -- EventSat carries 0.0 against an effective
            # 258 deg, which would put every target on the far side of Earth.
            ltan_hours = sat_orbit.ltan_hours + node_offset_deg / 15.0
            raan_deg = sat_orbit.raan_deg          # ignored downstream
        else:
            ltan_hours = None
            raan_deg = sat_orbit.raan_deg + node_offset_deg

        return OrbitConfig(
            epoch=sat_orbit.epoch,
            # One-sided, and never below: see the altitude note at module level.
            altitude_km=sat_orbit.altitude_km
            + rng.uniform(ALTITUDE_MIN_KM, ALTITUDE_MAX_KM),
            eccentricity=sat_orbit.eccentricity,
            inclination_deg=sat_orbit.inclination_deg + rng.uniform(-spread, spread),
            raan_deg=raan_deg,
            arg_perigee_deg=sat_orbit.arg_perigee_deg + rng.uniform(-spread, spread),
            true_anomaly_deg=rng.uniform(0.0, 360.0),
            ltan_hours=ltan_hours,
            propagator_type=sat_orbit.propagator_type,
        )

    def _phase_into_view(self, orbit: OrbitConfig, state: SatState) -> float:
        """True anomaly [deg] placing this target nearest the satellite's zenith.

        Scaffolding for the tracking-only phase of the mission, not physics: it
        overrides where the target would naturally be. See
        PHASE_TARGETS_INTO_VIEW.

        Advancing a circular orbit's true anomaly by d is the same as advancing
        time by d/n, so sampling one track across one period covers every phase.
        That costs PHASE_SAMPLES propagations rather than as many propagator
        constructions.

        Args:
            orbit: The target orbit, with any true anomaly.
            state: Start-of-episode state; supplies time and position.

        Returns:
            True anomaly in degrees, wrapped to [0, 360).
        """
        track = create_orbit_track(orbit)
        zenith = state.r_eci / np.linalg.norm(state.r_eci)

        radius_m = R_EARTH + orbit.altitude_km * 1e3
        period_s = 2.0 * np.pi * np.sqrt(radius_m ** 3 / MU_EARTH)

        best_cos, best_offset_s = -2.0, 0.0
        for k in range(PHASE_SAMPLES):
            offset_s = k * period_s / PHASE_SAMPLES
            los = track.state_at(state.t + offset_s)[0] - state.r_eci
            cos_off = float(los @ zenith) / np.linalg.norm(los)
            if cos_off > best_cos:
                best_cos, best_offset_s = cos_off, offset_s

        advance_deg = 360.0 * best_offset_s / period_s
        return float((orbit.true_anomaly_deg + advance_deg) % 360.0)

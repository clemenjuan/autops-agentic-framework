"""Target Track Mission
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
"""
from __future__ import annotations

import numpy as np
from dataclasses import replace
from typing import List, Dict, Tuple

from src.environment.orbital.propagator import create_orbit_track

from src.mission.adcs_mission_base import Mission, MissionState, MissionEvents
from src.environment.orbital.adcs.configs import TargetTrackConfig, OrbitConfig
from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.adcs.control import Setpoint
from src.environment.orbital.adcs.dynamics import dcm_eci_to_body
from src.environment.orbital.adcs.constants import R_EARTH, MU_EARTH

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


class Target:
    def __init__(self, orbit:OrbitConfig):
        self.orbit = orbit
        self.orbit_track = create_orbit_track(orbit)

    def r_eci(self, t):
        return self.orbit_track.state_at(t)[0]

    def v_eci(self,t):
        return self.orbit_track.state_at(t)[1]

class TargetTrackMission(Mission):
    def __init__(self, cfg, sat_orbit):
        super().__init__(cfg)
        self.cfg:TargetTrackConfig = cfg
        self.sat_orbit = sat_orbit

    def reset(self, sat_state, rng) -> MissionState:
        targets = self.generateTargets(self.sat_orbit, rng, sat_state)
        visible = self.check_visibility(sat_state, targets)
        setpoint, search = self.selection(visible, sat_state)

        return MissionState(
            setpoint = setpoint,
            target_idx=0,
            hold_timer=0.0,
            extra=(targets, visible, search)
        )

    def update(self, ms, state, env_data, pointing_error_deg, dt) -> Tuple[MissionState, MissionEvents]:
       pass

    def info(self, ms) -> Dict[str, any]:
        return super().info(ms)

    def check_visibility(self, sat_state:SatState, targets:List[Target]) -> np.ndarray:
        visible = []
        for target in targets:
            contained = self._check_cone(sat_state, target)
            if contained:
                visible.append(target)
        return visible


    def selection(self, visible, sat_state: SatState) -> Tuple[Setpoint, bool]:
        '''For now we slect just the first target in the list.
        Later one would have tom implemnt an actual algorithm to decide the target or let
         the agent decide which one to take next '''
        if len(visible) == 0:
            search = True
            setpoint = self._search_setpoint(sat_state)
        else:
            search = False
            setpoint = self._generate_setpoint(sat_state, visible[0])
        return setpoint, search

    def _search_setpoint(self, sat_state: SatState) -> Setpoint:
        """Point the camera radially outward, to wait for a target to cross.

        Zenith is not a fixed direction: it sweeps through a full turn once per
        orbit, so this cannot be a constant attitude. The setpoint rate is the
        orbital angular velocity, without which the controller would lag the
        sweep by a fixed offset it can never null out.
        """
        zenith_eci = sat_state.r_eci / np.linalg.norm(sat_state.r_eci)
        return self._pointing_setpoint(sat_state, zenith_eci)

    def _pointing_setpoint(
        self, sat_state: SatState, direction_eci: np.ndarray
    ) -> Setpoint:
        """Setpoint putting the boresight on ``direction_eci``.

        Aiming the boresight leaves roll about it free, and the reward reads the
        full attitude error, so the roll is pinned by aligning
        ``cfg.reference_body`` with the orbit normal -- an orbit-frame-like
        attitude that varies smoothly rather than an arbitrary constant.

        Args:
            sat_state: Current state; supplies position and velocity for the
                orbit normal.
            direction_eci: Unit vector the boresight should point along, ECI.

        The target rate is left at zero deliberately, and unlike
        SlewSequenceMission that is not because the target holds still. These
        targets move! Handing the rate over as feedforward
        would be solving part of the control problem in the environment. The
        agent is asked to work it out instead, so the setpoint states where to
        point and nothing about how to follow it.

        Note this leaves the observation non-Markov for tracking: see the
        pointing-rate note in the module docstring.

        Args:
            sat_state: Current state; supplies position and velocity for the
                orbit normal.
            direction_eci: Unit vector the boresight should point along, ECI.

        Returns:
            The setpoint attitude, at zero target rate.
        """
        orbit_normal_eci = np.cross(sat_state.r_eci, sat_state.v_eci)
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

    def generateTargets(
        self, sat_orbit: OrbitConfig, rng: np.random.Generator, sat_state: SatState
    ) -> List[Target]:
        """Draw this episode's targets.

        Each target is a complete random orbit on its own. When
        PHASE_TARGETS_INTO_VIEW is set, one extra step then rewrites its true
        anomaly so it opens inside the field of view; dropping that step is all
        it takes to go back to freely phased targets.

        Args:
            sat_orbit: The satellite's orbit; targets are generated relative to it.
            rng: The episode's seeded generator, the sole source of randomness.
            sat_state: Start-of-episode state. Only read when phasing, to supply
                the position the targets are placed against.

        Returns:
            ``cfg.num_targets`` targets, each on its own propagated orbit.
        """
        targets = []
        for _ in range(self.cfg.num_targets):
            orbit = self._generate_random_orbit(sat_orbit, rng)

            if PHASE_TARGETS_INTO_VIEW:
                orbit = replace(
                    orbit, true_anomaly_deg=self._phase_into_view(orbit, sat_state)
                )

            targets.append(Target(orbit))

        return targets

    def _generate_random_orbit(
        self, sat_orbit: OrbitConfig, rng: np.random.Generator
    ) -> OrbitConfig:
        """One target orbit, offset from the satellite's.

        Complete on its own: the true anomaly is drawn across the whole orbit, so
        where the target starts is left to chance. ``generateTargets`` overrides
        that only while PHASE_TARGETS_INTO_VIEW is set.
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

    def _phase_into_view(self, orbit: OrbitConfig, sat_state: SatState) -> float:
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
            sat_state: Start-of-episode state; supplies time and position.

        Returns:
            True anomaly in degrees, wrapped to [0, 360).
        """
        track = create_orbit_track(orbit)
        zenith = sat_state.r_eci / np.linalg.norm(sat_state.r_eci)

        radius_m = R_EARTH + orbit.altitude_km * 1e3
        period_s = 2.0 * np.pi * np.sqrt(radius_m ** 3 / MU_EARTH)

        best_cos, best_offset_s = -2.0, 0.0
        for k in range(PHASE_SAMPLES):
            offset_s = k * period_s / PHASE_SAMPLES
            los = track.state_at(sat_state.t + offset_s)[0] - sat_state.r_eci
            cos_off = float(los @ zenith) / np.linalg.norm(los)
            if cos_off > best_cos:
                best_cos, best_offset_s = cos_off, offset_s

        advance_deg = 360.0 * best_offset_s / period_s
        return float((orbit.true_anomaly_deg + advance_deg) % 360.0)

    def _check_cone(self, sat_state:SatState, target:Target) -> bool:
        t = sat_state.t

        # Determine the direction the body is in body coordinates 
        los_eci = target.r_eci(t) - sat_state.r_eci
        range_m = np.linalg.norm(los_eci)
        los_body = dcm_eci_to_body(sat_state.q_eci_body) @ (los_eci / range_m)

        #Check offset angle 
        cos_off = float(los_body @ self.cfg.boresight_body) # Works since both are normed vectors 

        if self._check_line_of_sight(sat_state.r_eci, los_eci):
            return cos_off >= np.cos(self.cfg.fov_half_angle)

        return False


    def _check_line_of_sight(self, r_sat, los:np.ndarray) -> bool:
        s_prime = - np.dot(r_sat, los) / np.dot(los,los)
        s_prime = np.clip(s_prime, 0, 1)

        distance = np.linalg.norm(r_sat + s_prime * los)
        return distance >= R_EARTH

    def _generate_setpoint(self, sat_state, target:Target) -> Setpoint:
        '''
        Calculates the actual quaternion setpoint the satellite targets based 
        on the satellites and targets relative position to one another
        '''
        los_eci = target.r_eci(sat_state.t) - sat_state.r_eci
        los_hat = los_eci / np.linalg.norm(los_eci)
        return self._pointing_setpoint(sat_state, los_hat)

    @staticmethod
    def _visible(ms: MissionState) -> Tuple[Setpoint, ...]:
        """Unpack the target setpoints out of ``ms.extra``.

        Keeps the ``extra = (targets,)`` convention in one place rather than
        spreading ``ms.extra[0]`` through the class.
            """
        return ms.extra[1]

    @staticmethod
    def _targets(ms: MissionState) -> Tuple[Setpoint, ...]:
        """Unpack the target setpoints out of ``ms.extra``.
    
        Keeps the ``extra = (targets,)`` convention in one place rather than
        spreading ``ms.extra[0]`` through the class.
            """
        return ms.extra[0]

    @staticmethod
    def _search(ms: MissionState) -> Tuple[Setpoint, ...]:
        """Unpack the target setpoints out of ``ms.extra``.

        Keeps the ``extra = (targets,)`` convention in one place rather than
        spreading ``ms.extra[0]`` through the class.
        """
        return ms.extra[2]
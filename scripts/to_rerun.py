
import rerun as rr
import rerun.blueprint as rrb
import argparse
import sys
import numpy as np

from typing import Optional, Sequence, Dict, Callable, List, Tuple
from pathlib import Path
from src.environment.orbital.adcs.constants import R_EARTH, MU_EARTH
from src.environment.orbital.adcs.eventsat import satellite
from src.environment.orbital.adcs.dynamics import dcm_eci_to_body


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.episode_io import Recording

M_TO_KM = 1e-3

# The orbit view's marker is not to scale. In order to make it visible the
# marker is sized as a fraction of Earth's radius and only its proportions are real
MARKER_FRACTION = 0.1

# Same reasoning for the target markers, which are points rather than boxes and
# so carry a radius. Points3D defaults to a radius in scene units, and the scene
# here is measured in kilometres across a ~13000 km span, so the default is
# sub-pixel.
TARGET_MARKER_FRACTION = 0.025

ARROW_SCALE = 1.3
DEFAULT_BORESIGHT = np.array([0.0, 0.0, -1.0])

# The body triad is drawn shorter than the boresight on purpose. EventSat's
# camera looks along +x, so at equal length the boresight arrow would sit
# exactly inside the x axis and one of the two would be invisible.
AXES_SCALE = 0.8

AXES_LABELS = ["x", "y", "z"]

# The cone is drawn as a wireframe -- a closed rim plus a few lines back to the
# apex -- rather than a filled Mesh3D. The point of drawing it is to watch
# targets pass into it, and a solid surface hides exactly that.
CONE_SEGMENTS = 64
CONE_GENERATORS = 8

# How far to draw the cone when the recording carries nothing to size it from.
# A cone is unbounded, so its length is a presentation choice either way; this
# is the fallback for when there are no targets to reach towards.
CONE_RANGE_FRACTION = 0.25

############################################################################
# Palette
############################################################################
# Every colour in the viewer lives here, and nothing below names a literal.
#
# The grouping is the content, not the tidiness: entities sharing a hue are
# meant to read as one thing, so a change here is a change of meaning rather
# than of taste. Four families:
#
#   spacecraft  orange   the satellite itself, and the path it actually flew
#   camera      cyan     where it is actually looking
#   command     magenta  where it was told to look
#   reference   grey     scaffolding -- the triad, the rings, the limit lines
#
# Camera and command are deliberately far apart on the wheel: watching those
# two converge is the entire job of the attitude view, and two arrows a shade
# apart would make a small pointing error look like a rendering artefact.
#
# Reference is dim across the board. It is there to be looked past, not at,
# and it is what keeps a four-hue scene from reading as noise.

# Spacecraft. The flown track is the same hue darkened -- it is the satellite's
# own history, and the orbit ring it sits on is not.
SAT_COLOR   = [0xFF, 0xA5, 0x00]            # amber      -- box and orbit marker
TRACK_COLOR = [0xC0, 0x7A, 0x1E]            # dark amber -- the arc actually flown

# Camera. The cone is the arrow's hue carried down and made translucent, so it
# reads as belonging to that arrow instead of competing with it.
BORESIGHT_COLOR = [0x00, 0xC8, 0xFF]        # cyan               -- where it looks
CONE_COLOR      = [0x00, 0xC8, 0xFF, 0x55]  # cyan, ~33% alpha   -- the field of view

# Command. Shared with the tracked target below, on purpose: the arrow and the
# thing it points at are one fact, and giving them one hue says so.
COMMAND_COLOR = [0xFF, 0x4F, 0xD8]          # magenta -- where it was told to look

# Reference.
AXES_COLOR         = [0xC0, 0xC0, 0xC0]     # light grey  -- body triad
EARTH_COLOR        = [0x2E, 0x4A, 0x6B]     # dusk blue   -- Earth
ORBIT_COLOR        = [0x5E, 0x6E, 0x86]     # slate grey  -- satellite's orbit ring
TARGET_ORBIT_COLOR = [0x52, 0x6B, 0x57]     # moss grey   -- target orbit rings
LIMIT_COLOR        = [0x8C, 0x8C, 0x8C]     # mid grey    -- plot threshold lines

# Targets, indexed by the `target_status` channel: 0 untracked, 1 cleared,
# 2 currently tracked. The one place in the viewer where hue carries a value
# rather than an identity, which is why it is a ramp and not a family.
TARGET_PALETTE = np.array([
    [0x78, 0x78, 0x78],                     # grey    -- 0, not yet tracked
    [0x3C, 0xB0, 0x50],                     # green   -- 1, cleared
    COMMAND_COLOR,                          # magenta -- 2, being tracked now
], dtype=np.uint8)

############################################################################
# Helpers
############################################################################
def _timelines(rec:Recording) -> List[rr.TimeColumn]:
    indexes=[
        rr.TimeColumn("step", sequence=rec.step),
        rr.TimeColumn("t", duration=rec.t_rel)
    ]
    return indexes


def _rerun_quat(q:np.ndarray) -> np.ndarray:
    """A recorded attitude column, (T, 4), in rerun's component order.

    One function for it because two views turn geometry with this channel --
    the orbit marker and the attitude closeup -- and a second, uncommented
    copy of the reorder is exactly how the two would come to disagree. Getting
    it wrong raises nothing; it renders a plausible, wrong attitude. See
    `log_attitude` for why the reorder is the whole conversion.
    """
    return np.roll(q, -1, axis=1)


############################################################################
# Blueprint
############################################################################
# Display names for the plot groups. Keys are `PLOT_GROUPS` entries, which are
# also the `--plots` choices and the tail of each view's origin.
_VIEW_NAMES = {
    "pointing":     "Pointing",
    "rates":        "Body rates",
    "wheels":       "Wheel speeds",
    "wheel_torque": "Wheel torque",
    "mtq":          "Magnetorquers",
    "reward":       "Reward",
    "policy":       "Policy I/O",
    "misc":         "Other",
}

# What `blueprint_default` shows when `--plots` says nothing: did it point, did
# the wheels run out, what was it paid. Everything else is logged and one click
# away in the viewer's entity tree.
DEFAULT_PLOTS = ("pointing", "wheels", "reward/dense")


def _plot_views(groups:Sequence[str]) -> List[rrb.TimeSeriesView]:
    """One time-series view per plot group.

    `origin` is the whole selection mechanism: a view rooted at
    `plots/wheels` shows every series logged below it, which is why
    `log_scalars` puts the group in the entity path. For a finer cut, pass
    `contents` instead -- it takes globs, with a leading '-' to exclude, e.g.
    contents=["plots/pointing/**", "- plots/pointing/target_idx"].
    """
    return [
        rrb.TimeSeriesView(origin=f"plots/{g}", name=_VIEW_NAMES.get(g, g))
        for g in groups
    ]


def blueprint_default(plots:Optional[Sequence[str]] = None) -> rrb.Blueprint:
    """Three areas: orbit and attitude on the left, the chosen plots on the right."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(origin="world", name="Orbit"),
                rrb.Spatial3DView(origin="body", name="Attitude"),
            ),
            rrb.Vertical(*_plot_views(plots or DEFAULT_PLOTS)),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def blueprint_attitude(plots:Optional[Sequence[str]] = None) -> rrb.Blueprint:
    """No orbit view: the closeup, with the plots that explain what it is doing.

    Reward is split here rather than shown whole, because the event bonuses
    (500, 100) flatten the dense terms (1.0, 0.1) onto the same axis.
    """
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="body", name="Attitude"),
            rrb.Vertical(*_plot_views(plots or ("pointing", "rates", "wheel_torque", "mtq"))),
            column_shares=[2, 4],
        ),
        collapse_panels=True,
    )


def blueprint_reward(plots:Optional[Sequence[str]] = None) -> rrb.Blueprint:
    """For reading the policy rather than the spacecraft."""
    if plots:
        return rrb.Blueprint(rrb.Vertical(*_plot_views(plots)), collapse_panels=True)
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.TimeSeriesView(origin="plots/reward/dense", name="Reward (dense)"),
            rrb.TimeSeriesView(origin="plots/reward/event", name="Reward (events)"),
            rrb.TimeSeriesView(origin="plots/reward/cumulative", name="Return"),
            rrb.TimeSeriesView(origin="plots/policy", name="Policy I/O"),
        ),
        collapse_panels=True,
    )


BLUEPRINTS: Dict[str, Callable[..., rrb.Blueprint]] ={
    "default":  blueprint_default,
    "attitude": blueprint_attitude,
    "reward":   blueprint_reward,
}

############################################################################
# Orbit
############################################################################
def log_earth() -> None:
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    half_size= [R_EARTH * M_TO_KM] * 3 
    rr.log(
        "world/earth",
        rr.Ellipsoids3D(half_sizes=half_size, colors=EARTH_COLOR),
        static=True
    )

def _orbit_ring(r0: np.ndarray, v0:np.ndarray, n: int = 360) ->np.ndarray:
    """The array needed to build the orbital ring for any object

    Args:
        r0: Position at t0, ECI [m], shape (3,).
        v0: Velocity at t0, ECI [m/s], shape (3,).
        n: Points around the ring.

    Returns:
        Points around the closed orbit, ECI [m], shape (n, 3).

    """

    # Orbit-plane basis. The normal comes first because the other two are only
    # defined relative to it: x towards the satellite at t0, y completing the
    # right-handed set and so pointing along the motion.
    h = np.cross(r0, v0)
    n_hat = h / np.linalg.norm(h)
    x_hat = r0 / np.linalg.norm(r0)
    y_hat = np.cross(n_hat, x_hat)
    
    # vis-viva. Using |r0| instead would be within metres here, but only
    # because the orbit is circular -- this stays right if it ever is not.
    a = 1.0 / (2.0 / np.linalg.norm(r0) - v0 @ v0 / MU_EARTH)
    
    # endpoint=True (the default) makes theta[-1] coincide with theta[0],
    # which is what closes the strip instead of leaving a seam.
    theta = np.linspace(0, 2 * np.pi, n)
    ring = a * (np.cos(theta)[:, None] * x_hat + np.sin(theta)[:, None] * y_hat)

    return ring

def log_orbit_path(rec:Recording) -> None:
    """The whole orbit the satellite is on, as one closed ring.

    Derived rather than recorded. An ADCS episode is seconds long and sweeps
    well under a degree, so `r_eci` on its own draws a short tick beside the
    Earth and reads as nothing. The state vector at t0 pins the orbit exactly,
    and at the eccentricity these episodes fly (~1e-5) a circle in the orbit
    plane is indistinguishable from the ellipse.

    Taken from the recording, not from `eventsat.orbit`: the launch lottery
    randomises RAAN/ArgP/TA per episode, so the nominal config would draw a
    different ring than the one actually flown.
    """
    r0 = rec["r_eci"][0]
    v0 = rec["v_eci"][0]

    rr.log(
        "world/orbit",
        rr.LineStrips3D(_orbit_ring(r0,v0) * M_TO_KM, colors=ORBIT_COLOR),
        static=True
    )

def log_target_orbits(rec:Recording) -> None:
    """The orbits the targets are on as closed rings"""

    target_positions = rec["target_positions"] #(T, num_targets, 3)
    target_velocities = rec["target_velocities"] #(T, num_targets, 3)
    num_targets = target_positions.shape[1]

    for i in range(num_targets):
        rr.log(
            f"world/targets/{i}/orbit",
            rr.LineStrips3D(
                _orbit_ring(r0=target_positions[0][i], v0=target_velocities[0][i]) * M_TO_KM,
                colors=TARGET_ORBIT_COLOR),
            static = True
            )

def log_orbit_track(rec:Recording) -> None:
    """Where the satellite actually was, over the episode.

    A separate entity from `world/orbit` because it answers a different
    question: a ~165 km arc, invisible at orbit scale on its own, but the
    thing worth looking at once the satellite marker moves along it.

    Coloured in the spacecraft family rather than with the ring, for the same
    reason: this arc is the satellite's own history, the ring is geometry it
    happens to sit on.
    """
    rr.log(
        "world/track",
        rr.LineStrips3D(rec["r_eci"] * M_TO_KM, colors=TRACK_COLOR),
        static=True
    )

def log_orbit_motion(rec:Recording) -> None:
    """Carry the satellite frame along the orbit and turn it with the attitude.

    Both halves on one transform because they are one pose: `world/satellite`
    is the spacecraft's frame expressed in ECI. Anything hung below it -- the
    marker, and later the boresight and the visibility cone -- inherits
    position and orientation together and needs to know nothing about time.

    The attitude is the closeup's channel under the closeup's conversion, via
    `_rerun_quat`. What differs is only the scale it is drawn at: the marker is
    a fraction of Earth's radius rather than 0.37 m, so the rotation is
    actually visible from orbit distance.
    """
    rr.send_columns(
        "world/satellite",
        indexes= _timelines(rec),
        columns=rr.Transform3D.columns(
            translation=rec["r_eci"] * M_TO_KM,
            quaternion=_rerun_quat(rec["q_eci_body"]),
        ),
    )

def log_satellite_marker() -> None:
    """A stand-in box for the satellite, as a child of the orbit transform.

    Child of `world/satellite`, so it inherits that transform and needs to know
    nothing about time -- the same reason the visibility cone and the pointing
    arrows can hang here later and move for free.

    Deliberately not to scale (see MARKER_FRACTION), but the proportions are
    EventSat's own, so the box still reads as an orientation once a rotation
    joins the translation on the parent transform.
    """
    dims = satellite.dimensions
    marker = dims / dims.max() * MARKER_FRACTION * R_EARTH * M_TO_KM
    rr.log(
        "world/satellite/marker",
        rr.Boxes3D(sizes=marker, colors=SAT_COLOR),
        static=True
    )

def _fov_range_km(rec:Recording) -> float:
    """How far down the boresight to draw, in the orbit view's kilometres.

    The furthest a target ever was, so the cone reaches whatever it is meant to
    be catching rather than stopping short of it or swamping the Earth. That is
    a slant range while the cone is sized along its axis, so the cone runs a
    little past the targets -- which is the harmless direction to be wrong in,
    since it keeps them enclosed.

    Falls back to a fraction of Earth's radius when the recording has no
    targets, e.g. a slew episode.
    """
    if "target_positions" in rec:
        los = rec["target_positions"] - rec["r_eci"][:, None, :]
        return float(np.linalg.norm(los, axis=2).max()) * M_TO_KM
    return CONE_RANGE_FRACTION * R_EARTH * M_TO_KM


def _cone_wireframe(axis:np.ndarray, half_angle:float, length:float) -> List[np.ndarray]:
    """Line strips for a cone with its apex at the origin, in the axis's frame.

    Args:
        axis: Unit vector the cone opens along, shape (3,).
        half_angle: Angle from the axis to the surface [rad]. Must be under
            pi/2 -- at or past it the cone turns inside out and `tan` runs
            away, but a camera that sees behind itself is not a camera.
        length: Distance from apex to rim, along the axis.

    Returns:
        The rim first, then the generator lines, each an (n, 3) strip.
    """
    axis = axis / np.linalg.norm(axis)

    # Any two vectors perpendicular to the axis will do -- the cone is
    # symmetric about it, so where the rim's points start is arbitrary. The
    # seed is only chosen to not be parallel to the axis, which would make the
    # first cross product zero.
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    radius = length * np.tan(half_angle)

    # endpoint=True (the default) puts the last point on the first, which is
    # what closes the rim instead of leaving a seam -- same as `_orbit_ring`.
    theta = np.linspace(0, 2 * np.pi, CONE_SEGMENTS)
    rim = length * axis + radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)

    strips = [rim]
    stride = max(1, (CONE_SEGMENTS - 1) // CONE_GENERATORS)
    strips += [np.array([np.zeros(3), rim[i]]) for i in range(0, CONE_SEGMENTS - 1, stride)]
    return strips


def log_orbit_boresight(rec:Recording) -> None:
    """The camera axis in the orbit view, at orbit scale.

    Separate from `log_boresight` for the reason `log_satellite_marker` is
    separate from `log_satellite_geometry`: same vector, but one view measures
    in metres and the other in thousands of kilometres, so a single entity
    cannot serve both. Child of `world/satellite`, so the orbit transform gives
    it position and attitude and it needs no time column of its own.
    """
    rr.log(
        "world/satellite/boresight",
        rr.Arrows3D(
            vectors=[_boresight(rec) * _fov_range_km(rec)],
            colors=BORESIGHT_COLOR,
            labels=["boresight"],
            show_labels=True,
        ),
        static=True,
    )


def log_fov_cone(rec:Recording) -> None:
    """The camera's field of view, as a wireframe cone about the boresight.

    Static geometry in the body frame, hung under `world/satellite`: the cone
    does not change shape, only where it points, and the orbit transform
    already carries that. So this is one log call for the whole episode rather
    than a column per step.

    This is the same cone `TargetTrackMission._check_cone` tests against, which
    is what makes it worth drawing -- a target inside it on screen is a target
    the mission counted as visible. The occultation half of that test is not
    drawn, so a target behind the Earth still appears enclosed.

    Only target-track missions define a field of view, so this is a no-op
    elsewhere.
    """
    half_angle = rec.meta["mission"].get("fov_half_angle")
    if half_angle is None:
        return

    strips = _cone_wireframe(_boresight(rec), float(half_angle), _fov_range_km(rec))
    rr.log(
        "world/satellite/fov",
        rr.LineStrips3D(strips, colors=CONE_COLOR),
        static=True,
    )


def log_targets(rec:Recording)->None:
    pos = rec["target_positions"] * M_TO_KM #(T, num_targets, 3)
    num_targets = pos.shape[1]
    colors = TARGET_PALETTE[rec["target_status"]] # (T, num_targets, 3)

    # One radius per point rather than one for the entity: every column handed
    # to `partition` has to share the flattened length, so a scalar would not
    # line up with `positions`.
    radii = np.full(pos.shape[0] * num_targets, TARGET_MARKER_FRACTION * R_EARTH * M_TO_KM)

    rr.send_columns(
        "world/targets/markers",
        indexes=_timelines(rec),
        columns=rr.Points3D.columns(
            positions=pos.reshape(-1,3),
            colors=colors.reshape(-1,3),
            radii=radii,
    ).partition([num_targets] * pos.shape[0])
    )


############################################################################
# Attitude
############################################################################

def log_satellite_geometry() -> None:
    """The satellite at true size, for the attitude closeup.

    Is independend of the marker used in the orbital figure and thus not inherits 
    it's motion

    Sized from `satellite.dimensions` -- note the config flags the axis order as
    unconfirmed against CMO p.16, so the box is the right size but possibly the
    wrong way round.
    """
    rr.log("body", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "body/satellite/box",
        rr.Boxes3D(sizes=satellite.dimensions, colors=SAT_COLOR),
        static=True
    )

def log_body_frame() -> None:
    """The body triad, so the closeup has an orientation reference.

    Child of `body/satellite`, so it turns with the box and needs no time
    column -- the same free ride the boresight arrow takes.

    Worth reading against the caveat in `log_satellite_geometry`: the box's
    axis order is unconfirmed against CMO p.16. These arrows are the frame
    `q_eci_body` and `omega_body` are written in regardless, so wherever the
    two disagree it is the box that is wrong, not the triad.
    """
    length = satellite.dimensions.max() * AXES_SCALE
    rr.log(
        "body/satellite/frame",
        rr.Arrows3D(
            vectors=np.eye(3) * length,
            colors=AXES_COLOR,
            labels=AXES_LABELS,
            show_labels=True,
        ),
        static=True,
    )

def _boresight(rec:Recording) -> np.ndarray:
    boresight = rec.meta["mission"].get("boresight_body")
    if boresight is None:
        boresight = DEFAULT_BORESIGHT
    return np.asarray(boresight, dtype=float)

def log_boresight(rec:Recording) -> None:
    """Where the camera looks, in the body frame -- so the transform turns it."""
    boresight = _boresight(rec)
    length = satellite.dimensions.max() * ARROW_SCALE
    rr.log(
        "body/satellite/boresight",
        rr.Arrows3D(
            vectors=[boresight * length],
            colors=BORESIGHT_COLOR,
            labels=["boresight"],
            show_labels=True,
        ),
        static = True
        )

def log_target_direction(rec:Recording) -> None:
    """Where it was told to look. Logged in ECI, hence on the root, not under
    the satellite transform: the setpoint is not a body-frame quantity.

    The label and the colour are a separate static log rather than fields on
    the column batch: they describe the arrow, not any one step, so sending
    them once beats repeating them for every row. Components of one archetype
    can be split across calls this way, which is what `from_fields` is for --
    it writes only the fields named and leaves `vectors` to the column send
    below.
    """
    boresight = _boresight(rec)
    length = satellite.dimensions.max() * ARROW_SCALE
    vectors = np.array([dcm_eci_to_body(q).T @ boresight for q in rec["target_q_eci_body"]]) * length
    rr.log(
        "body/target_direction",
        rr.Arrows3D.from_fields(
            colors=COMMAND_COLOR,
            labels=["target"],
            show_labels=True,
        ),
        static=True,
    )
    rr.send_columns(
        "body/target_direction",
        indexes=_timelines(rec),
        columns=rr.Arrows3D.columns(vectors=vectors)
    )

def log_attitude(rec:Recording) -> None:
    """Turn the closeup box with the recorded attitude.

    Rotation only, no translation: that is what keeps the box at the origin
    while it turns, and what makes this view about attitude alone.

    On the quaternion, which looks wrong and is not
    -----------------------------------------------
    Two things differ between the archive and rerun, and only one needs fixing.

    Component order does: the channel is scalar-first (w, x, y, z), rerun is
    scalar-last (x, y, z, w). Hence the roll.

    Direction looks like it does, and does not. `Transform3D` asks "the geometry
    below this node is written in the node's own frame -- where does it sit in
    the parent's?", i.e. body -> ECI, while the channel is named ECI -> body.
    But that name belongs to `dcm_eci_to_body(q)`, not to `q`: under this
    codebase's Hamilton convention the textbook rotation matrix R(q) already
    *is* body -> ECI, which is why `dynamics.dcm_eci_to_body` returns R(q).T to
    get the direction it is named after (dynamics.py:53-68). Rerun applies that
    same textbook R(q). So the two transposes that would have cancelled never
    happen, and the reorder is the whole conversion.

    Checked rather than assumed: Rotation.from_quat(np.roll(q, -1)) reproduces
    dcm_eci_to_body(q).T exactly on this data. Worth re-checking if the
    kinematics convention in `dynamics` ever changes, because getting this
    backwards raises no error -- it renders a plausible, wrong attitude.

    """
    rr.send_columns(
        "body/satellite",
        indexes=_timelines(rec),
        columns=rr.Transform3D.columns(
            quaternion=_rerun_quat(rec["q_eci_body"]),
        ),
    )

############################################################################
# Scalar Plots
############################################################################

# Channels that are not plotted. The spatial ones are the 3D views' business;
# `t` is the timeline itself, so plotting it draws a diagonal and nothing else.
_NOT_PLOTTED = ("t", "r_eci", "v_eci", "q_eci_body", "target_q_eci_body")

# Channel -> group. A channel missing here lands in "misc" rather than being
# dropped, so a channel added to `adcs_record` still shows up somewhere without
# this file changing; promote it to a real group once it has earned one.
_GROUP_OF = {
    "pointing_error_deg": "pointing",
    "hold_timer":         "pointing",
    "body_rate":          "rates",
    "omega_body":         "rates",
    "wheel_speeds":       "wheels",
    "wheel_torque":       "wheel_torque",
    "mtq_dipole":         "mtq",
    "action":             "policy",
    "obs":                "policy",
}

# Group names are what `--plots` accepts and what a blueprint points a
# TimeSeriesView's `origin` at. Static, unlike group membership.
PLOT_GROUPS = ("pointing", "rates", "wheels", "wheel_torque", "mtq",
               "reward_dense", "reward_total", "policy", "misc")


def _component_names(rec:Recording, channel:str, width:int) -> List[str]:
    """Labels for one column of a (N, k) channel.

    Fanning a multi-component channel into one entity path per component is
    what makes each line separately toggleable and separately coloured; these
    are the leaf names those paths get. Unknown channels fall back to indices,
    which is ugly but never wrong.
    """
    n = rec.n_wheels
    if channel == "omega_body":
        return ["x", "y", "z"]
    if channel in ("wheel_speeds", "wheel_torque"):
        return [f"wheel_{i}" for i in range(width)]
    if channel == "mtq_dipole":
        return [f"mtq_{ax}" for ax in "xyz"[:width]]
    if channel == "action":
        # The one channel still carrying both actuators, because it is the
        # policy's own output vector: wheel torques first, dipoles after, all
        # normalised to +-1 so they share an axis honestly.
        return [f"wheel_{i}" for i in range(n)] + [f"mtq_{ax}" for ax in "xyz"[: width - n]]
    if channel == "obs":
        # concat([q_error(4), omega_body(3), wheel_speeds(4)]), see
        # adcs_gymnasium_wrapper.py:214. Normalised, so these are not the same
        # numbers as the `rates` and `wheels` groups -- that is the point of
        # keeping them: a scaling bug shows up here and nowhere else.
        return (["q_err_w", "q_err_x", "q_err_y", "q_err_z"]
                + ["omega_x", "omega_y", "omega_z"]
                + [f"wheel_{i}" for i in range(width - 7)])
    return [str(i) for i in range(width)]


def _send_scalar(path:str, series:np.ndarray, indexes:List[rr.TimeColumn]) -> None:
    """One scalar series, as a column rather than a row at a time."""
    rr.send_columns(path, indexes=indexes, columns=rr.Scalars.columns(scalars=series))


def log_threshold(path:str, value:float, name:str, rec:Recording,
                  indexes:List[rr.TimeColumn]) -> None:
    """One horizontal reference line, as a constant series.

    Rerun's time-series views have no horizontal-line primitive, so a threshold
    is an ordinary series that happens not to vary. Styled thin and grey by the
    static `SeriesLines` so it reads as a reference rather than as data -- the
    styling is static because it describes the series, not any one sample.
    """
    rr.log(path, rr.SeriesLines(colors=LIMIT_COLOR, widths=1.0, names=name), static=True)
    _send_scalar(path, np.full(len(rec), float(value)), indexes)


def log_thresholds(rec:Recording) -> None:
    """The limits the episode was actually flown against.

    Logged under each group's own subtree, so a view rooted at `plots/wheels`
    picks up its limit lines without naming them -- the same `origin` rule that
    selects the data.

    Limits absent from older archives come back as None from `Recording` rather
    than raising, so each block is guarded: a file recorded before `build_meta`
    grew a field still plots, just without that line.
    """
    indexes = _timelines(rec)

    log_threshold("plots/pointing/limits/tolerance", rec.tolerance_deg, "tolerance", rec, indexes)
    log_threshold("plots/pointing/limits/hold_time", rec.hold_time, "hold time", rec, indexes)
    log_threshold("plots/rates/limits/slew_penalty", rec.body_rate_thresh, "slew penalty", rec, indexes)

    for sign in (+1, -1):
        log_threshold(f"plots/wheels/limits/max_speed_{'pos' if sign > 0 else 'neg'}",
                      sign * rec.wheel_max_speed, "wheel limit", rec, indexes)
        # Normalised commands, so the rail is at +-1 by construction, not config.
        log_threshold(f"plots/policy/limits/action_{'pos' if sign > 0 else 'neg'}",
                      sign * 1.0, "action rail", rec, indexes)

    # Per-actuator, and per-axis within that: nothing requires a cluster to be
    # built from identical wheels, so these are arrays rather than scalars.
    if rec.wheel_max_torque is not None:
        for sign in (+1, -1):
            log_threshold(f"plots/wheel_torque/limits/max_{'pos' if sign > 0 else 'neg'}",
                          sign * rec.wheel_max_torque.max(), "wheel torque limit", rec, indexes)
    if rec.mtq_max_dipole is not None:
        for sign in (+1, -1):
            log_threshold(f"plots/mtq/limits/max_{'pos' if sign > 0 else 'neg'}",
                          sign * rec.mtq_max_dipole.max(), "MTQ limit", rec, indexes)


def log_scalars(rec:Recording) -> None:
    """Every numeric channel in the archive, as a time series.

    Swept from `rec.arrays`. A channel added to `adcs_record` plots itself, so 
    everything is logged. Which groups are shown is the blueprint's decision,
    so toggling a plot on is a click rather than another run.
    """
    indexes = _timelines(rec)

    # Reward first, so the generic sweep below can skip what it has claimed.
    _send_scalar("plots/reward/total", rec["reward"], indexes)
    _send_scalar("plots/reward/cumulative", rec.cum_reward, indexes)
    for term, series in rec.dense_terms.items():
        _send_scalar(f"plots/reward/dense/{term}", series, indexes)
    for term, series in rec.event_terms.items():
        _send_scalar(f"plots/reward/event/{term}", series, indexes)

    for name in sorted(rec.arrays):
        if name in _NOT_PLOTTED or name == "reward" or name.startswith("reward_"):
            continue
        data = rec[name]
        # `phase` is <U16; nothing else in the archive is non-numeric. Bools
        # (terminated, truncated) go through as 0/1, which plots fine.
        if data.dtype.kind not in "fiub":
            continue

        series = data.astype(np.float64)
        group = _GROUP_OF.get(name, "misc")
        if series.ndim == 1:
            _send_scalar(f"plots/{group}/{name}", series, indexes)
        else:
            for i, label in enumerate(_component_names(rec, name, series.shape[1])):
                _send_scalar(f"plots/{group}/{name}/{label}", series[:, i], indexes)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        prog="to_rerun",
        description="Show all recorded data in rerun viewer",
    )

    # Positional: there is no sensible default recording, while every other
    # argument has one.
    parser.add_argument(
        "record",
        type=Path,
        help="episode .npz, as written by adcs_record",
    )

    parser.add_argument(
            "--blueprint",
            type=str,
            choices=list(BLUEPRINTS),
            default="default",
            help="Describes what layout the window should open with",
        )

    # Selects which plot views the layout opens with, not what gets logged --
    # every channel is sent regardless, so turning a group on afterwards is a
    # click in the viewer rather than another run.
    parser.add_argument(
        "--plots",
        type=str,
        default=None,
        metavar=",".join(PLOT_GROUPS[:3]) + ",...",
        help=f"comma-separated plot groups to show, from: {', '.join(PLOT_GROUPS)} "
             "(default: whatever the blueprint chooses)",
    )

    args = parser.parse_args(argv)

    if args.plots is not None:
        args.plots = [g.strip() for g in args.plots.split(",") if g.strip()]
        unknown = [g for g in args.plots if g not in PLOT_GROUPS]
        if unknown:
            # parser.error, not a raise: this is a bad command line, so it earns
            # the usage message and exit code 2 that argparse gives.
            parser.error(
                f"unknown plot group(s) {', '.join(unknown)}; "
                f"choose from {', '.join(PLOT_GROUPS)}"
            )
    return args

MISSION_SCENES: Dict[str, Tuple[Callable[[Recording], None], ...]] = {
    "slew": (),
    "target_track": (log_target_orbits, log_targets, log_fov_cone)
}

def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    #Open the recording
    rec = Recording.load(args.record)

    rr.init("Simulation Analysis", spawn=True)

    # This creates blueprints which has everything prearanged
    # and overwrites how the user left the viewer 
    rr.send_blueprint(BLUEPRINTS[args.blueprint](args.plots))

    ### General (mission unspecific) calls ###
    # All static object created and logged 
    log_earth()
    log_orbit_path(rec=rec)
    log_orbit_track(rec=rec)
    log_satellite_marker()
    log_satellite_geometry()
    log_boresight(rec=rec)
    log_orbit_boresight(rec=rec)
    log_body_frame()
    # log_series_styling()
    # log_ground_targets()


    # Temporal pass
    log_orbit_motion(rec=rec)
    log_attitude(rec=rec)
    log_target_direction(rec=rec)
    log_scalars(rec=rec)
    log_thresholds(rec=rec)
    # log_events()

    ### Mission specific calls ###
    for log_scene in MISSION_SCENES.get(rec.mission_type, ()):
        log_scene(rec)

    return 0



if __name__ == "__main__":
    raise SystemExit(main())

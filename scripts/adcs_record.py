"""Records one ADCS episode to a compressed ``.npz`` file.

The episode is driven by a *pluggable policy* -- any ``policy(obs) -> action``
callable -- and run with a concrete *SEED*, so the same file can be produced
again bit-for-bit. The resulting archive is the input for the Rerun debug
viewer and for matplotlib plots, neither of which live here: this module only
records. That keeps ``rerun-sdk`` and ``matplotlib`` out of the import path, so
capturing an episode never depends on a viewer being installed.

One episode per file. Every array shares the same leading dimension
``N = n_steps + 1``: row 0 is the state returned by ``reset()``, which has no
action and no reward attached, so those channels are filled with NaN there.
A single ``N`` means ``t`` indexes every channel directly, which is what both
Rerun's timeline and matplotlib want.

Schema
------
Channels are not hard-coded. Each one is a :class:`Channel` -- a name, an
extractor, and a dtype -- and :data:`DEFAULT_CHANNELS` is just a list of them.
Recording something new is one entry::

    channels = [*DEFAULT_CHANNELS,
                Channel("gyro_bias", lambda f: f.env.sensor_state.gyro_bias)]
    arrays = record_episode(policy, channels=channels)

The reward channels are generated from ``REWARD_COMPONENTS + EVENT_COMPONENTS``
rather than listed, so a reward term added in ``adcs_rewards`` shows up here
without a change -- the same contract ``EventSatEnv._get_info`` already keeps.

The written file also carries a ``channels`` manifest inside its metadata
(name -> shape, dtype, doc), so a consumer can iterate the archive generically
instead of hard-coding whatever the schema happened to be on the day it was
written.

Reading a file back is :mod:`scripts.episode_io`, not this module:
``Recording.load(path)`` for a plot or the viewer, ``load_episode(path)`` for
the two raw dicts. Both are re-exported here, but importing *this* module to
get at them pulls in the env and with it the Orekit JVM, which a reader has no
use for.

Which channels a file carries depends on the mission it recorded:
:data:`MISSION_CHANNELS` maps a mission type to its set, and
``record_episode`` picks from it unless handed an explicit list. The three
``target_*`` channels below are target_track only; everything else is common.

Defaults, for the EventSat mission config (``step_s = 0.2``, ``max_steps =
2000``):

    t                  (N,)     float64   seconds since epoch
    q_eci_body         (N, 4)   float64   scalar-first, ECI -> body
    omega_body         (N, 3)   float64   rad/s, body frame
    wheel_speeds       (N, 4)   float64   rad/s
    r_eci              (N, 3)   float64   m
    v_eci              (N, 3)   float64   m/s
    target_q_eci_body  (N, 4)   float64   current setpoint
    target_positions   (N, T, 3)float64   m, target_track only
    target_velocities  (N, T, 3)float64   m/s, target_track only
    target_status      (N, T)   int8      0/1/2, target_track only
    target_idx         (N,)     int32
    hold_timer         (N,)     float64   s inside tolerance
    phase              (N,)     <U16
    pointing_error_deg (N,)     float64
    body_rate          (N,)     float64   |omega|, rad/s
    wheel_speed_frac   (N,)     float64   max |w| / w_max
    obs                (N, 11)  float32
    action             (N, 7)   float32   normalised, NaN at row 0
    wheel_torque       (N, 4)   float64   N*m, NaN at row 0
    mtq_dipole         (N, 3)   float64   A*m^2, NaN at row 0
    reward             (N,)     float64   NaN at row 0
    reward_<component> (N,)     float64   6 channels, NaN at row 0
    terminated         (N,)     bool
    truncated          (N,)     bool
    meta               ()       <U        JSON, see `load_episode`

``wheel_torque`` and ``mtq_dipole`` are what was *asked for*: the action vector
scaled by ``max_action``. Not what was delivered -- the torque the wheels
actually produce is clipped against the momentum envelope inside
``actuators.apply_reaction_wheel``, and the two diverge exactly when the wheels
saturate, which is when you care.

They are two channels rather than one because they are two units, N*m and
A*m^2, differing by two orders of magnitude. ``action`` stays packed at (N, 7),
and so does ``max_action`` in the metadata: those are the policy's own
interface, normalised to [-1, 1] throughout, and splitting them would
misdescribe what the policy emits.

Not recorded yet
----------------
Achieved wheel torque, MTQ dipole, disturbance torque, coarse-sun currents and
the FSS/EHS validity flags are locals inside ``simulation.step`` and are not
reachable from out here. To add them, give ``step`` an optional
``trace: dict | None = None`` out-param that it fills from values it has
already computed, forward it in ``EventSatEnv.step`` as ``self.last_trace``,
and add one :class:`Channel` per quantity. Recomputing them in this module
instead would duplicate the physics *and* draw different sensor noise than the
policy actually saw, so that is not the path.

Usage
-----
    uv run python scripts/adcs_record.py --policy checkpoint --seed 42
    uv run python scripts/adcs_record.py --policy checkpoint --mission target_track

The mission decides three things: which channels are recorded, which
per-mission checkpoint ``--policy checkpoint`` loads, and which subdirectory
of ``data/records/`` the file lands in.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, replace, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.environment.orbital.adcs.adcs_gymnasium_wrapper import EventSatEnv
from src.environment.orbital.adcs.adcs_rewards import (
    EVENT_COMPONENTS,
    REWARD_COMPONENTS,
)

# Re-exported so `save_episode`/`load_episode` still resolve here, which is
# where every caller and every docstring in this module expects them. They live
# in episode_io because a reader -- a plot, the Rerun viewer -- must not have to
# import this module to get at them: that import chain reaches `propagator`,
# which boots the Orekit JVM.
#
# Imported as `scripts.episode_io` rather than as a bare `episode_io`: the
# latter resolves only while this file is run as a script, when its own
# directory heads sys.path, and would break the moment anything imports
# `scripts.adcs_record` instead. The sys.path insert above is what makes the
# qualified form work in both cases.
from scripts.episode_io import (  # noqa: F401
    Recording,
    load_episode,
    save_episode,
)
from src.environment.orbital.adcs.configs import AdcsEnvConfig
from src.environment.orbital.adcs.eventsat import(
    env as EVENTSAT_ENV_CONFIG, 
    DEFAULT_MISSION, 
    MISSION_CONFIGS,
    )
from src.mission.registry import MISSION_TYPES

# Repo root is two levels up: scripts -> root. Artifacts belong in the
# git-ignored data/ tree, not next to the source, same as adcs_train_ppo.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Must match src.rl.adcs_train_ppo.CHECKPOINT_DIR. Duplicated rather than
# imported:
# importing it would pull in ray at module import time, and the whole point of
# the pluggable policy is that recording works without ray installed.
CHECKPOINT_ROOT = REPO_ROOT / "data" / "trained_models" / "adcs_ppo"

RECORD_ROOT = REPO_ROOT / "data" / "records"

SEED = 42

# Bumped when the meaning of an existing channel changes, not when one is
# added -- consumers keyed off the `channels` manifest survive additions.
# 2: `command_torque` (N, 7) split into `wheel_torque` (N, 4) [N*m] and
#    `mtq_dipole` (N, 3) [A*m^2] -- one array cannot carry two units, and
#    every consumer was slicing it apart again. v1 archives are not read.
SCHEMA_VERSION = 2

# Every reward term, dense and event, in the order adcs_rewards declares them.
ALL_REWARD_COMPONENTS: Tuple[str, ...] = REWARD_COMPONENTS + EVENT_COMPONENTS


# A policy maps one observation to one action. Deterministic by contract: a
# sampling policy would make the seed meaningless.
Policy = Callable[[np.ndarray], np.ndarray]


@dataclass
class Frame:
    """Everything one recorded row can be derived from.

    Built once per row and handed to every channel's extractor. The reset row
    has no action and no reward, so those three fields are None there; an
    extractor that touches them returns None in turn and the recorder fills
    the sentinel for its dtype.

    Attributes:
        env: The live environment. Read `state`, `mission_state`, `estimator`
            and `_step_count` off it. Note that `mission_state` is mutated in
            place by the mission, so an extractor must copy what it reads.
        obs: Observation returned alongside this row, shape (11,) float32.
        info: The env's info dict for this row.
        action: Normalised action that produced this row, or None at reset.
        reward: Scalar reward for this row, or None at reset.
        reward_info: Per-component reward dict, or None at reset. The
            components are also present in `info` under a `reward/` prefix.
        terminated: Mission completed on this step.
        truncated: Step cap hit on this step.
    """

    env: EventSatEnv
    obs: np.ndarray
    info: Dict[str, Any]
    action: Optional[np.ndarray] = None
    reward: Optional[float] = None
    reward_info: Optional[Dict[str, float]] = None
    terminated: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class Channel:
    """One recorded quantity.

    Attributes:
        name: Key under which the stacked array lands in the NPZ.
        extract: Given a Frame, returns this row's value, or None if the
            quantity is undefined on that row (the reset row, typically).
        dtype: Storage dtype. Drives which sentinel fills a None row, so a
            channel that can be None must use a dtype that has one -- float
            (NaN) or bool/int/str (False/-1/"").
        doc: One line, units included. Goes into the file's channel manifest.
    """

    name: str
    extract: Callable[[Frame], Any]
    dtype: np.dtype = np.dtype(np.float64)
    doc: str = ""


# What a None row becomes, by dtype kind. Floats get NaN so plots break the
# line rather than drawing a fake zero; there is no such value for the others,
# hence the explicit out-of-range choices.
_FILL: Dict[str, Any] = {
    "f": np.nan,
    "b": False,
    "i": -1,
    "u": 0,
    "U": "",
}


def _info(key: str) -> Callable[[Frame], Any]:
    """Extractor reading one key out of the env's info dict."""
    return lambda frame: frame.info[key]


def _reward_component(name: str) -> Callable[[Frame], Any]:
    """Extractor for one reward term, None on the reset row.

    Reads `reward_info` rather than the `reward/<name>` info key: on reset the
    env reports every component as 0.0 to keep its key set stable, and a
    recorded 0.0 there is a lie -- nothing was earned, the term is undefined.
    """
    return lambda frame: (
        None if frame.reward_info is None else frame.reward_info[name]
    )


def _state(attr: str) -> Callable[[Frame], Any]:
    """Extractor for one SatState field, copied."""
    return lambda frame: np.copy(getattr(frame.env.state, attr))


def _target_quat(frame: Frame) -> np.ndarray:
    """Current setpoint quaternion, copied.

    The copy is load-bearing: SlewSequenceMission.update mutates MissionState
    in place, so storing the reference gives every row the final target.
    """
    return np.copy(frame.env.mission_state.setpoint.target_q_eci_body)


def _commanded(frame: Frame) -> Optional[np.ndarray]:
    """Commanded actuator effort: action * env.max_action, still packed.

    The scaling is one operation over the whole action vector, so it happens
    once here; `_wheel_torque` and `_mtq_dipole` slice the result into the two
    quantities it actually holds. Commanded, not achieved -- see the module
    docstring.
    """
    if frame.action is None:
        return None
    return np.asarray(frame.action, dtype=np.float64) * frame.env.max_action


def _wheel_torque(frame: Frame) -> Optional[np.ndarray]:
    """Commanded wheel motor torque [N*m], (n_wheels,)."""
    commanded = _commanded(frame)
    if commanded is None:
        return None
    return commanded[: frame.env.satellite.wheel_axes.shape[1]]


def _mtq_dipole(frame: Frame) -> Optional[np.ndarray]:
    """Commanded magnetorquer dipole [A*m^2], (n_rods,)."""
    commanded = _commanded(frame)
    if commanded is None:
        return None
    return commanded[frame.env.satellite.wheel_axes.shape[1] :]

def _target_positions(frame:Frame) -> Optional[np.ndarray]:
    """Target positions [m], (n_targets, 3)"""
    targets = frame.env.mission_state.extra[0]
    return np.array([tgt.r_eci(frame.env.state.t) for tgt in targets])

def _target_velocities(frame:Frame) -> Optional[np.ndarray]:
    """Target velocities [m/s], (n_targets, 3)"""
    targets = frame.env.mission_state.extra[0]
    return np.array([tgt.v_eci(frame.env.state.t) for tgt in targets])

def _target_status(frame:Frame) -> Optional[np.ndarray]:
    """Per target status: 0 untracked, 1 cleared, 2 currently tracked (n_targets,)"""
    targets, cleared, current = frame.env.mission_state.extra
    return np.fromiter(
        (2 if i == current else 1 if i in cleared else 0 for i in range(len(targets))),
        dtype=np.int8, count=len(targets),
    )

DEFAULT_CHANNELS: Tuple[Channel, ...] = (
    Channel("t", lambda f: f.env.state.t, doc="s since epoch"),
    Channel("q_eci_body", _state("q_eci_body"), doc="scalar-first, ECI -> body"),
    Channel("omega_body", _state("omega_body"), doc="rad/s, body frame"),
    Channel("wheel_speeds", _state("wheel_speeds"), doc="rad/s"),
    Channel("r_eci", _state("r_eci"), doc="m, ECI"),
    Channel("v_eci", _state("v_eci"), doc="m/s, ECI"),
    Channel("target_q_eci_body", _target_quat, doc="current setpoint quaternion"),
    Channel(
        "target_idx",
        _info("mission/target_idx"),
        dtype=np.dtype(np.int32),
        doc="index of the target being tracked",
    ),
    Channel("hold_timer", _info("mission/hold_timer"), doc="s inside tolerance"),
    Channel(
        "phase",
        _info("mission/phase"),
        dtype=np.dtype("<U16"),
        doc="mission phase label",
    ),
    Channel("pointing_error_deg", _info("adcs/pointing_error_deg"), doc="deg to target"),
    Channel("body_rate", _info("adcs/body_rate"), doc="|omega_body|, rad/s"),
    Channel("wheel_speed_frac", _info("adcs/wheel_speed_frac"), doc="max |w| / w_max"),
    Channel(
        "obs",
        lambda f: f.obs,
        dtype=np.dtype(np.float32),
        doc="normalised observation, (11,)",
    ),
    Channel(
        "action",
        lambda f: f.action,
        dtype=np.dtype(np.float32),
        doc="normalised action in [-1, 1], (7,)",
    ),
    Channel(
        "wheel_torque",
        _wheel_torque,
        doc="commanded wheel motor torque [N*m]",
    ),
    Channel(
        "mtq_dipole",
        _mtq_dipole,
        doc="commanded magnetorquer dipole [A*m^2]",
    ),
    Channel("reward", lambda f: f.reward, doc="total step reward"),
    *(
        Channel(
            f"reward_{name}",
            _reward_component(name),
            doc=f"weighted reward term '{name}'",
        )
        for name in ALL_REWARD_COMPONENTS
    ),
    Channel(
        "terminated",
        lambda f: f.terminated,
        dtype=np.dtype(np.bool_),
        doc="mission completed",
    ),
    Channel(
        "truncated",
        lambda f: f.truncated,
        dtype=np.dtype(np.bool_),
        doc="step cap reached",
    ),
)

TARGET_TRACK_CHANNELS: Tuple[Channel, ...] = (*DEFAULT_CHANNELS,
    Channel("target_positions", _target_positions, doc="m, ECI, (n_targets,3)"),
    Channel("target_velocities", _target_velocities, doc="m/s, ECI, (n_targets,3)"),
    Channel("target_status", _target_status, dtype= np.dtype(np.int8), 
            doc="0 untracked, 1 cleared, 2 tracked")
)

MISSION_CHANNELS: Dict[str, Tuple[Channel, ...]] = {
    "slew": DEFAULT_CHANNELS,
    "target_track": TARGET_TRACK_CHANNELS
}


class EpisodeRecorder:
    """Accumulates one episode row by row, then stacks it into arrays.

    Split out from `record_episode` so a caller who already owns a rollout
    loop -- an evaluation harness, a notebook -- can drive the recording
    directly instead of handing over control of the loop.

    Shapes are never declared: each channel's shape and the sentinel it pads
    with are inferred from its first non-None value, so adding a channel means
    adding an extractor and nothing else.
    """

    def __init__(self, channels: Sequence[Channel] = DEFAULT_CHANNELS) -> None:
        self.channels = tuple(channels)

        # Check wether channels are empty 
        if not self.channels:
            raise ValueError("at least one channel is required ")
        
        # Check for duplicate channels 
        names = [c.name for c in self.channels]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate channel names: {duplicates}")

        # Create all empty lists 
        self.columns: Dict[str, List[Any]] = {c.name: [] for c in self.channels}

        self.has_reset = False 

    @property
    def n_rows(self)->int:
        """Rows recorded so far, reset row included"""
        return len(self.columns[self.channels[0].name])

    @property
    def n_steps(self) -> int:
        """Rows recorded by `record_step`, i.e. excluding the reset row."""
        if self.has_reset:
            return self.n_rows - 1
        else:
            return self.n_rows

    def record_reset(self, env: EventSatEnv, obs: np.ndarray, info: Dict[str, Any]) -> None:
        """Record row 0 from `env.reset()`.

        Must be called exactly once, before any `record_step`.
        """
        #Check if it is the first call 
        if self.has_reset:
            raise RuntimeError("record_reset already called; only called once per episode")
        
        self._append(Frame(env=env, obs=obs, info=info))
        self.has_reset = True

    def record_step(
        self,
        env: EventSatEnv,
        obs: np.ndarray,
        info: Dict[str, Any],
        action: np.ndarray,
        reward: float,
        reward_info: Optional[Dict[str, float]],
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record one row from an `env.step()` return."""

        # Check wether a reet row has been inserted
        if not self.has_reset:
            raise RuntimeError("record_reset must be called before record_step")

        # Rebuild reward dictionary
        if reward_info is None:
            reward_info = {n: info[f"reward/{n}"] for n in ALL_REWARD_COMPONENTS}

        self._append(
            Frame(
                env=env,
                obs=obs,
                info=info,
                action=action,
                reward=reward,
                reward_info=reward_info,
                terminated=terminated,
                truncated=truncated
            )
        )

    def _append(self, frame: Frame) -> None:
        """Run every extractor against one frame and stash the results."""
        for channel in self.channels:
            try:
                value = channel.extract(frame)
            except Exception as exc:
                raise RuntimeError(f"{channel.name!r} failed to extract at row {self.n_rows}")from exc

            self.columns[channel.name].append(value)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Stack the recorded rows into one array per channel.

        Returns:
            Channel name -> array of shape (N, *value_shape) and the channel's
            dtype, where N is `n_steps + 1`. Rows whose extractor returned
            None carry the sentinel for that dtype.

        Raises:
            ValueError: If no rows were recorded, if a channel never produced
                a non-None value (its shape is then unknowable), or if a
                string value would be truncated by its fixed-width dtype.
        """

        # Check if recording is empty
        n_rows = self.n_rows
        if n_rows == 0:
            raise ValueError("nothing recorded: call record_reset first")


        # Check wether all rows have the same length 
        lengths = {name: len(col) for name,col in self.columns.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Columns have different lengths: {lengths}")

        arrays: Dict[str, np.ndarray] = {}
        for channel in self.channels:
            column = self.columns[channel.name]

            # 1. shape, from the first row that actually has a value 
            first = next((val for val in column if val is not None), None)
            if first is None:
                raise ValueError(
                    f"channel {channel.name!r} is None in every row; its shape is unknowable"
                )
            shape = np.asarray(first).shape

            # 2. the sentinel that marks "undefined here"
            if channel.dtype.kind not in _FILL:
                raise ValueError(
                    f"channel {channel.name!r} has dtype {channel.dtype}, "
                    f"which has no fill value (supported kinds: {sorted(_FILL)})"
                )
            fill = _FILL[channel.dtype.kind]

            # 3. pre-fill, then write the rows that have values
            out = np.full((n_rows, *shape), fill, dtype=channel.dtype)
            for i, value in enumerate(column):
                if value is not None:
                    out[i] = value

            # 4. fixed-width strings truncate silently
            # Check wether the input value comes out at the other end
            if channel.dtype.kind == "U":
                for i, value in enumerate(column):
                    if value is not None and out[i] != value:
                        raise ValueError(
                            f"channel {channel.name!r}: {value!r} does not fit "
                            f"in {channel.dtype}"
                        )

            arrays[channel.name] = out
        return arrays
    
    def manifest(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Channel name -> {shape, dtype, doc}, for the file's metadata."""
        return {
            channel.name: {
                "shape": list(arrays[channel.name].shape),
                "dtype": str(arrays[channel.name].dtype),
                "doc": channel.doc,
            }
            for channel in self.channels
        }

# 4 reaction-wheel torques + 3 magnetorquer dipoles -- matches
# EventSatEnv.action_space, which is Box(-1, 1, shape=(7,)).
N_ACTIONS = 7


def zero_policy() -> Policy:
    """Policy that commands nothing. Free-drift baseline, needs no checkpoint."""
    return lambda obs: np.zeros(N_ACTIONS, dtype=np.float32)


def random_policy(seed: int = SEED) -> Policy:
    """Uniform random actions from a private, seeded generator.

    Its own RNG rather than the env's: drawing from `env.np_random` would
    couple the action sequence to how much sensor noise was consumed that
    step, so a change to a sensor model would silently change the actions.
    """
    rng = np.random.default_rng(seed)
    return lambda obs: rng.uniform(-1.0, 1.0, size=N_ACTIONS).astype(np.float32)


def _find_rl_module(root: Path) -> Path:
    """Locate the RLModule directory inside an RLlib checkpoint.

    RLlib has moved this layout between releases, and `adcs_train_ppo` saves
    with whichever of `save_to_path`/`save` the installed version offers -- so
    the depth of the module inside the checkpoint is not fixed. Probing a list
    of known layouts keeps a recording working across versions instead of
    failing on the first path that changed, which is the same tolerance
    `build_algo`/`restore_checkpoint` already apply on the training side.

    Args:
        root: A checkpoint directory, as written by `adcs_train_ppo`.

    Returns:
        The directory to hand to `RLModule.from_checkpoint`. Falls through to
        `root` itself, which is correct when the checkpoint *is* the module.

    Raises:
        FileNotFoundError: If none of the candidate layouts exists.
    """
    candidates = (
        root / "learner_group" / "learner" / "rl_module" / "default_policy",
        root / "learner_group" / "learner" / "rl_module",
        root / "rl_module" / "default_policy",
        root / "rl_module",
        root,
    )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"no RLModule directory found under {root}; tried "
        + ", ".join(str(c.relative_to(root)) or "." for c in candidates)
    )


def load_checkpoint(checkpoint_path: Optional[Path | str] = None) -> Policy:
    """Restore a trained PPO policy as a deterministic `policy(obs) -> action`.

    Takes the *mean* of the action distribution, never a sample: sampling
    would make the recorded episode unreproducible even with a fixed seed.

    `ray` and `torch` are imported inside this function on purpose. They are
    not declared in pyproject.toml, and importing them at module scope would
    make the whole recorder unusable without them -- which is exactly what the
    pluggable-policy design exists to avoid.

    Args:
        checkpoint_path: An RLlib checkpoint directory. Defaults to
            CHECKPOINT_DIR, where adcs_train_ppo writes.

    Returns:
        A deterministic policy callable.

    Raises:
        ImportError: If ray[rllib] or torch is not installed.
        FileNotFoundError: If the checkpoint directory holds no RLModule.
    """
    try:
        import torch
        from ray.rllib.core.rl_module.rl_module import RLModule
    except ImportError as exc:
        raise ImportError(
            'ray[rllib] and torch are required to load a trained policy. '
            "Install with: uv sync --extra dev --extra rl"
        ) from exc

    root = Path(checkpoint_path or CHECKPOINT_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"no checkpoint at {root}")

    module_dir = _find_rl_module(root)
    rl_module = RLModule.from_checkpoint(module_dir)

    # A MultiRLModule wraps the per-policy modules; a single-agent checkpoint
    # may already be the module itself.
    if hasattr(rl_module, "__getitem__") and not hasattr(rl_module, "forward_inference"):
        rl_module = rl_module["default_policy"]

    def policy(obs: np.ndarray) -> np.ndarray:
        batch = {"obs": torch.from_numpy(np.asarray(obs, dtype=np.float32)[None])}
        with torch.no_grad():
            out = rl_module.forward_inference(batch)

        if "actions" in out:
            action = out["actions"][0]
        else:
            # Continuous action head emits [mean(7), log_std(7)]. The mean is
            # the deterministic action; sampling would break reproducibility.
            dist_inputs = out["action_dist_inputs"][0]
            action = dist_inputs[: dist_inputs.shape[-1] // 2]

        return action.cpu().numpy().astype(np.float32)

    return policy

def _jsonable(value: Any) -> Any:
    """Coerce a config value into something json.dumps accepts.

    Mission configs carry numpy -- TargetTrackConfig's pointing vectors are
    ndarrays -- and `asdict` copies them through verbatim, unlike the
    hardware fields below which are converted one by one. np.float64 needs
    no help (it subclasses float), but np.int64 does.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

def _git_commit() -> Optional[str]:
    """Short HEAD hash, or None outside a git checkout or without git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None

def build_meta(
    recorder: EpisodeRecorder,
    config: AdcsEnvConfig,
    *,
    env: EventSatEnv,
    arrays: Dict[str,np.ndarray],
    seed: int,
    policy_name: str,
    checkpoint_path: Optional[Path] = None,
    terminated: bool = False,
    truncated: bool = False,
) -> Dict[str, Any]:
    """Assemble the run's metadata.

    Everything needed to interpret or reproduce the episode without the code
    that wrote it: seed, timing, mission and reward configuration, which
    policy ran, how the episode ended, the git commit, and the channel
    manifest.

    The hardware constants come off `env` rather than from the eventsat module,
    so they describe the satellite that actually flew this episode rather than
    whatever the module currently declares. They are what a consumer needs to
    interpret the channels at all: `wheel_torque` has no meaning without the
    limit it was measured against, and `wheel_speeds` says nothing about
    momentum without the cluster geometry.

    Args:
        recorder: The recorder that captured the episode; read for its row
            counts and the channel manifest.
        config: The env config the episode ran under.
        env: The env that ran it, for the actuator limits and wheel geometry.
        arrays: The stacked channels, for the manifest's shapes and dtypes.
        seed: Seed the episode was reset with.
        policy_name: Which policy drove it.
        checkpoint_path: Where that policy was loaded from, if it was.
        terminated: Whether the mission completed.
        truncated: Whether the step cap was hit.
    """

    # Reason for stopping 
    if terminated:
        reason = "mission_done"
    elif truncated:
        reason = "max_steps"
    else:
        reason = "incomplete"

    return {
        #Versioning 
        "schema_version": SCHEMA_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),

        # what produced it
        "seed": seed,
        "policy": {
            "name": policy_name,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        },

        # how it ended
        "n_rows": recorder.n_rows,
        "n_steps": recorder.n_steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": reason,

        # config
        "step_s": env.sim.step_s,
        "start_step": config.start_step,
        "body_rate_thresh": config.body_rate_thresh,
        "max_body_rate": config.max_body_rate,
        "reward_weights": dict(config.reward_weights or {}),
        "mission": {k: _jsonable(v) for k, v in asdict(config.mission).items()},


        # Which reward channels are paid every step and which only on an event.
        # Recorded rather than inferred: the two are indistinguishable in the
        # data -- a dense term can sit at zero for a whole episode, and an event
        # term is a plain float column like any other.
        "dense_components": list(REWARD_COMPONENTS),
        "event_components": list(EVENT_COMPONENTS),

        # Hardware limits and geometry, so the channels can be read without
        # importing the sim. `max_action` is what `action` was scaled by, and
        # keeps the action vector's own layout: wheel torque [N*m] first, MTQ
        # dipole [A*m^2] after. `Recording` slices it into the two limits.
        "max_action": env.max_action.tolist(),
        "wheel_max_speed": float(env.wheel_max_speed),
        "wheel_axes": env.satellite.wheel_axes.tolist(),
        "wheel_inertia": env.satellite.wheel_inertia.tolist(),

        "channels": recorder.manifest(arrays),
    }
    
   


def record_episode(
    policy: Policy,
    *,
    seed: int = SEED,
    config: Optional[AdcsEnvConfig] = None,
    channels: Sequence[Channel] = None,
    max_steps: Optional[int] = None,
    policy_name: str = "unknown",
    checkpoint_path: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Run one episode under `policy` and return its arrays and metadata.

    Reproducibility rests on `env.reset(seed=seed)`. `AdcsEnvConfig.seed` is
    copied onto the sim config by the env but never consumed -- every draw
    goes through `env.np_random` -- so setting only the config seed produces a
    different episode every time. Both are set here.

    Args:
        policy: Deterministic `policy(obs) -> action`.
        seed: Seed for the env RNG, and recorded in the metadata.
        config: Env config. Defaults to the EventSat config.
        channels: What to record. None picks the set MISSION_CHANNELS holds
            for the config's mission.
        max_steps: Override the mission's step cap, for short test episodes.

    Returns:
        (arrays, meta) -- ready to hand to `save_episode`.
    """
    config = config or EVENTSAT_ENV_CONFIG
    if channels is None:
        channels = MISSION_CHANNELS[config.mission.type]

    # Modify the config with given parameters  
    if max_steps is not None:
        config = replace(config, mission= replace(config.mission, max_steps=max_steps))
    config = replace(config, seed=seed)

    # setup all classses for recording 
    env = EventSatEnv(config=config)
    recorder = EpisodeRecorder(channels=channels)

    # Reset the env and record the values acccordingly
    obs, info = env.reset(seed=seed)
    recorder.record_reset(env=env, obs=obs, info=info)

    terminated = False 
    truncated = False 

    while not (terminated or truncated):
        action = np.clip(policy(obs), env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        recorder.record_step(
            env, obs, info, action, reward, None, terminated, truncated
        )

    arrays = recorder.to_arrays()
    meta = build_meta(
        recorder,
        config,
        env=env,
        arrays=arrays,
        seed=seed,
        policy_name=policy_name,
        checkpoint_path=checkpoint_path,
        terminated=terminated,
        truncated=truncated,
    )

    return arrays, meta





def _default_output_path(seed: int, policy_name: str, mission: str) -> Path:
    """`data/records/<mission>/adcs_ep_<policy>_seed<seed>.npz`.

    Absolute, so the file lands in the same place whatever the working
    directory. The name carries the two inputs that define the episode, which
    makes a directory listing readable and means re-recording the same
    (policy, seed) overwrites rather than accumulating near-identical files.

    Creates nothing: `save_episode` makes the directory when it writes, so
    asking for the default path stays free of side effects.
    """
    return RECORD_ROOT / mission / f"adcs_ep_{policy_name}_seed{seed}.npz"

def checkpoint_dir(mission: str) -> Path:
    """Where `mission`'s trained policy is read from.

    Mirrors `adcs_train_ppo.checkpoint_dir`, which writes there. The layout is
    duplicated rather than imported for the same reason CHECKPOINT_ROOT is:
    importing it would pull in ray. A change to the scheme has to be made in
    both places or recording silently stops finding policies.
    """
    return CHECKPOINT_ROOT / mission


# Policy builders by --policy name. Both this dispatch and the CLI's `choices`
# read from here, so adding a policy is one entry rather than two edits that can
# drift apart. The uniform (seed, checkpoint_path) signature is what lets them
# share one table -- each builder ignores the argument it does not need.
POLICY_BUILDERS: Dict[str, Callable[[int, Optional[Path]], Policy]] = {
    "zero": lambda seed, checkpoint_path: zero_policy(),
    "random": lambda seed, checkpoint_path: random_policy(seed),
    "checkpoint": lambda seed, checkpoint_path: load_checkpoint(checkpoint_path),
}


def _build_policy(
    name: str, seed: int, checkpoint_path: Optional[Path]
) -> Policy:
    """Resolve a --policy name to a callable.

    Args:
        name: A key of POLICY_BUILDERS.
        seed: Policy seed, used only by the stochastic builders.
        checkpoint_path: Passed to `load_checkpoint`; None means its default.

    Returns:
        A `policy(obs) -> action` callable.

    Raises:
        ValueError: If `name` is not a known policy.
    """
    try:
        builder = POLICY_BUILDERS[name]
    except KeyError:
        # The KeyError adds nothing the message does not already say.
        raise ValueError(
            f"unknown policy {name!r}; expected one of {sorted(POLICY_BUILDERS)}"
        ) from None

    return builder(seed, checkpoint_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI: which policy, which seed, how long, and where to write.

    Args:
        argv: Argument list. None reads sys.argv, so the CLI behaves normally;
            passing a list lets a test drive the parser without a subprocess.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="adcs_record",
        description="Record one ADCS episode to a compressed .npz file.",
    )
    parser.add_argument(
        "--policy",
        # Read from the dispatch table, so a builder added there is offered by
        # the CLI without a second edit.
        choices=sorted(POLICY_BUILDERS),
        default="checkpoint",
        help="what drives the episode (default: %(default)s)",
    )
    parser.add_argument(
        "--mission",
        # Read from the registry, so a mission added there is offered by the
        # CLI without a second edit.
        choices=sorted(MISSION_TYPES),
        default=DEFAULT_MISSION,
        help="which mission to record on (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        # Spelled out rather than %(default)s: the default is None, and what is
        # worth showing is the path load_checkpoint falls back to.
        help=f"RLlib checkpoint directory (default: {CHECKPOINT_ROOT}/<mission>)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="env and policy seed (default: %(default)s)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override the mission step cap, for short test recordings",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: data/records/<mission>/adcs_ep_<policy>_seed<seed>.npz)",
    )

    args = parser.parse_args(argv)

    # `choices` and `type` cannot express "positive". parser.error gives the
    # usage message and exit code 2 that a CLI should, rather than a traceback.
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be >= 1")

    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Record one episode and write it to disk.

    Args:
        argv: Argument list, or None to read sys.argv.

    Returns:
        Process exit code. 0 on success, 1 if the policy could not be built,
        2 (via argparse) on a bad argument.
    """
    args = parse_args(argv)

    # Resolve what load_checkpoint would fall back to, so the metadata names
    # the checkpoint that actually produced the file rather than None. Held at
    # None for the other policies, which never read it -- recording a path that
    # was ignored would be worse than recording nothing.
    checkpoint_path = (
        (args.checkpoint or checkpoint_dir(args.mission)) if args.policy == "checkpoint" else None
    )

    config = replace(EVENTSAT_ENV_CONFIG, mission=MISSION_CONFIGS[args.mission])

    # Only the anticipated failures: no ray, no checkpoint, unknown name. An
    # unexpected exception should still surface as a traceback rather than be
    # flattened into an exit code.
    try:
        policy = _build_policy(args.policy, args.seed, checkpoint_path)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    arrays, meta = record_episode(
        policy,
        seed=args.seed,
        config=config,
        max_steps=args.max_steps,
        policy_name=args.policy,
        checkpoint_path=checkpoint_path,
    )

    path = save_episode(
        arrays, args.out or _default_output_path(args.seed, args.policy, args.mission), meta
    )

    # Path first and on its own line, so it can be copied or piped.
    print(path)
    print(
        f"  {meta['n_rows']} rows, {meta['termination_reason']}, "
        f"{path.stat().st_size / 1024:.1f} kB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

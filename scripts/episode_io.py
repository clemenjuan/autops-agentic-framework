"""Read and write one recorded ADCS episode.

The archive format is a compressed NPZ: one array per channel, all sharing the
leading dimension ``N = n_steps + 1``, plus a single ``meta`` entry holding a
JSON string. :mod:`scripts.adcs_record` writes it; matplotlib plots and the
Rerun viewer read it.

This module sits beside ``adcs_record`` rather than inside it, and imports
nothing from the simulation, on purpose. Importing ``adcs_record`` reaches
``adcs_gymnasium_wrapper`` -> ``propagator``, which boots the Orekit JVM at
module import time -- so a consumer that only wants to draw a line chart would
pay for a JVM and the Orekit data files first. Keeping the read side free of
those imports is what makes ``Recording.load`` cheap; it depends on nothing but
numpy and the standard library.

Which of the three names to use
-------------------------------
``save_episode``  the writer's entry point. Takes the ``arrays`` and ``meta``
                  dicts that ``EpisodeRecorder`` produces, because that is what
                  the recorder has -- it never builds a ``Recording``.

``Recording``     the reader's entry point, and what every plot should take::

                      rec = Recording.load(path)
                      ax.plot(rec.t_rel, rec["pointing_error_deg"])

``load_episode``  the raw primitive underneath ``Recording.load``, for a
                  consumer that wants the two plain dicts -- a schema check, a
                  one-off in a notebook. Reach for ``Recording`` otherwise.

:class:`Recording` is a *view*: the raw channels plus everything derivable from
them by slicing, relabelling, or reading the file's own channel manifest. It
recomputes no physics -- anything it cannot derive from the archive is a
channel the recorder should have written, or a constant ``build_meta`` should
have carried.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Prefix under which `adcs_record` stores the individual reward terms. One
# definition, because both the writer's channel names and this module's
# `reward_terms` split depend on it.
REWARD_PREFIX = "reward_"

# How the dense/event split is recovered from a file written before
# `build_meta` carried the two component lists. Both members of
# `adcs_rewards.EVENT_COMPONENTS` end in this, so it reconstructs the split
# exactly for every file recorded so far -- but it is a guess about naming, not
# a fact from the file, which is why newer archives state the lists outright.
_LEGACY_EVENT_SUFFIX = "_bonus"


def save_episode(
    arrays: Dict[str, np.ndarray],
    path: Path | str,
    meta: Dict[str, Any],
) -> Path:
    """Write one episode to a compressed NPZ.

    The metadata goes in as a single `meta` entry holding a JSON string, so
    the archive is self-contained: there is no sidecar to lose track of.

    Args:
        arrays: Channel name -> array, from `EpisodeRecorder.to_arrays`.
        path: Destination. Parent directories are created.
        meta: JSON-serialisable run metadata.

    Returns:
        The path written.

    Raises:
        ValueError: If `arrays` already contains a `meta` key, or if the
            channel arrays disagree on their leading dimension.
    """
    if "meta" in arrays:
        raise ValueError("'meta' is reserved for the metadata blob")

    lengths = {name: a.shape[0] for name, a in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"arrays disagree on their leading dimension: {lengths}")

    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def load_episode(path: Path | str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Read back what `save_episode` wrote.

    Args:
        path: An NPZ written by this module.

    Returns:
        (arrays, meta). Arrays are materialised into a plain dict, so the
        archive is closed before this returns -- NpzFile is lazy, and handing
        one back leaks a file handle for as long as the caller holds it.
    """
    with np.load(path, allow_pickle=False) as data:
        if "meta" not in data.files:
            raise ValueError(f"{path} has no 'meta' entry; not written by save_episode")
        arrays = {name: data[name] for name in data.files if name != "meta"}
        meta = json.loads(data["meta"].item())
    return arrays, meta


@dataclass(frozen=True)
class Recording:
    """One recorded episode, with the views a consumer actually plots.

    Channels are reached by subscript (``rec["omega_body"]``), which keeps the
    archive's own names visible rather than hiding them behind an attribute per
    channel that would have to be extended every time a channel is added.
    Attributes are reserved for what the archive does *not* hold literally:
    elapsed time, the two halves of ``command_torque``, the reward split, the
    event rows, and the configured limits that threshold lines are drawn at.

    Limits that older archives do not carry are reported as None rather than
    raising, so a file recorded before `build_meta` grew a field still plots --
    just without that saturation band.

    Attributes:
        arrays: Channel name -> array, exactly as `load_episode` returns them.
        meta: The run metadata blob.
    """

    arrays: Dict[str, np.ndarray]
    meta: Dict[str, Any]

    @classmethod
    def load(cls, path: Path | str) -> "Recording":
        """Read a recording off disk."""
        return cls(*load_episode(path))

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError:
            # The bare KeyError names the channel but not the alternatives,
            # and the whole point of a manifest-driven schema is that the
            # alternatives vary by file.
            raise KeyError(
                f"no channel {name!r} in this recording; it has {sorted(self.arrays)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self.arrays

    def __len__(self) -> int:
        """Rows, i.e. the reset row plus one per step."""
        return int(self["t"].shape[0])

    def doc(self, name: str) -> str:
        """The channel's one-line description, units included.

        Straight from the file's manifest, so an axis label follows the
        recorder's own wording instead of restating it here. Falls back to the
        channel name for a file written without a manifest.
        """
        return self.meta.get("channels", {}).get(name, {}).get("doc", name)

    # -------------------------------------------------------------------------
    # Time
    # -------------------------------------------------------------------------

    @property
    def t_rel(self) -> np.ndarray:
        """Seconds since the start of the episode.

        The ``t`` channel is absolute -- seconds since the epoch -- so with a
        non-zero ``start_step`` it opens at a large offset that no plot wants
        on its x-axis. ``rec["t"]`` still gives the absolute clock.
        """
        return self["t"] - self["t"][0]

    @property
    def step(self) -> np.ndarray:
        """Row index, for plotting against steps rather than seconds."""
        return np.arange(len(self))

    @property
    def step_s(self) -> float:
        """Simulation timestep [s]."""
        return float(self.meta["step_s"])

    # -------------------------------------------------------------------------
    # Actuators
    # -------------------------------------------------------------------------

    @property
    def n_wheels(self) -> int:
        """Reaction wheels in this recording.

        Read off the data rather than assumed to be four, which is what lets
        `wheel_torque` and `mtq_dipole` split `command_torque` correctly for a
        satellite with a different wheel count.
        """
        return int(self["wheel_speeds"].shape[1])

    @property
    def wheel_torque(self) -> np.ndarray:
        """Commanded wheel motor torque [N*m], (N, n_wheels).

        Commanded, not achieved: the recorder stores ``action * max_action``,
        and the wheels clip against their momentum envelope inside
        `actuators.apply_reaction_wheel`. The two diverge exactly at
        saturation.
        """
        return self["command_torque"][:, : self.n_wheels]

    @property
    def mtq_dipole(self) -> np.ndarray:
        """Commanded magnetorquer dipole [A*m^2], (N, n_rods)."""
        return self["command_torque"][:, self.n_wheels :]

    @property
    def max_action(self) -> Optional[np.ndarray]:
        """Per-actuator command limit, or None if the file predates the field.

        Wheel torque [N*m] first, magnetorquer dipole [A*m^2] after, matching
        the layout of `command_torque`.
        """
        limits = self.meta.get("max_action")
        return None if limits is None else np.asarray(limits, dtype=float)

    @property
    def wheel_max_torque(self) -> Optional[np.ndarray]:
        """Per-wheel torque limit [N*m], or None.

        One entry per wheel rather than a scalar: nothing requires a cluster to
        be built from identical wheels.
        """
        limits = self.max_action
        return None if limits is None else limits[: self.n_wheels]

    @property
    def mtq_max_dipole(self) -> Optional[np.ndarray]:
        """Per-rod dipole limit [A*m^2], or None."""
        limits = self.max_action
        return None if limits is None else limits[self.n_wheels :]

    @property
    def wheel_max_speed(self) -> float:
        """Wheel speed at saturation [rad/s]."""
        return float(self.meta["wheel_max_speed"])

    @property
    def wheel_axes(self) -> Optional[np.ndarray]:
        """Wheel spin axes in the body frame, (3, n_wheels), or None.

        With `wheel_inertia`, enough to turn `wheel_speeds` into the cluster's
        momentum contribution in the body frame.
        """
        axes = self.meta.get("wheel_axes")
        return None if axes is None else np.asarray(axes, dtype=float)

    @property
    def wheel_inertia(self) -> Optional[np.ndarray]:
        """Per-wheel axial inertia [kg*m^2], (n_wheels,), or None."""
        inertia = self.meta.get("wheel_inertia")
        return None if inertia is None else np.asarray(inertia, dtype=float)

    # -------------------------------------------------------------------------
    # Reward
    # -------------------------------------------------------------------------

    @property
    def reward_terms(self) -> Dict[str, np.ndarray]:
        """Term name -> weighted per-step series, in declaration order.

        Assembled from the channel manifest rather than from a list written out
        here, so a term added to `adcs_rewards` plots itself -- the same
        contract the recorder and `EventSatEnv._get_info` already keep.
        """
        names = self.meta.get("channels") or self.arrays
        return {
            name[len(REWARD_PREFIX) :]: self.arrays[name]
            for name in names
            if name.startswith(REWARD_PREFIX) and name in self.arrays
        }

    @property
    def dense_terms(self) -> Dict[str, np.ndarray]:
        """The reward terms paid every step. Stack these; stem the events."""
        return self._reward_split(dense=True)

    @property
    def event_terms(self) -> Dict[str, np.ndarray]:
        """The sparse bonuses, paid only on the step their event fired."""
        return self._reward_split(dense=False)

    def _reward_split(self, *, dense: bool) -> Dict[str, np.ndarray]:
        """One half of the dense/event split.

        Prefers the component lists in the metadata. Falls back to the naming
        convention for archives written before those were carried -- see
        `_LEGACY_EVENT_SUFFIX`.
        """
        terms = self.reward_terms
        declared = self.meta.get("dense_components" if dense else "event_components")

        if declared is not None:
            wanted = set(declared)
            return {name: series for name, series in terms.items() if name in wanted}

        return {
            name: series
            for name, series in terms.items()
            if name.endswith(_LEGACY_EVENT_SUFFIX) is not dense
        }

    @property
    def reward_weights(self) -> Dict[str, float]:
        """Weight per component, as the episode was run with.

        The recorded terms are already weighted; this is what divides them back
        out to the raw quantity a term measures. Entries recorded as null -- a
        weight that existed when the file was written and has since been
        dropped -- are left out rather than handed on as None.
        """
        weights = self.meta.get("reward_weights") or {}
        return {name: value for name, value in weights.items() if value is not None}

    @property
    def cum_reward(self) -> np.ndarray:
        """Return to date, (N,).

        `nancumsum` rather than `cumsum`: the reset row's reward is NaN by
        construction -- nothing was earned there -- and one NaN would otherwise
        take the entire curve with it.
        """
        return np.nancumsum(self["reward"])

    # -------------------------------------------------------------------------
    # Events and outcome
    # -------------------------------------------------------------------------

    @property
    def clear_steps(self) -> np.ndarray:
        """Rows on which a target was cleared.

        Derived from `target_idx`, not from the `target_cleared_bonus` channel:
        that channel is *weighted*, so a run whose weight is 0.0 would report
        no clears at all. The index increments on every clear, the last one
        included.
        """
        idx = self["target_idx"]
        return np.flatnonzero(np.diff(idx, prepend=idx[0]) > 0)

    @property
    def done_step(self) -> Optional[int]:
        """Row the mission completed on, or None if it never did."""
        hit = np.flatnonzero(self["terminated"])
        return int(hit[0]) if hit.size else None

    @property
    def termination_reason(self) -> str:
        """``mission_done``, ``max_steps`` or ``incomplete``."""
        return str(self.meta["termination_reason"])

    # -------------------------------------------------------------------------
    # Task configuration -- what threshold lines are drawn at
    # -------------------------------------------------------------------------

    @property
    def tolerance_deg(self) -> float:
        """Pointing error that counts as on target [deg]."""
        return float(self.meta["mission"]["tolerance_deg"])

    @property
    def hold_time(self) -> float:
        """Seconds inside tolerance before a target is cleared."""
        return float(self.meta["mission"]["hold_time"])

    @property
    def sigma(self) -> float:
        """Boundary-layer width the reward uses, as a quaternion vector norm.

        Recomputed rather than read: `MissionConfig.sigma` is a property, so
        `dataclasses.asdict` leaves it out of the recorded mission config.
        """
        return float(np.sin(np.deg2rad(self.tolerance_deg) / 2.0))

    @property
    def num_targets(self) -> int:
        """Targets in the episode's sequence."""
        return int(self.meta["mission"]["num_targets"])

    @property
    def body_rate_thresh(self) -> float:
        """Body rate above which the slew-rate penalty starts [rad/s]."""
        return float(self.meta["body_rate_thresh"])

    @property
    def max_body_rate(self) -> float:
        """Body rate the observation normalises against [rad/s]."""
        return float(self.meta["max_body_rate"])

    @property
    def label(self) -> str:
        """Short identifier for a legend entry or a figure title."""
        return f"{self.meta['policy']['name']} (seed {self.meta['seed']})"

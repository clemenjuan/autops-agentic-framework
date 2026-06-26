"""Constellation world-model trace recorder for the SSA scenario.

Mirrors ``src.eventsat.trace.WorldModelTraceEpisode`` but records a satellite
axis (``sat_0..sat_{N-1}``) plus SSA collective coverage fields, producing the
``ssa_world_model_v1`` dataset documented in space-world-models'
``docs/research_tracker.md``.

The 25D obs/state encoding is reused per satellite. Actions are one-hot over the
8 SSA modes (``src.ssa.env.SSA_MODES``), which differ in order and length from
the 7 EventSat modes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.eventsat.trace import STATE_NAMES, state_vector_from_observation
from src.eventsat.world_model import OBS25_NAMES, eventsat_observation_to_vector
from src.ssa.env import SSA_MODES

SSA_MODE_TO_INDEX = {mode: idx for idx, mode in enumerate(SSA_MODES)}


def ssa_action_from_mode(mode: str) -> np.ndarray:
    """One-hot over the 8 SSA modes; unknown modes fall back to charging."""
    vec = np.zeros(len(SSA_MODES), dtype=np.float32)
    vec[SSA_MODE_TO_INDEX.get(mode, SSA_MODE_TO_INDEX["charging"])] = 1.0
    return vec


def ssa_mode_index(mode: str) -> int:
    return SSA_MODE_TO_INDEX.get(mode, SSA_MODE_TO_INDEX["charging"])


def _satellite_ids(observation: Any) -> List[str]:
    if not hasattr(observation, "constellation_state"):
        return []
    return list(observation.constellation_state.satellites.keys())


def _global_float(observation: Any, key: str, default: float = 0.0) -> float:
    if not hasattr(observation, "constellation_state"):
        return default
    info = getattr(observation.constellation_state, "global_info", {}) or {}
    try:
        return float(info.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class ConstellationTraceEpisode:
    """One SSA episode converted to the constellation world-model schema."""

    episode_id: int
    seed: int
    sat_ids: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        step: int,
        observation: Any,
        env_actions: Dict[str, Any],
        rewards: Dict[str, float],
        info: Dict[str, Any],
    ) -> None:
        if not self.sat_ids:
            self.sat_ids = _satellite_ids(observation)
        sat_ids = self.sat_ids
        per_sat_info = dict(info.get("per_satellite", {}))

        obs = np.zeros((len(sat_ids), len(OBS25_NAMES)), dtype=np.float32)
        state = np.zeros((len(sat_ids), len(STATE_NAMES)), dtype=np.float32)
        action = np.zeros((len(sat_ids), len(SSA_MODES)), dtype=np.float32)
        reward = np.zeros((len(sat_ids),), dtype=np.float32)
        mode = np.zeros((len(sat_ids),), dtype=np.int64)
        resolved = np.zeros((len(sat_ids),), dtype=np.int64)
        forced = np.zeros((len(sat_ids),), dtype=np.float32)

        for idx, sat_id in enumerate(sat_ids):
            sat_action = env_actions.get(sat_id, {}) if isinstance(env_actions, dict) else {}
            requested = str(sat_action.get("mode", "charging")) if isinstance(sat_action, dict) else "charging"
            sat_info = per_sat_info.get(sat_id, {}) if isinstance(per_sat_info, dict) else {}
            requested = str(sat_info.get("requested_mode", requested))
            resolved_mode = str(sat_info.get("resolved_mode", requested))

            obs[idx] = eventsat_observation_to_vector(observation, sat_id).obs25.astype(np.float32)
            state[idx] = state_vector_from_observation(observation, sat_id)
            action[idx] = ssa_action_from_mode(requested)
            reward[idx] = float(rewards.get(sat_id, 0.0)) if isinstance(rewards, dict) else 0.0
            mode[idx] = ssa_mode_index(requested)
            resolved[idx] = ssa_mode_index(resolved_mode)
            forced[idx] = float(bool(sat_info.get("forced", False)))

        self.rows.append(
            {
                "step": int(step),
                "obs": obs,
                "action": action,
                "state": state,
                "reward": reward,
                "mode": mode,
                "resolved_mode": resolved,
                "forced_mode": forced,
                "delivered_coverage": _global_float(observation, "ssa_delivered_coverage"),
                "onboard_coverage": _global_float(observation, "ssa_onboard_coverage"),
                "archive_records": _global_float(observation, "ssa_ground_archive_records"),
            }
        )

    def as_arrays(self) -> Dict[str, np.ndarray]:
        n_sat = len(self.sat_ids)
        if not self.rows:
            return {
                "obs": np.zeros((0, n_sat, len(OBS25_NAMES)), dtype=np.float32),
                "action": np.zeros((0, n_sat, len(SSA_MODES)), dtype=np.float32),
                "state": np.zeros((0, n_sat, len(STATE_NAMES)), dtype=np.float32),
                "reward": np.zeros((0, n_sat), dtype=np.float32),
                "mode": np.zeros((0, n_sat), dtype=np.int64),
                "resolved_mode": np.zeros((0, n_sat), dtype=np.int64),
                "forced_mode": np.zeros((0, n_sat), dtype=np.float32),
                "delivered_coverage": np.zeros((0,), dtype=np.float32),
                "onboard_coverage": np.zeros((0,), dtype=np.float32),
                "archive_records": np.zeros((0,), dtype=np.int64),
            }
        return {
            "obs": np.stack([r["obs"] for r in self.rows]).astype(np.float32),
            "action": np.stack([r["action"] for r in self.rows]).astype(np.float32),
            "state": np.stack([r["state"] for r in self.rows]).astype(np.float32),
            "reward": np.stack([r["reward"] for r in self.rows]).astype(np.float32),
            "mode": np.stack([r["mode"] for r in self.rows]).astype(np.int64),
            "resolved_mode": np.stack([r["resolved_mode"] for r in self.rows]).astype(np.int64),
            "forced_mode": np.stack([r["forced_mode"] for r in self.rows]).astype(np.float32),
            "delivered_coverage": np.asarray([r["delivered_coverage"] for r in self.rows], dtype=np.float32),
            "onboard_coverage": np.asarray([r["onboard_coverage"] for r in self.rows], dtype=np.float32),
            "archive_records": np.asarray([r["archive_records"] for r in self.rows], dtype=np.int64),
        }

    def write_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **self.as_arrays(),
            sat_ids=np.asarray(self.sat_ids),
            episode_seed=np.asarray(self.seed, dtype=np.int64),
            episode_id=np.asarray(self.episode_id, dtype=np.int64),
        )


def write_ssa_trace_metadata(path: Path, payload: Dict[str, Any]) -> None:
    base = {
        "schema": "ssa_world_model_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "obs_names": list(OBS25_NAMES),
        "action_names": list(SSA_MODES),
        "state_names": list(STATE_NAMES),
        "mode_names": list(SSA_MODES),
        "collective_fields": ["delivered_coverage", "onboard_coverage", "archive_records"],
        "notes": "Constellation EventSat with a satellite axis; AUTOPS-native state only.",
    }
    base.update(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base, indent=2), encoding="utf-8")

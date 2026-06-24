"""World-model trace schema/export helpers for AUTOPS EventSat.

The exporter writes simulator-generated telemetry for offline LeWM/Dreamer
training. It deliberately records AUTOPS-native state only; thermal and pointing
are not synthesized here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.eventsat.world_model import (
    ACTION_NAMES,
    MODE_LIST,
    OBS25_NAMES,
    action_from_mode,
    eventsat_observation_to_vector,
)

STATE_NAMES = (
    "battery_soc",
    "current_mode_idx",
    "in_sunlight",
    "ground_pass_active",
    "orbital_phase",
    "time_to_next_eclipse",
    "time_to_next_pass",
    "remaining_pass_duration",
    "following_gap_steps",
    "data_stored_mb",
    "obc_data_mb",
    "jetson_raw_mb",
    "jetson_compressed_mb",
    "data_downlinked_mb",
    "uncompressed_observations",
    "compression_progress",
    "undetected_observations",
    "detection_progress",
    "total_observation_s",
    "total_detections",
    "storage_capacity_mb",
    "jetson_capacity_mb",
    "daily_downlink_budget_mb",
    "achievable_downlink_mb",
    "health_nominal",
)


def state_vector_from_observation(observation: Any) -> np.ndarray:
    encoded = eventsat_observation_to_vector(observation)
    raw = encoded.raw
    mode = str(raw.get("current_mode", "charging"))
    return np.asarray(
        [
            raw.get("battery_soc", 0.0),
            MODE_LIST.index(mode) if mode in MODE_LIST else 0,
            float(bool(raw.get("in_sunlight", False))),
            float(bool(raw.get("ground_pass_active", False))),
            raw.get("orbital_phase", 0.0),
            raw.get("time_to_next_eclipse", 0.0),
            raw.get("time_to_next_pass", 0.0),
            raw.get("remaining_pass_duration", 0.0),
            raw.get("following_gap_steps", 0.0),
            raw.get("data_stored_mb", 0.0),
            raw.get("obc_data_mb", 0.0),
            raw.get("jetson_raw_mb", 0.0),
            raw.get("jetson_compressed_mb", 0.0),
            raw.get("data_downlinked_mb", 0.0),
            raw.get("uncompressed_observations", 0.0),
            raw.get("compression_progress", 0.0),
            raw.get("undetected_observations", 0.0),
            raw.get("detection_progress", 0.0),
            raw.get("total_observation_s", 0.0),
            raw.get("total_detections", 0.0),
            raw.get("storage_capacity_mb", 0.0),
            raw.get("jetson_capacity_mb", 0.0),
            raw.get("daily_downlink_budget_mb", 0.0),
            raw.get("achievable_downlink_mb", 0.0),
            1.0 if raw.get("health_status", "nominal") == "nominal" else 0.0,
        ],
        dtype=np.float32,
    )


def action_vector_from_env_action(env_actions: Dict[str, Any]) -> np.ndarray:
    sat_action = (env_actions or {}).get("eventsat_0", {})
    if not isinstance(sat_action, dict):
        sat_action = {}
    return action_from_mode(str(sat_action.get("mode", "charging")))


def mode_index(mode: str) -> int:
    return MODE_LIST.index(mode) if mode in MODE_LIST else MODE_LIST.index("charging")


@dataclass
class WorldModelTraceEpisode:
    """One EventSat episode converted to the world-model dataset schema."""

    episode_id: int
    seed: int
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
        encoded = eventsat_observation_to_vector(observation)
        requested_mode = str(info.get("requested_mode") or (env_actions.get("eventsat_0", {}) or {}).get("mode", "charging"))
        resolved_mode = str(info.get("resolved_mode", requested_mode))
        self.rows.append(
            {
                "step": int(step),
                "obs": encoded.obs25.astype(np.float32),
                "action": action_vector_from_env_action(env_actions),
                "state": state_vector_from_observation(observation),
                "reward": float(sum(rewards.values())) if rewards else 0.0,
                "mode": mode_index(requested_mode),
                "resolved_mode": mode_index(resolved_mode),
                "forced_mode": float(bool(info.get("forced", False))),
            }
        )

    def as_arrays(self) -> Dict[str, np.ndarray]:
        if not self.rows:
            return {
                "obs": np.zeros((0, len(OBS25_NAMES)), dtype=np.float32),
                "action": np.zeros((0, len(ACTION_NAMES)), dtype=np.float32),
                "state": np.zeros((0, len(STATE_NAMES)), dtype=np.float32),
                "reward": np.zeros((0,), dtype=np.float32),
                "mode": np.zeros((0,), dtype=np.int64),
                "resolved_mode": np.zeros((0,), dtype=np.int64),
                "forced_mode": np.zeros((0,), dtype=np.float32),
            }
        return {
            "obs": np.stack([r["obs"] for r in self.rows]).astype(np.float32),
            "action": np.stack([r["action"] for r in self.rows]).astype(np.float32),
            "state": np.stack([r["state"] for r in self.rows]).astype(np.float32),
            "reward": np.asarray([r["reward"] for r in self.rows], dtype=np.float32),
            "mode": np.asarray([r["mode"] for r in self.rows], dtype=np.int64),
            "resolved_mode": np.asarray([r["resolved_mode"] for r in self.rows], dtype=np.int64),
            "forced_mode": np.asarray([r["forced_mode"] for r in self.rows], dtype=np.float32),
        }

    def write_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **self.as_arrays(),
            episode_seed=np.asarray(self.seed, dtype=np.int64),
            episode_id=np.asarray(self.episode_id, dtype=np.int64),
        )


def write_trace_metadata(path: Path, payload: Dict[str, Any]) -> None:
    base = {
        "schema": "eventsat_world_model_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "obs_names": list(OBS25_NAMES),
        "action_names": list(ACTION_NAMES),
        "state_names": list(STATE_NAMES),
        "mode_names": list(MODE_LIST),
        "notes": "AUTOPS-native state only; thermal and pointing are absent unless the simulator is extended.",
    }
    base.update(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base, indent=2), encoding="utf-8")

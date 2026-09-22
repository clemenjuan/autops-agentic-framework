"""Shared EventSat RL observation and action contracts.

This is the **RL-specific** vectorisation of the general observation the env
publishes: it turns a satellite's resources/metadata into the fixed 25D float
vector a PPO policy consumes. Symbolic and LLM models do NOT use it -- hence the
``rl`` in the name.

Single source of truth: the RLlib space adapter (training) and
``SubsymbolicEventSat`` (inference) share both the observation encoder and the
controller-visible action grounding. The pure helpers take environment values
already resolved by the caller, so training and inference cannot drift in their
RL contract.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from src.rl import bound_observation_vector, bounded_ratio, downlink_utilization

# EventSat RL contract constants (single home; re-exported by space_adapters).
MODE_LIST = [
    "charging",
    "communication",
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "safe",
]
MODE_TO_IDX = {mode: idx for idx, mode in enumerate(MODE_LIST)}
OBS_DIM = 25
# One categorical operational-mode decision.  Keep this as a list so the
# shared RLlib machinery can add future categorical dimensions without
# changing the model/space contract.
ACTION_DIMS = [len(MODE_LIST)]
_DEFAULT_JETSON_CAPACITY_MB = 249036.8
_DEFAULT_MAX_PASS_STEPS = 10.0


def ground_eventsat_mode(
    mode: str,
    *,
    battery_soc: float,
    health_status: str,
    ground_pass_active: bool,
    battery_min_soc: float = 0.20,
) -> str:
    """Apply the controller-visible EventSat safety shield to one mode.

    Contact availability is an observation, not a command veto: a radio attempt
    without contact must reach the environment and incur its physical cost and
    failed-action reward. Keep the contact argument for caller compatibility.
    Training and evaluation retain the same health/battery protections.
    """
    if health_status != "nominal":
        return "safe"
    if battery_soc < battery_min_soc and mode != "charging":
        return "charging"
    return mode


def encode_eventsat_rl_obs(
    res: Mapping[str, Any],
    meta: Mapping[str, Any],
    status: str,
    *,
    obc_cap: float,
    jetson_cap: float,
    orbital_period: float,
    max_steps: float,
    compression_time: float,
    detection_steps: float,
    current_step: int,
    detection_progress: float,
) -> np.ndarray:
    """Build the normalised 25D EventSat RL observation vector.

    Constants (``obc_cap``, ``jetson_cap``, ``orbital_period``, ``max_steps``,
    ``compression_time``, ``detection_steps``) and the per-step ``current_step``
    / ``detection_progress`` are resolved by the caller; the vector math here is
    shared and identical for training and inference.
    """
    vec = np.zeros(OBS_DIM, dtype=np.float32)

    vec[0] = float(res.get("battery_soc", 0.5))
    vec[1] = bounded_ratio(res.get("obc_data_mb", meta.get("obc_data_mb", 0.0)), obc_cap)
    vec[2] = bounded_ratio(meta.get("jetson_raw_mb", 0.0), jetson_cap)
    vec[3] = bounded_ratio(meta.get("jetson_compressed_mb", 0.0), jetson_cap)

    orbital_phase = float(meta.get("orbital_phase", 0.0))
    vec[4] = math.sin(orbital_phase * 2 * math.pi)
    vec[5] = math.cos(orbital_phase * 2 * math.pi)

    op = orbital_period or 1.0
    vec[6] = min(float(meta.get("time_to_next_eclipse", orbital_period)) / op, 1.0)
    vec[7] = min(float(meta.get("time_to_next_pass", orbital_period)) / op, 1.0)
    vec[8] = min(float(meta.get("remaining_pass_duration", 0.0)) / _DEFAULT_MAX_PASS_STEPS, 1.0)
    vec[9] = float(current_step) / (max_steps or 1.0)

    vec[10] = 1.0 if meta.get("in_sunlight", False) else 0.0
    vec[11] = (
        1.0 if meta.get("contact_window_active", meta.get("ground_pass_active", False)) else 0.0
    )
    vec[12] = 1.0 if meta.get("health_status", "nominal") == "nominal" else 0.0

    vec[13] = min(float(meta.get("uncompressed_observations", 0)) / 10.0, 1.0)
    vec[14] = min(float(meta.get("compression_progress", 0)) / (compression_time or 1.0), 1.0)
    vec[15] = min(float(meta.get("undetected_observations", 0)) / 10.0, 1.0)
    vec[16] = min(float(detection_progress) / (detection_steps or 1.0), 1.0)
    vec[17] = downlink_utilization(res, meta, obc_cap)

    mode_idx = MODE_TO_IDX.get(str(status or "charging"), 0)
    vec[18 + mode_idx] = 1.0

    return bound_observation_vector(vec, signed_indices=(4, 5))

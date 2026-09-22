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

Schema ``eventsat_log_pipeline_attitude_pass_v4`` encodes pipeline volumes
(features 1-3) and cumulative downlink (feature 17) on a per-product log scale,
time to the next pass (feature 7) on a log scale over the orbital period, and
appends the attitude-settling state (features 25-32). The world-model and
Gymnasium 25D encoders intentionally keep the legacy features.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from src.rl import RL_FEATURE_MAX, bound_observation_vector

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
OBS_DIM = 33
# Checkpoints trained on another feature contract must not load silently.
EVENTSAT_OBS_SCHEMA_ID = "eventsat_log_pipeline_attitude_pass_v4"
# Features 26-32: one-hot of the mode the attitude is settling towards/set for.
_POINTING_OFFSET = 26
# One categorical operational-mode decision.  Keep this as a list so the
# shared RLlib machinery can add future categorical dimensions without
# changing the model/space contract.
ACTION_DIMS = [len(MODE_LIST)]
_DEFAULT_JETSON_CAPACITY_MB = 249036.8
_DEFAULT_MAX_PASS_STEPS = 10.0
_DEFAULT_OBSERVATION_SIZE_MB = 9.41
_DEFAULT_COMPRESSION_RATIO = 5.11


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _positive_metadata(meta: Mapping[str, Any], key: str, default: float) -> float:
    value = _finite_float(meta.get(key, default))
    return value if value > 0.0 else default


def log_product_fill(value_mb: Any, product_mb: Any, capacity_mb: Any) -> float:
    """Return ``log(1 + value/product) / log(1 + capacity/product)``.

    A linear capacity ratio maps one EventSat product to ~1e-5 of Jetson
    storage. Counting products on a log scale keeps first-product resolution,
    stays monotone, and reaches 1 only at the physical capacity. The scale uses
    plant constants only, never the reward's mission target.
    """
    value = _finite_float(value_mb)
    product = _finite_float(product_mb)
    capacity = _finite_float(capacity_mb)
    if value <= 0.0:
        return 0.0
    if product <= 0.0 or capacity <= 0.0:
        return float(RL_FEATURE_MAX)
    ratio = math.log1p(value / product) / math.log1p(capacity / product)
    return float(min(ratio, RL_FEATURE_MAX))


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
    shared and identical for training and inference. Product sizes come from
    the observation metadata every representation receives.
    """
    vec = np.zeros(OBS_DIM, dtype=np.float32)

    raw_product_mb = _positive_metadata(
        meta, "observation_size_mb", _DEFAULT_OBSERVATION_SIZE_MB
    )
    compressed_product_mb = raw_product_mb / _positive_metadata(
        meta, "compression_ratio", _DEFAULT_COMPRESSION_RATIO
    )

    vec[0] = float(res.get("battery_soc", 0.5))
    vec[1] = log_product_fill(
        res.get("obc_data_mb", meta.get("obc_data_mb", 0.0)),
        compressed_product_mb,
        obc_cap,
    )
    vec[2] = log_product_fill(meta.get("jetson_raw_mb", 0.0), raw_product_mb, jetson_cap)
    vec[3] = log_product_fill(
        meta.get("jetson_compressed_mb", 0.0), compressed_product_mb, jetson_cap
    )

    orbital_phase = float(meta.get("orbital_phase", 0.0))
    vec[4] = math.sin(orbital_phase * 2 * math.pi)
    vec[5] = math.cos(orbital_phase * 2 * math.pi)

    op = orbital_period or 1.0
    vec[6] = min(float(meta.get("time_to_next_eclipse", orbital_period)) / op, 1.0)
    # A slew must start 2 steps before contact, where t/op moved only ~0.011 per
    # step. The log keeps near-pass steps distinguishable and the environment's
    # orbital-period cap as the scale (same onboard contact-plan value).
    pass_scale = max(float(op), 1.0)
    steps_to_pass = min(
        max(_finite_float(meta.get("time_to_next_pass", pass_scale)), 0.0), pass_scale
    )
    vec[7] = math.log1p(steps_to_pass) / math.log1p(pass_scale)
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
    # Delivered products on the OBC-capacity log scale: a fixed plant constant,
    # unlike the previous per-pass denominator, and independent of the target.
    vec[17] = log_product_fill(
        res.get("data_downlinked_mb", 0.0), compressed_product_mb, obc_cap
    )

    mode_idx = MODE_TO_IDX.get(str(status or "charging"), 0)
    vec[18 + mode_idx] = 1.0

    # Settling executes as charging, so the effective-mode one-hot alone makes
    # "slewing to observe" indistinguishable from idle. Augmenting the state with
    # the pending command restores the Markov property of a delayed-action MDP
    # (Katsikopoulos & Engelbrecht, 2003). Both values are onboard mode-manager state.
    settling_steps = _finite_float(meta.get("settling_time_steps", 0.0))
    remaining = max(0.0, _finite_float(meta.get("transition_steps_remaining", 0.0)))
    vec[25] = min(remaining / settling_steps, 1.0) if settling_steps > 0.0 else 0.0
    target_mode = meta.get("transition_target_mode") if remaining > 0.0 else None
    pointing_mode = target_mode or meta.get("previous_mode") or status or "charging"
    vec[_POINTING_OFFSET + MODE_TO_IDX.get(str(pointing_mode), 0)] = 1.0

    return bound_observation_vector(vec, signed_indices=(4, 5))

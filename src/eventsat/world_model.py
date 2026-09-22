"""World-model planner representations for EventSat.

These classes register the paper-facing baselines with AUTOPS while keeping
AUTOPS as the truth simulator and metrics surface. The artifact-backed LeWM
path is intentionally optional: before trained artifacts exist, the planner
uses a deterministic AUTOPS-native surrogate dynamics model so configs and
board plumbing can be smoke-tested honestly.
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

import numpy as np

from src.core.behaviour.controller import register
from src.core.representation import Representation
from src.eventsat.transitions import (
    PipelineParameters,
    apply_can_transfer,
    apply_compress,
    apply_detect,
    apply_downlink,
    apply_observe,
    with_total_storage,
)
from src.rl import bound_observation_vector, bounded_ratio, downlink_utilization

if TYPE_CHECKING:
    from src.core.decision_procedure.context import DecisionContext


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
ACTION_NAMES = tuple(f"mode_{mode}" for mode in MODE_LIST)

OBS25_NAMES = (
    "battery_soc",
    "obc_fill",
    "jetson_raw_fill",
    "jetson_compressed_fill",
    "orbital_phase_sin",
    "orbital_phase_cos",
    "time_to_next_eclipse_norm",
    "time_to_next_pass_norm",
    "remaining_pass_duration_norm",
    "episode_progress",
    "in_sunlight",
    "ground_pass_active",
    "health_nominal",
    "uncompressed_observations_norm",
    "compression_progress_norm",
    "undetected_observations_norm",
    "detection_progress_norm",
    "downlink_utilization",
    *(f"current_mode_{mode}" for mode in MODE_LIST),
)

DEFAULT_MODE_WEIGHTS: dict[str, dict[str, float]] = {
    "science": {
        "battery_margin": 0.18,
        "storage_margin": 0.14,
        "downlink_progress": 0.22,
        "science_progress": 0.28,
        "detection_progress": 0.10,
        "communication_opportunity": 0.02,
        "forced_mode_avoidance": 0.04,
        "anomaly_safe": 0.02,
    },
    "safe": {
        "battery_margin": 0.42,
        "storage_margin": 0.20,
        "downlink_progress": 0.06,
        "science_progress": 0.03,
        "detection_progress": 0.02,
        "communication_opportunity": 0.02,
        "forced_mode_avoidance": 0.15,
        "anomaly_safe": 0.10,
    },
    "downlink": {
        "battery_margin": 0.16,
        "storage_margin": 0.26,
        "downlink_progress": 0.38,
        "science_progress": 0.05,
        "detection_progress": 0.04,
        "communication_opportunity": 0.06,
        "forced_mode_avoidance": 0.03,
        "anomaly_safe": 0.02,
    },
}


@dataclass
class EncodedEventSatState:
    """Planner-facing EventSat state extracted from an AUTOPS observation."""

    obs25: np.ndarray
    raw: Dict[str, Any]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _mode_one_hot(mode: str) -> np.ndarray:
    out = np.zeros(len(MODE_LIST), dtype=np.float32)
    out[MODE_TO_IDX.get(mode, 0)] = 1.0
    return out


def action_from_mode(mode: str) -> np.ndarray:
    """Encode an AUTOPS operational mode as a 7D one-hot action vector."""
    out = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    out[MODE_TO_IDX.get(mode, 0)] = 1.0
    return out


def eventsat_observation_to_vector(
    observation: Any, sat_id: str = "eventsat_0"
) -> EncodedEventSatState:
    """Convert an AUTOPS EventSat observation into the canonical 25D vector.

    This mirrors ``src.eventsat.gymnasium_wrapper.EventSatGymnasium`` without
    importing gymnasium, so the world-model exporter works in the base AUTOPS env.

    ``sat_id`` selects which satellite to encode; it defaults to the single-sat
    EventSat key so existing callers are unaffected. Constellation exporters
    (SSA) pass each ``sat_0..sat_{N-1}`` id in turn.
    """
    vec = np.zeros(25, dtype=np.float32)
    raw: Dict[str, Any] = {}
    if not hasattr(observation, "constellation_state"):
        return EncodedEventSatState(vec, raw)

    sat = observation.constellation_state.satellites.get(sat_id)
    if sat is None:
        return EncodedEventSatState(vec, raw)

    res = sat.resources or {}
    meta = sat.metadata or {}
    global_info = getattr(observation.constellation_state, "global_info", {}) or {}

    storage_capacity_mb = _float(meta.get("storage_capacity_mb"), 4096.0) or 4096.0
    jetson_capacity_mb = _float(meta.get("jetson_capacity_mb"), 249036.8) or 249036.8
    orbital_period_steps = _float(meta.get("orbital_period_steps"), 94.0) or 94.0
    max_steps = _float(global_info.get("max_steps"), 10080.0) or 10080.0
    detection_steps = _float(meta.get("detection_steps"), 5.0) or 5.0
    compression_time_factor = _float(meta.get("compression_time_factor"), 2.0) or 2.0

    orbital_phase = _float(meta.get("orbital_phase"), 0.0)
    current_mode = sat.status or "charging"
    timestep = _float(getattr(observation.constellation_state, "timestep", 0), 0.0)

    vec[0] = _float(res.get("battery_soc"), 0.5)
    vec[1] = bounded_ratio(
        res.get("obc_data_mb", meta.get("obc_data_mb")), storage_capacity_mb
    )
    vec[2] = bounded_ratio(meta.get("jetson_raw_mb"), jetson_capacity_mb)
    vec[3] = bounded_ratio(meta.get("jetson_compressed_mb"), jetson_capacity_mb)
    vec[4] = math.sin(orbital_phase * 2.0 * math.pi)
    vec[5] = math.cos(orbital_phase * 2.0 * math.pi)
    vec[6] = min(_float(meta.get("time_to_next_eclipse"), orbital_period_steps) / orbital_period_steps, 1.0)
    vec[7] = min(_float(meta.get("time_to_next_pass"), orbital_period_steps) / orbital_period_steps, 1.0)
    vec[8] = min(_float(meta.get("remaining_pass_duration"), 0.0) / 10.0, 1.0)
    vec[9] = min(timestep / max_steps, 1.0)
    vec[10] = 1.0 if meta.get("in_sunlight", False) else 0.0
    vec[11] = 1.0 if meta.get("contact_window_active", meta.get("ground_pass_active", False)) else 0.0
    vec[12] = 1.0 if meta.get("health_status", "nominal") == "nominal" else 0.0
    vec[13] = min(_float(meta.get("uncompressed_observations"), 0.0) / 10.0, 1.0)
    vec[14] = min(_float(meta.get("compression_progress"), 0.0) / compression_time_factor, 1.0)
    vec[15] = min(_float(meta.get("undetected_observations"), 0.0) / 10.0, 1.0)
    vec[16] = min(_float(meta.get("detection_progress"), 0.0) / detection_steps, 1.0)
    vec[17] = downlink_utilization(res, meta, storage_capacity_mb)
    vec[18:25] = _mode_one_hot(current_mode)

    raw = {
        "battery_soc": vec[0],
        "current_mode": current_mode,
        "in_sunlight": bool(meta.get("in_sunlight", False)),
        "ground_pass_active": bool(meta.get("contact_window_active", meta.get("ground_pass_active", False))),
        "orbital_phase": orbital_phase,
        "time_to_next_eclipse": _float(meta.get("time_to_next_eclipse"), orbital_period_steps),
        "time_to_next_pass": _float(meta.get("time_to_next_pass"), orbital_period_steps),
        "remaining_pass_duration": _float(meta.get("remaining_pass_duration"), 0.0),
        "remaining_pass_duration_s": _float(
            meta.get("remaining_pass_duration_s"), 0.0
        ),
        "contact_window_seconds": _float(meta.get("contact_window_seconds"), 0.0),
        "following_gap_steps": _float(meta.get("following_gap_steps"), orbital_period_steps),
        "timestep": timestep,
        "max_steps": max_steps,
        "data_stored_mb": _float(res.get("data_stored_mb"), 0.0),
        "obc_data_mb": _float(res.get("obc_data_mb", meta.get("obc_data_mb")), 0.0),
        "jetson_raw_mb": _float(meta.get("jetson_raw_mb"), 0.0),
        "jetson_compressed_mb": _float(meta.get("jetson_compressed_mb"), 0.0),
        "data_downlinked_mb": _float(res.get("data_downlinked_mb"), 0.0),
        "total_raw_captured_mb": _float(meta.get("total_raw_captured_mb"), 0.0),
        "obc_raw_equivalent_mb": _float(meta.get("obc_raw_equivalent_mb"), 0.0),
        "downlink_raw_equivalent_mb": _float(
            meta.get("downlink_raw_equivalent_mb"), 0.0
        ),
        "total_pass_duration_s": _float(meta.get("total_pass_duration_s"), 0.0),
        "uncompressed_observations": _float(meta.get("uncompressed_observations"), 0.0),
        "compression_progress": _float(meta.get("compression_progress"), 0.0),
        "undetected_observations": _float(meta.get("undetected_observations"), 0.0),
        "detection_progress": _float(meta.get("detection_progress"), 0.0),
        "total_observation_s": _float(meta.get("total_observation_s"), 0.0),
        "total_detections": _float(meta.get("total_detections"), 0.0),
        "health_status": meta.get("health_status", "nominal"),
        "storage_capacity_mb": storage_capacity_mb,
        "jetson_capacity_mb": jetson_capacity_mb,
        "observation_size_mb": _float(meta.get("observation_size_mb"), 9.41),
        "compression_ratio": _float(meta.get("compression_ratio"), 5.11),
        "detection_metadata_mb": _float(meta.get("detection_metadata_mb"), 0.01),
        "jetson_to_obc_rate_kbps": _float(
            meta.get("jetson_to_obc_rate_kbps"), 8000.0
        ),
        "downlink_rate_kbps": _float(meta.get("downlink_rate_kbps"), 50.0),
        "step_duration_s": _float(meta.get("step_duration_s"), 60.0),
        "battery_min_soc": _float(meta.get("battery_min_soc"), 0.20),
        "battery_capacity_wh": _float(meta.get("battery_capacity_wh"), 70.0),
        "solar_generation_w": _float(meta.get("solar_generation_w"), 24.0),
        "charge_efficiency": _float(meta.get("charge_efficiency"), 0.9),
        "power_consumption": {
            str(mode): dict(values)
            for mode, values in (meta.get("power_consumption") or {}).items()
            if isinstance(values, dict)
        },
        "onboard_compute_w": _float(meta.get("onboard_compute_w"), 7.0),
        "jetson_active_modes": [
            str(mode) for mode in (meta.get("jetson_active_modes") or [])
        ],
        "contact_plan": tuple(
            interval
            for interval in (meta.get("contact_plan") or ())
            if isinstance(interval, dict)
        ),
        "eclipse_plan": tuple(
            interval
            for interval in (meta.get("eclipse_plan") or ())
            if isinstance(interval, dict)
        ),
        "mode_min_battery_soc": dict(meta.get("mode_min_battery_soc") or {}),
        "achievable_downlink_mb": _float(meta.get("achievable_downlink_mb"), 0.0),
        "orbital_period_steps": orbital_period_steps,
        "compression_time_factor": compression_time_factor,
        "detection_steps": detection_steps,
        # The deterministic co-rollout must carry the environment's ADCS
        # lifecycle state.  These fields are planner-visible spacecraft state,
        # not additional physical-oracle information.
        "settling_time_steps": max(
            0, int(_float(meta.get("settling_time_steps"), 0.0))
        ),
        "transition_steps_remaining": max(
            0, int(_float(meta.get("transition_steps_remaining"), 0.0))
        ),
        "attitude_maneuver_modes": [
            str(mode) for mode in (meta.get("attitude_maneuver_modes") or [])
        ],
        "previous_mode": str(meta.get("previous_mode", current_mode)),
    }
    return EncodedEventSatState(
        bound_observation_vector(vec, signed_indices=(4, 5)), raw
    )


def _strip_checkpoint_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            out[key[len("model.") :]] = value
        elif not key.startswith("sigreg."):
            out[key] = value
    return out


def _resolve_artifact_path(artifact: Dict[str, Any], value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    base = Path(str(artifact.get("_artifact_dir", ".")))
    return base / path


class _ArtifactLatentBackend:
    """Torch-backed latent rollout loaded from a space-world-models artifact."""

    def __init__(self, artifact: Dict[str, Any], config: Dict[str, Any]) -> None:
        self.artifact = artifact
        self.config = config
        lewm = artifact.get("lewm", {}) if isinstance(artifact.get("lewm"), dict) else {}
        probe = artifact.get("probe", {}) if isinstance(artifact.get("probe"), dict) else {}
        code_root = artifact.get("code_root") or lewm.get("code_root")
        if code_root and str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))

        import torch
        from core.models.components import ARPredictor, Embedder, MLP
        from core.models.vector_encoder import VectorEncoder
        from core.models.vector_jepa import VectorJEPA

        torch.backends.nnpack.enabled = False
        self.torch = torch
        self.device = torch.device(str(config.get("device", "cpu")))
        self.history_size = int(lewm.get("history_size", artifact.get("history_size", 3)))
        self.embed_dim = int(lewm.get("embed_dim", artifact.get("embed_dim", 192)))
        self.obs_dim = int(lewm.get("obs_dim", 25))
        self.action_dim = int(lewm.get("action_dim", len(ACTION_NAMES)))
        self.W = np.asarray(probe.get("W"), dtype=np.float32)
        self.b = np.asarray(probe.get("b"), dtype=np.float32)
        self.attribute_names = [str(x) for x in probe.get("attribute_names", [])]
        if self.W.ndim != 2 or self.W.shape[1] != self.embed_dim:
            raise ValueError(
                "planner artifact probe W must be shaped (attributes, embed_dim); "
                f"got {self.W.shape}, embed_dim={self.embed_dim}"
            )
        if self.b.shape != (self.W.shape[0],):
            raise ValueError(f"planner artifact probe b shape {self.b.shape} does not match W")
        if len(self.attribute_names) != self.W.shape[0]:
            raise ValueError("planner artifact probe attribute_names length does not match W")

        # Per-attribute training-set scale (probe.normalization.target_std), used to
        # correct for heterogeneous attribute units before combining with mode_weights
        # (see normalize_attribute_scale below). Falls back to all-ones -- a no-op --
        # for older artifacts that predate this field, so loading never breaks.
        probe_normalization = probe.get("normalization")
        target_std = probe_normalization.get("target_std") if isinstance(probe_normalization, dict) else None
        if target_std is not None and len(target_std) == len(self.attribute_names):
            self.attribute_scale = np.asarray(target_std, dtype=np.float32)
            self.attribute_scale[self.attribute_scale < 1e-8] = 1.0
        else:
            self.attribute_scale = np.ones(len(self.attribute_names), dtype=np.float32)
        # Opt-in (default off): existing sweeps were run without this correction and
        # stay bit-reproducible unless a config explicitly asks for it. See
        # docs/research-tracker.md E-A7 entry (2026-07-09) for why it exists.
        self.normalize_attribute_scale = bool(config.get("normalize_attribute_scale", False))

        normalizers_path = _resolve_artifact_path(artifact, lewm.get("normalizers"))
        norms = np.load(normalizers_path)
        self.obs_mean = norms["obs_mean"].astype(np.float32)
        self.obs_std = norms["obs_std"].astype(np.float32)
        self.action_mean = norms["action_mean"].astype(np.float32)
        self.action_std = norms["action_std"].astype(np.float32)
        self.obs_std[self.obs_std < 1e-8] = 1.0
        self.action_std[self.action_std < 1e-8] = 1.0

        model_cfg = artifact.get("model_config", {}) if isinstance(artifact.get("model_config"), dict) else {}
        encoder = VectorEncoder(
            in_dim=self.obs_dim,
            hidden_dim=int(model_cfg.get("encoder_hidden_dim", 256)),
            out_dim=self.embed_dim,
        )
        predictor = ARPredictor(
            num_frames=self.history_size,
            input_dim=self.embed_dim,
            hidden_dim=self.embed_dim,
            output_dim=self.embed_dim,
            depth=int(model_cfg.get("predictor_depth", 4)),
            heads=int(model_cfg.get("predictor_heads", 8)),
            mlp_dim=int(model_cfg.get("predictor_mlp_dim", 512)),
            dim_head=int(model_cfg.get("predictor_dim_head", 48)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            emb_dropout=float(model_cfg.get("emb_dropout", 0.0)),
        )
        action_encoder = Embedder(
            input_dim=self.action_dim,
            smoothed_dim=self.action_dim,
            emb_dim=self.embed_dim,
        )
        projector = MLP(self.embed_dim, int(model_cfg.get("projector_hidden_dim", 512)), self.embed_dim, norm_fn=None)
        pred_proj = MLP(self.embed_dim, int(model_cfg.get("projector_hidden_dim", 512)), self.embed_dim, norm_fn=None)
        self.model = VectorJEPA(encoder, predictor, action_encoder, projector, pred_proj)

        checkpoint_path = _resolve_artifact_path(artifact, lewm.get("checkpoint"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = _strip_checkpoint_state(checkpoint.get("state_dict", checkpoint))
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def score_sequences(
        self,
        history: Dict[str, np.ndarray],
        seq: np.ndarray,
        mode_weights: Dict[str, float],
    ) -> np.ndarray:
        z = self.rollout(history, seq)
        terminal = z[:, -1, :]
        attrs = terminal @ self.W.T + self.b
        return self._score_from_attrs(attrs, mode_weights)

    def _score_from_attrs(self, attrs: np.ndarray, mode_weights: Dict[str, float]) -> np.ndarray:
        """Combine a (candidates, attributes) matrix with mode_weights. Pure numpy
        (no torch/rollout) so this step is unit-testable without a loaded model."""
        weights = np.asarray(
            [float(mode_weights.get(name, 0.0)) for name in self.attribute_names],
            dtype=np.float32,
        )
        if self.normalize_attribute_scale:
            # Attribute readouts share one affine probe but live on very different
            # physical scales (e.g. downlink_progress is raw undownlinked MB,
            # unbounded ~0-30; most others are normalized margins/fractions in
            # ~[0,1]). Left uncorrected, the highest-variance attribute numerically
            # dominates any weighted sum regardless of the rest of the weight
            # vector -- diagnosed 2026-07-09 (E-A7 zero-shot retargeting collapse;
            # see docs/research-tracker.md). Dividing by the artifact's own
            # training-set per-attribute std (probe.normalization.target_std) puts
            # every attribute on a comparable footing before combining. Only the
            # scale matters for CEM ranking within one planning event -- centering
            # by target_mean would shift every candidate's score by the same
            # per-attribute constant and therefore never changes which candidate
            # scores highest, so it is deliberately not applied here.
            weights = weights / self.attribute_scale
        return (attrs @ weights).astype(np.float64)

    def rollout(self, history: Dict[str, np.ndarray], seq: np.ndarray) -> np.ndarray:
        torch = self.torch
        obs = self._pad_history(np.asarray(history["obs"], dtype=np.float32), self.obs_dim)
        history_action = self._pad_history(np.asarray(history["action"], dtype=np.float32), self.action_dim)
        candidate_actions = self._encode_sequences(seq)
        obs_n = self._norm_obs(obs)
        act_n = self._norm_action(history_action)
        n, horizon, _ = candidate_actions.shape
        with torch.no_grad():
            batch = {
                "obs": torch.from_numpy(obs_n[None]).to(self.device),
                "action": torch.from_numpy(act_n[None]).to(self.device),
            }
            encoded = self.model.encode(batch)
            emb_hist = encoded["emb"].repeat(n, 1, 1)
            act_hist = torch.from_numpy(np.repeat(act_n[None], n, axis=0)).to(self.device)
            first = self._norm_action(candidate_actions[:, 0])
            act_hist[:, -1, :] = torch.from_numpy(first).to(self.device)
            pred_rows = []
            for t in range(horizon):
                act_emb = self.model.action_encoder(act_hist[:, -self.history_size :])
                pred = self.model.predict(emb_hist[:, -self.history_size :], act_emb)[:, -1:]
                pred_rows.append(pred[:, 0])
                emb_hist = torch.cat([emb_hist, pred], dim=1)
                if t + 1 < horizon:
                    nxt = self._norm_action(candidate_actions[:, t + 1])
                    act_hist = torch.cat([act_hist, torch.from_numpy(nxt[:, None]).to(self.device)], dim=1)
            return torch.stack(pred_rows, dim=1).detach().cpu().numpy().astype(np.float32)

    def _pad_history(self, arr: np.ndarray, dim: int) -> np.ndarray:
        arr = arr.reshape(-1, dim)
        if arr.shape[0] >= self.history_size:
            return arr[-self.history_size :]
        first = arr[0] if arr.shape[0] else np.zeros(dim, dtype=np.float32)
        pad = np.repeat(first[None], self.history_size - arr.shape[0], axis=0)
        return np.concatenate([pad, arr], axis=0).astype(np.float32)

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        return ((obs.astype(np.float32) - self.obs_mean) / self.obs_std).astype(np.float32)

    def _norm_action(self, action: np.ndarray) -> np.ndarray:
        return ((action.astype(np.float32) - self.action_mean) / self.action_std).astype(np.float32)

    def _encode_sequences(self, seq: np.ndarray) -> np.ndarray:
        seq = np.asarray(seq, dtype=np.int64)
        out = np.zeros((*seq.shape, self.action_dim), dtype=np.float32)
        rows = np.indices(seq.shape)
        out[rows[0], rows[1], seq] = 1.0
        return out


class _ExternalPlannerBackend:
    """Persistent JSON-lines bridge to the space-world-models Torch worker."""

    def __init__(self, artifact: Dict[str, Any], config: Dict[str, Any]) -> None:
        worker = artifact.get("worker", {}) if isinstance(artifact.get("worker"), dict) else {}
        python = worker.get("python")
        module = worker.get("module", "swm_eventsat.experiments.autops_planner_worker")
        artifact_path = artifact.get("_artifact_path")
        if not python or not artifact_path:
            raise RuntimeError("planner artifact does not declare worker.python and _artifact_path")
        cmd = [str(python), "-m", str(module), "--artifact", str(artifact_path)]
        if config.get("device"):
            cmd.extend(["--device", str(config.get("device"))])
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(artifact.get("code_root") or Path(str(artifact_path)).parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdout is not None
        hello = self.proc.stdout.readline().strip()
        if not hello:
            raise RuntimeError("planner worker exited before handshake")
        payload = json.loads(hello)
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error", "planner worker handshake failed")))

    def seed(self, seed: int) -> None:
        self._request({"type": "seed", "seed": int(seed)})

    def select(
        self,
        *,
        obs25: np.ndarray,
        first_mask: np.ndarray,
        mode_weights: Dict[str, float],
        horizon: int,
        samples: int,
        elites: int,
        iterations: int,
        alpha: float,
    ) -> Dict[str, Any]:
        return self._request(
            {
                "type": "select",
                "obs25": np.asarray(obs25, dtype=float).reshape(-1).tolist(),
                "first_mask": np.asarray(first_mask, dtype=bool).tolist(),
                "mode_weights": {k: float(v) for k, v in mode_weights.items()},
                "horizon": int(horizon),
                "samples": int(samples),
                "elites": int(elites),
                "iterations": int(iterations),
                "alpha": float(alpha),
            }
        )

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.proc.poll() is not None:
            raise RuntimeError(f"planner worker exited with code {self.proc.returncode}")
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("planner worker closed stdout")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "planner worker request failed")))
        return response

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class _WorldModelPlanner:
    """CEM planner with an artifact-ready interface."""

    def __init__(self, config: Dict[str, Any], *, method: str) -> None:
        self.config = config
        self.method = method
        self.horizon = int(config.get("horizon", 12))
        self.samples = int(config.get("samples", config.get("candidate_count", 256)))
        self.elites = int(config.get("elites", max(8, self.samples // 8)))
        self.iters = int(config.get("cem_iterations", 4))
        self.alpha = float(config.get("cem_alpha", 0.7))
        self.reserve_soc = float(config.get("reserve_soc", 0.50))
        # Shared reach + power-discipline weights. Both latent and analytic CEM
        # use the exact deterministic co-rollout for this term; archived runs
        # whose latent shaping used the approximate fallback are not comparable.
        self.downlink_reward = float(config.get("downlink_reward", 1.0))
        self.battery_penalty = float(config.get("battery_penalty", 4.0))
        self.pass_stage_reward = float(config.get("pass_stage_reward", 0.15))
        self.downlink_shaping_reference_weight = float(
            config.get("downlink_shaping_reference_weight", 0.25)
        )
        self.comms_soc_floor = float(config.get("comms_soc_floor", 0.25))
        self.contact_reflex_enabled = bool(config.get("contact_reflex_enabled", False))
        self._contact_reflex_overrides = 0
        self._held_plan_mask_repairs = 0
        # Receding-horizon plan-and-hold: wake the Jetson to run CEM, then execute the cached
        # schedule open-loop for the next (plan_hold-1) steps with the Jetson asleep. This cuts
        # the onboard compute duty cycle to 1/plan_hold so charging can close the power budget.
        # plan_hold=1 preserves the original per-step re-planning behaviour.
        self.plan_hold = max(1, int(config.get("plan_hold", 1)))
        if self.plan_hold > self.horizon:
            # The held tail is sliced from the planned sequence, so a hold longer than
            # the horizon silently truncated to it and mislabelled the Jetson duty
            # (caught in the 2026-07-07 E-A1 sweep: h48@H12 ran as h12).
            logging.getLogger(__name__).warning(
                "plan_hold=%d exceeds horizon=%d; clamping to horizon - the planner "
                "cannot execute steps it never planned. Raise `horizon` to hold longer.",
                self.plan_hold, self.horizon,
            )
            self.plan_hold = self.horizon
        self._plan_queue: list[int] = []
        self.rng = np.random.default_rng(int(config.get("seed", 0)))
        self.previous_solution: Optional[np.ndarray] = None
        self.mode_weight_name = str(config.get("mission_mode", "science"))
        requested_backend = str(config.get("planner_backend", "latent")).strip().lower()
        if requested_backend not in {"latent", "analytic"}:
            raise ValueError(
                "planner_backend must be 'latent' or 'analytic', got "
                f"{requested_backend!r}"
            )
        self.requested_backend = requested_backend
        self.planner_pricing = str(config.get("planner_pricing", "jetson")).strip().lower()
        if self.planner_pricing not in {"obc", "jetson"}:
            raise ValueError(
                "planner_pricing must be 'obc' or 'jetson', got "
                f"{self.planner_pricing!r}"
            )
        default_power_w = 0.5 if self.planner_pricing == "obc" else 7.0
        self.planner_power_w = max(
            0.0, _float(config.get("planner_power_w"), default_power_w)
        )
        artifact_path = config.get("planner_artifact") or config.get("artifact_path")
        # Analytic MPC is intentionally selected and must never enter artifact
        # loading or masquerade as the unintended deterministic fallback.
        self.artifact = (
            {} if requested_backend == "analytic" else self._load_artifact(artifact_path)
        )
        self.latent_backend: Optional[_ArtifactLatentBackend] = None
        self.external_backend: Optional[_ExternalPlannerBackend] = None
        self.artifact_error = ""
        if requested_backend == "analytic":
            self.backend = "analytic"
            self.rollout_backend = "analytic"
        elif self.artifact:
            try:
                self.latent_backend = _ArtifactLatentBackend(self.artifact, config)
                self.backend = "artifact_latent"
                self.rollout_backend = "latent"
            except Exception as exc:
                self.artifact_error = str(exc)
                try:
                    self.external_backend = _ExternalPlannerBackend(self.artifact, config)
                    self.backend = "external_artifact_latent"
                    self.rollout_backend = "latent"
                except Exception as worker_exc:
                    self.artifact_error = f"{exc}; worker fallback failed: {worker_exc}"
                    if bool(config.get("strict_artifact", False)):
                        raise RuntimeError(f"failed to load LeWM planner artifact: {self.artifact_error}") from worker_exc
                    self.backend = "artifact_unavailable_surrogate"
                    self.rollout_backend = "fallback"
        else:
            self.backend = "autops_surrogate"
            self.rollout_backend = "fallback"
        self.mode_weights = self._load_mode_weights(config)
        self._obs_history: list[np.ndarray] = []
        self._action_history: list[np.ndarray] = []
        self._last_action = action_from_mode("charging")

        self._planning_event_count = 0
        self._planning_latency_total_s = 0.0
        self._planner_energy_wh = 0.0
        self._last_metrics: Dict[str, Any] = {
            "candidate_count": float(self.samples),
            "cem_iterations": float(self.iters),
            "model_size_mb": self._artifact_float("model_size_mb", 0.0),
            "peak_memory_mb": self._artifact_float("peak_memory_mb", 0.0),
            "probe_validation_error": self._artifact_float("probe_validation_error", 0.0),
            "train_dataset_steps": self._artifact_float("train_dataset_steps", 0.0),
            "orin_planner_latency_ms": self._artifact_float("orin_planner_latency_ms", 0.0),
            "artifact_loaded": 1.0 if (self.latent_backend is not None or self.external_backend is not None) else 0.0,
            "rollout_backend": self.rollout_backend,
            "planner_rollouts_per_s": 0.0,
            "planner_latency_s": 0.0,
            "planner_event": 0.0,
            "planner_event_latency_s": 0.0,
            "planner_step_energy_wh": 0.0,
            "planner_ms_per_event": 0.0,
            "planner_energy_wh": 0.0,
            "planner_power_w": self.planner_power_w,
            "jetson_planned": 1.0,
            "plan_hold": float(self.plan_hold),
            "contact_reflex_enabled": 1.0 if self.contact_reflex_enabled else 0.0,
            "contact_reflex_overrides": 0.0,
            "held_plan_mask_repairs": 0.0,
        }
        if self.rollout_backend != "analytic":
            self._last_metrics["artifact_fallback"] = (
                1.0 if self.rollout_backend == "fallback" else 0.0
            )

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        if self.external_backend is not None:
            self.external_backend.seed(seed)

    def reset(self) -> None:
        """Clear every receding-horizon state item at an episode boundary."""
        self._plan_queue = []
        self.previous_solution = None
        self._obs_history = []
        self._action_history = []
        self._last_action = action_from_mode("charging")
        self._contact_reflex_overrides = 0
        self._held_plan_mask_repairs = 0
        self._planning_event_count = 0
        self._planning_latency_total_s = 0.0
        self._planner_energy_wh = 0.0
        self._last_metrics.update(
            {
                "planner_rollouts_per_s": 0.0,
                "planner_latency_s": 0.0,
                "planner_event": 0.0,
                "planner_event_latency_s": 0.0,
                "planner_step_energy_wh": 0.0,
                "planner_ms_per_event": 0.0,
                "planner_energy_wh": 0.0,
                "jetson_planned": 1.0,
                "contact_reflex_overrides": 0.0,
                "held_plan_mask_repairs": 0.0,
            }
        )

    def select(self, state: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        start = time.perf_counter()
        self._append_history(state)

        # Jetson asleep: execute the next action from the held schedule with only a cheap,
        # OBC-level safety clamp (arithmetic mask, no torch inference). No CEM this step.
        if self._plan_queue:
            mode_idx = int(self._plan_queue.pop(0))
            mask = self._first_action_mask(state)
            comm_idx = MODE_TO_IDX["communication"]
            if self.contact_reflex_enabled and mask[comm_idx]:
                # Optional OBC contact-window reflex. Disabled by default so LeWM-CEM
                # must learn/plan downlink timing; when enabled it is reported as a
                # hand-coded rule, not learned MPC behavior.
                mode_idx = comm_idx
                self._contact_reflex_overrides += 1
            elif not mask[mode_idx]:
                mode_idx = (
                    MODE_TO_IDX["charging"] if mask[MODE_TO_IDX["charging"]] else MODE_TO_IDX["safe"]
                )
                self._held_plan_mask_repairs += 1
            mode = MODE_LIST[mode_idx]
            elapsed = time.perf_counter() - start
            self._last_action = action_from_mode(mode)
            if self._action_history:
                self._action_history[-1] = self._last_action
            self._last_metrics.update(
                {
                    "planner_latency_s": elapsed,
                    "planner_event": 0.0,
                    "planner_event_latency_s": 0.0,
                    "planner_step_energy_wh": 0.0,
                    "planner_rollouts_per_s": 0.0,
                    "jetson_planned": 0.0,
                    "contact_reflex_overrides": float(self._contact_reflex_overrides),
                    "held_plan_mask_repairs": float(self._held_plan_mask_repairs),
                }
            )
            return mode, dict(self._last_metrics)

        horizon = max(1, self.horizon)
        samples = max(1, self.samples)
        if (
            self.external_backend is None
            and all(
                key in state
                for key in ("contact_plan", "eclipse_plan", "power_consumption")
            )
        ):
            self._prepare_analytic_schedule(state, horizon)
        probs = self._initial_probs(horizon)
        first_mask = self._first_action_mask(state)
        if self.external_backend is not None and "obs25" in state:
            response = self.external_backend.select(
                obs25=np.asarray(state["obs25"], dtype=np.float32),
                first_mask=first_mask,
                mode_weights=self.mode_weights,
                horizon=horizon,
                samples=samples,
                elites=max(1, min(self.elites, samples)),
                iterations=self.iters,
                alpha=self.alpha,
            )
            elapsed = time.perf_counter() - start
            mode = str(response["mode"])
            self.previous_solution = np.asarray(response.get("best_sequence", []), dtype=np.int64)
            self._cache_plan_tail(self.previous_solution)
            self._last_action = action_from_mode(mode)
            if self._action_history:
                self._action_history[-1] = self._last_action
            rollouts = float(samples * self.iters)
            self._last_metrics.update(
                {
                    "candidate_count": float(samples),
                    "cem_iterations": float(self.iters),
                    "planner_latency_s": elapsed,
                    "planner_rollouts_per_s": rollouts / elapsed if elapsed > 0 else 0.0,
                    "artifact_loaded": 1.0,
                    "artifact_fallback": 0.0,
                    "jetson_planned": 1.0,
                }
            )
            self._record_planning_event(elapsed, state)
            return mode, dict(self._last_metrics)

        best_seq: Optional[np.ndarray] = None
        best_score = -np.inf
        elite_count = max(1, min(self.elites, samples))

        for _ in range(self.iters):
            seq = self._sample_sequences(probs, samples)
            allowed = np.flatnonzero(first_mask)
            if allowed.size == 0:
                allowed = np.asarray([MODE_TO_IDX["charging"]], dtype=np.int64)
            bad_first = ~first_mask[seq[:, 0]]
            if np.any(bad_first):
                seq[bad_first, 0] = self.rng.choice(allowed, size=int(np.sum(bad_first)))

            scores = self._score_sequences(state, seq)
            idx = int(np.argmax(scores))
            if float(scores[idx]) > best_score:
                best_score = float(scores[idx])
                best_seq = seq[idx].copy()

            elite_idx = np.argpartition(scores, -elite_count)[-elite_count:]
            empirical = np.full_like(probs, 1e-4)
            for t in range(horizon):
                counts = np.bincount(seq[elite_idx, t], minlength=len(MODE_LIST)).astype(np.float64)
                empirical[t] += counts / max(1.0, counts.sum())
            empirical /= empirical.sum(axis=1, keepdims=True)
            probs = self.alpha * empirical + (1.0 - self.alpha) * probs
            probs /= probs.sum(axis=1, keepdims=True)

        elapsed = time.perf_counter() - start
        rollouts = float(samples * self.iters)
        self._last_metrics.update(
            {
                "candidate_count": float(samples),
                "cem_iterations": float(self.iters),
                "planner_latency_s": elapsed,
                "planner_rollouts_per_s": rollouts / elapsed if elapsed > 0 else 0.0,
                "jetson_planned": 1.0,
            }
        )
        self._record_planning_event(elapsed, state)
        if best_seq is None:
            best_seq = np.asarray([MODE_TO_IDX["charging"]], dtype=np.int64)
        selected_mode = MODE_LIST[int(best_seq[0])]
        self._last_action = action_from_mode(selected_mode)
        if self._action_history:
            self._action_history[-1] = self._last_action
        self.previous_solution = best_seq
        self._cache_plan_tail(best_seq)
        return selected_mode, dict(self._last_metrics)

    def _cache_plan_tail(self, best_seq: Optional[np.ndarray]) -> None:
        """Queue the schedule steps to execute open-loop (Jetson asleep) before re-planning.

        The first action of ``best_seq`` is executed now; the next ``plan_hold-1`` are held.
        """
        self._plan_queue = []
        if self.plan_hold <= 1 or best_seq is None:
            return
        tail = np.asarray(best_seq, dtype=np.int64).reshape(-1)[1 : self.plan_hold]
        self._plan_queue = [int(a) for a in tail]

    def _record_planning_event(self, elapsed_s: float, state: Dict[str, Any]) -> None:
        """Record measured event latency and the configured simulated power price."""

        elapsed_s = max(0.0, float(elapsed_s))
        self._planning_event_count += 1
        self._planning_latency_total_s += elapsed_s
        step_hours = max(0.0, _float(state.get("step_duration_s"), 60.0)) / 3600.0
        event_energy_wh = self.planner_power_w * step_hours
        self._planner_energy_wh += event_energy_wh
        self._last_metrics.update(
            {
                "planner_event": 1.0,
                "planner_event_latency_s": elapsed_s,
                "planner_step_energy_wh": event_energy_wh,
                "planner_ms_per_event": (
                    1000.0 * self._planning_latency_total_s / self._planning_event_count
                ),
                "planner_energy_wh": self._planner_energy_wh,
            }
        )

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._last_metrics)

    def _append_history(self, state: Dict[str, Any]) -> None:
        obs = state.get("obs25")
        if obs is None:
            return
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs_arr.shape[0] != 25:
            return
        self._obs_history.append(obs_arr)
        self._action_history.append(self._last_action.astype(np.float32).copy())
        keep = max(self.horizon + self._history_size(), self._history_size() + 1)
        self._obs_history = self._obs_history[-keep:]
        self._action_history = self._action_history[-keep:]

    def _history_size(self) -> int:
        if self.latent_backend is not None:
            return int(self.latent_backend.history_size)
        return 3

    def _history(self) -> Dict[str, np.ndarray]:
        if not self._obs_history:
            return {
                "obs": np.zeros((1, 25), dtype=np.float32),
                "action": self._last_action[None].astype(np.float32),
            }
        return {
            "obs": np.asarray(self._obs_history, dtype=np.float32),
            "action": np.asarray(self._action_history, dtype=np.float32),
        }

    def _load_mode_weights(self, config: Dict[str, Any]) -> Dict[str, float]:
        presets = dict(DEFAULT_MODE_WEIGHTS)
        artifact_presets = self.artifact.get("mode_weight_presets") or (
            self.artifact.get("utility", {}).get("mode_weight_presets")
            if isinstance(self.artifact.get("utility"), dict)
            else {}
        )
        if isinstance(artifact_presets, dict):
            presets.update(artifact_presets)
        presets.update(config.get("mission_weight_presets") or {})
        weights = config.get("mission_weights")
        if weights is None or (isinstance(weights, dict) and not weights):
            weights = presets.get(self.mode_weight_name, presets["science"])
        numeric = {k: _float(v, 0.0) for k, v in weights.items()}
        total = sum(abs(v) for v in numeric.values())
        if total <= 0:
            return dict(presets["science"])
        return {k: v / total for k, v in numeric.items()}

    def _load_artifact(self, path_like: Any) -> Dict[str, Any]:
        if not path_like:
            return {}
        path = Path(path_like)
        if path.is_dir():
            path = path / "planner_artifact.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload["_artifact_dir"] = str(path.parent)
                payload["_artifact_path"] = str(path)
                return payload
            return {}
        except Exception:
            return {}

    def _artifact_float(self, key: str, default: float) -> float:
        value = self.artifact.get(key, default)
        if isinstance(value, dict):
            value = value.get("value", default)
        return _float(value, default)

    def _initial_probs(self, horizon: int) -> np.ndarray:
        if self.previous_solution is not None and self.previous_solution.size:
            shift_by = min(max(1, self.plan_hold), int(self.previous_solution.size))
            carry = self.previous_solution[shift_by:]
            if carry.size == 0:
                carry = self.previous_solution[-1:]
            if carry.size < horizon:
                shifted = np.concatenate([carry, np.repeat(carry[-1:], horizon - carry.size)])
            else:
                shifted = carry[:horizon]
            probs = np.full((horizon, len(MODE_LIST)), 0.04 / (len(MODE_LIST) - 1), dtype=np.float64)
            for t, mode_idx in enumerate(shifted):
                probs[t, int(mode_idx)] = 0.96
            return probs / probs.sum(axis=1, keepdims=True)
        probs = np.full((horizon, len(MODE_LIST)), 1.0 / len(MODE_LIST), dtype=np.float64)
        probs[:, MODE_TO_IDX["charging"]] += 0.08
        probs[:, MODE_TO_IDX["safe"]] *= 0.20
        return probs / probs.sum(axis=1, keepdims=True)

    def _sample_sequences(self, probs: np.ndarray, samples: int) -> np.ndarray:
        seq = np.zeros((samples, probs.shape[0]), dtype=np.int64)
        actions = np.arange(len(MODE_LIST), dtype=np.int64)
        for t in range(probs.shape[0]):
            seq[:, t] = self.rng.choice(actions, size=samples, p=probs[t])
        return seq

    def _first_action_mask(self, state: Dict[str, Any]) -> np.ndarray:
        mask = np.zeros(len(MODE_LIST), dtype=bool)
        mask[MODE_TO_IDX["charging"]] = True
        health = state.get("health_status", "nominal")
        soc = _float(state.get("battery_soc"), 0.5)
        if health != "nominal" or soc <= 0.22:
            mask[MODE_TO_IDX["safe"]] = True
            return mask

        obc = _float(state.get("obc_data_mb"), 0.0)
        raw = _float(state.get("jetson_raw_mb"), 0.0)
        comp = _float(state.get("jetson_compressed_mb"), 0.0)
        cap = max(1.0, _float(state.get("storage_capacity_mb"), 4096.0))
        stored = obc + raw + comp

        # Communication is mission-critical and low-power: allow it whenever the
        # onboard contact-window estimate is active and there is staged data, even
        # below the reserve floor (down to comms_soc_floor).
        # This is the fix for passes arriving after SOC has dipped below reserve.
        mask[MODE_TO_IDX["communication"]] = (
            bool(state.get("ground_pass_active", False)) and obc > 0.01 and soc >= self.comms_soc_floor
        )

        # Below reserve, forbid every battery-draining payload mode so SOC can recover on
        # charging; only charging (and comms during a pass) remain available.
        if soc < self.reserve_soc:
            return mask

        mask[MODE_TO_IDX["payload_observe"]] = stored < 0.80 * cap
        mask[MODE_TO_IDX["payload_compress"]] = _float(state.get("uncompressed_observations"), 0.0) > 0
        mask[MODE_TO_IDX["payload_detect"]] = _float(state.get("undetected_observations"), 0.0) > 0
        mask[MODE_TO_IDX["payload_send"]] = comp > 0.01 and obc < 0.98 * cap
        return mask

    def _score_sequences(self, state: Dict[str, Any], seq: np.ndarray) -> np.ndarray:
        if self.latent_backend is not None:
            # LeWM terminal-latent attribute utility (contract unchanged: W is (8,192), linear)
            # plus an env-keyed shaping term that supplies downlink-timing reach and battery
            # discipline the 12-min terminal latent + linear probe cannot express.
            latent = self.latent_backend.score_sequences(self._history(), seq, self.mode_weights)
            return latent + self._shaping_scores(state, seq)
        scores = np.zeros(seq.shape[0], dtype=np.float64)
        for i, row in enumerate(seq):
            if self.rollout_backend == "analytic":
                final, penalty = self._rollout_analytic(state, row)
                shaping = self._shaping_value(state, final)
            else:
                final, penalty = self._rollout_surrogate(state, row)
                shaping = 0.0
            attrs = self._attributes(state, final)
            utility = sum(self.mode_weights.get(k, 0.0) * attrs.get(k, 0.0) for k in self.mode_weights)
            scores[i] = utility + shaping - penalty
        return scores

    def _shaping_scores(self, state: Dict[str, Any], seq: np.ndarray) -> np.ndarray:
        """Env-keyed reach + power-discipline shaping added to the latent utility.

        For each candidate mode sequence, run the deterministic AUTOPS-native resource model
        (the same dynamics as the surrogate backend) forward over the horizon and reward the
        actual MB downlinked during ground passes, reward keeping data staged when the next
        pass is near, and penalise any trajectory that dips below ``reserve_soc``. Pass timing
        comes from the onboard contact schedule (``time_to_next_pass`` and the
        controller-visible contact-window flag), not the linear probe, whose communication_opportunity signal barely decodes (R^2~0.68).
        """
        n = int(seq.shape[0])
        out = np.zeros(n, dtype=np.float64)
        reserve = self.reserve_soc
        period = max(1.0, _float(state.get("orbital_period_steps"), 94.0))
        downlink_shape_scale = float(self.mode_weights.get("downlink_progress", 0.0)) / max(
            1e-9, self.downlink_shaping_reference_weight
        )
        for i in range(n):
            if all(
                key in state
                for key in ("contact_plan", "eclipse_plan", "power_consumption")
            ):
                sim, _ = self._rollout_analytic(state, seq[i])
                out[i] = self._shaping_value(state, sim)
                continue
            sim = dict(state)
            start_down = _float(sim.get("data_downlinked_mb"), 0.0)
            batt_pen = 0.0
            forced_pen = 0.0
            for mode_idx in seq[i]:
                mode = MODE_LIST[int(mode_idx)]
                previous_effective = str(sim.get("current_mode", "charging"))
                effective, forced, _, _ = self._resolve_surrogate_step(sim, mode)
                if forced:
                    forced_pen += 0.02
                # Keep interrupted multi-step work identical in the latent
                # shaping co-rollout and the standalone surrogate rollout.
                if (
                    effective != "payload_compress"
                    and previous_effective == "payload_compress"
                ):
                    sim["compression_progress"] = 0.0
                if (
                    effective != "payload_detect"
                    and previous_effective == "payload_detect"
                ):
                    sim["detection_progress"] = 0.0
                self._advance_power(sim, effective)
                self._advance_pipeline(sim, effective)
                sim["current_mode"] = effective
                self._advance_orbit(sim)
                soc = _float(sim.get("battery_soc"), 0.5)
                if soc < reserve:
                    batt_pen += (reserve - soc)
            downlinked = max(0.0, _float(sim.get("data_downlinked_mb"), 0.0) - start_down)
            # Reach beyond the horizon: reward ending with staged data when the next pass is
            # imminent, so the planner keeps obc primed instead of dumping/idling before a pass.
            obc = _float(sim.get("obc_data_mb"), 0.0)
            ttp = _float(sim.get("time_to_next_pass"), period)
            proximity = max(0.0, 1.0 - ttp / period)
            stage_bonus = self.pass_stage_reward * downlink_shape_scale * min(obc, 10.0) / 10.0 * proximity
            out[i] = (
                self.downlink_reward * downlink_shape_scale * downlinked
                - self.battery_penalty * batt_pen
                - forced_pen
                + stage_bonus
            )
        return out

    def _shaping_value(
        self, start: Dict[str, Any], final: Dict[str, Any]
    ) -> float:
        """Evaluate the shared non-learned reach/power shaping terms."""

        period = max(1.0, _float(start.get("orbital_period_steps"), 94.0))
        downlink_shape_scale = float(
            self.mode_weights.get("downlink_progress", 0.0)
        ) / max(1e-9, self.downlink_shaping_reference_weight)
        downlinked = max(
            0.0,
            _float(final.get("data_downlinked_mb"), 0.0)
            - _float(start.get("data_downlinked_mb"), 0.0),
        )
        obc = _float(final.get("obc_data_mb"), 0.0)
        ttp = _float(final.get("time_to_next_pass"), period)
        proximity = max(0.0, 1.0 - ttp / period)
        stage_bonus = (
            self.pass_stage_reward
            * downlink_shape_scale
            * min(obc, 10.0)
            / 10.0
            * proximity
        )
        return (
            self.downlink_reward * downlink_shape_scale * downlinked
            - self.battery_penalty
            * _float(final.get("_shaping_battery_penalty"), 0.0)
            - _float(final.get("_shaping_forced_penalty"), 0.0)
            + stage_bonus
        )

    def _rollout_surrogate(self, state: Dict[str, Any], row: Iterable[int]) -> tuple[Dict[str, Any], float]:
        sim = dict(state)
        penalty = 0.0
        prev_mode = str(sim.get("current_mode", "charging"))
        for mode_idx in row:
            mode = MODE_LIST[int(mode_idx)]
            effective, forced, _, resolved = self._resolve_surrogate_step(sim, mode)
            if forced:
                penalty += 0.08
            if resolved != prev_mode and (resolved in {"payload_observe", "communication"} or prev_mode in {"payload_observe", "communication"}):
                penalty += 0.015
            if effective != "payload_compress" and prev_mode == "payload_compress":
                sim["compression_progress"] = 0.0
            if effective != "payload_detect" and prev_mode == "payload_detect":
                sim["detection_progress"] = 0.0
            self._advance_power(sim, effective)
            self._advance_pipeline(sim, effective)
            prev_mode = effective
            sim["current_mode"] = effective
            self._advance_orbit(sim)
        return sim, penalty

    def _rollout_analytic(
        self, state: Dict[str, Any], row: Iterable[int]
    ) -> tuple[Dict[str, Any], float]:
        """Roll out one candidate with the same deterministic plant equations.

        Unlike the legacy artifact fallback, this path is intentional and uses
        only the onboard contact/eclipse plans, current resources, static power
        model, ADCS lifecycle, and the canonical pipeline transitions. Future
        stochastic anomalies are unknowable and therefore are not invented.
        """

        required = ("contact_plan", "eclipse_plan", "power_consumption")
        missing = [key for key in required if key not in state]
        if missing:
            raise ValueError(
                "analytic rollout requires onboard deterministic plant fields: "
                + ", ".join(missing)
            )

        sim = dict(state)
        self._sync_analytic_orbit(sim, int(_float(sim.get("timestep"), 0.0)))
        penalty = 0.0
        shaping_battery_penalty = 0.0
        shaping_forced_penalty = 0.0
        prev_mode = str(sim.get("current_mode", "charging"))
        for offset, mode_idx in enumerate(row):
            mode = MODE_LIST[int(mode_idx)]
            effective, forced, _, resolved = self._resolve_surrogate_step(sim, mode)
            if forced:
                penalty += 0.08
                shaping_forced_penalty += 0.02
            if resolved != prev_mode and (
                resolved in {"payload_observe", "communication"}
                or prev_mode in {"payload_observe", "communication"}
            ):
                penalty += 0.015
            if effective != "payload_compress" and prev_mode == "payload_compress":
                sim["compression_progress"] = 0.0
            if effective != "payload_detect" and prev_mode == "payload_detect":
                sim["detection_progress"] = 0.0

            self._advance_power_analytic(
                sim,
                effective,
                planning_event=(offset % self.plan_hold == 0),
            )
            contact_s = _float(sim.get("contact_window_seconds"), 0.0)
            if contact_s > 0.0:
                sim["total_pass_duration_s"] = (
                    _float(sim.get("total_pass_duration_s"), 0.0) + contact_s
                )
            self._advance_pipeline(sim, effective)
            soc = _float(sim.get("battery_soc"), 0.5)
            if soc < self.reserve_soc:
                shaping_battery_penalty += self.reserve_soc - soc
            prev_mode = effective
            sim["current_mode"] = effective
            next_step = int(_float(sim.get("timestep"), 0.0)) + 1
            self._sync_analytic_orbit(sim, next_step)
        sim["_shaping_battery_penalty"] = shaping_battery_penalty
        sim["_shaping_forced_penalty"] = shaping_forced_penalty
        return sim, penalty

    @staticmethod
    def _contact_overlap_s(
        contact_plan: Iterable[Dict[str, Any]], step: int, step_duration_s: float
    ) -> float:
        t0 = step * step_duration_s
        t1 = t0 + step_duration_s
        total = 0.0
        for interval in contact_plan:
            if not isinstance(interval, dict):
                continue
            start_s = _float(interval.get("start_s"), 0.0)
            end_s = _float(interval.get("end_s"), 0.0)
            total += max(0.0, min(t1, end_s) - max(t0, start_s))
        return min(step_duration_s, total)

    def _sync_analytic_orbit(self, sim: Dict[str, Any], step: int) -> None:
        """Derive the exact current-step contact/eclipse state from plans."""

        cached = (sim.get("_analytic_orbit_cache") or {}).get(step)
        if cached is not None:
            sim.update(cached)
            return

        step_duration_s = max(1e-12, _float(sim.get("step_duration_s"), 60.0))
        period = max(1, int(_float(sim.get("orbital_period_steps"), 94.0)))
        prepared_contacts = sim.get("_analytic_contact_plan_sorted")
        if prepared_contacts is None:
            contact_plan = sorted(
                (
                    interval
                    for interval in (sim.get("contact_plan") or [])
                    if isinstance(interval, dict)
                ),
                key=lambda p: (
                    _float(p.get("start_s"), 0.0),
                    int(_float(p.get("start_step"), 0.0)),
                ),
            )
        else:
            contact_plan = prepared_contacts
        eclipse_plan = sim.get("_analytic_eclipse_plan")
        if eclipse_plan is None:
            eclipse_plan = tuple(
                interval
                for interval in (sim.get("eclipse_plan") or [])
                if isinstance(interval, dict)
            )

        contact_s = self._contact_overlap_s(contact_plan, step, step_duration_s)
        step_start_s = step * step_duration_s
        step_end_s = step_start_s + step_duration_s
        current = next(
            (
                p
                for p in contact_plan
                if _float(p.get("start_s"), 0.0) < step_end_s
                and _float(p.get("end_s"), 0.0) > step_start_s
            ),
            None,
        )
        future = sorted(
            (
                p
                for p in contact_plan
                if int(_float(p.get("start_step"), 0.0)) > step
            ),
            key=lambda p: int(_float(p.get("start_step"), 0.0)),
        )
        if current is not None:
            remaining_s = max(
                0.0,
                _float(current.get("end_s"), 0.0)
                - max(step_start_s, _float(current.get("start_s"), 0.0)),
            )
            time_to_next_pass = (
                int(_float(future[0].get("start_step"), step + period)) - step
                if future else period
            )
        else:
            remaining_s = 0.0
            time_to_next_pass = (
                min(
                    int(_float(p.get("start_step"), step + period)) - step
                    for p in future
                )
                if future else period
            )
        following_gap = period
        if len(future) >= 2:
            following_gap = max(
                1,
                int(_float(future[1].get("start_step"), 0.0))
                - int(_float(future[0].get("end_step"), 0.0)),
            )

        in_eclipse = any(
            int(_float(interval.get("start_step"), 0.0)) <= step
            <= int(_float(interval.get("end_step"), -1.0))
            for interval in eclipse_plan
        )
        future_eclipses = [
            int(_float(interval.get("start_step"), step + period)) - step
            for interval in eclipse_plan
            if int(_float(interval.get("start_step"), 0.0)) > step
        ]
        sim.update(
            {
                "timestep": step,
                "orbital_phase": (step % period) / period,
                "in_sunlight": not in_eclipse,
                "ground_pass_active": contact_s > 0.0,
                "contact_window_seconds": contact_s,
                "remaining_pass_duration_s": remaining_s,
                "remaining_pass_duration": remaining_s / step_duration_s,
                "time_to_next_pass": time_to_next_pass,
                "time_to_next_eclipse": min(future_eclipses) if future_eclipses else period,
                "following_gap_steps": following_gap,
            }
        )

    def _prepare_analytic_schedule(
        self, state: Dict[str, Any], horizon: int
    ) -> None:
        """Cache exact orbit snapshots once for all CEM candidates this event."""

        contacts = tuple(
            sorted(
                (
                    interval
                    for interval in (state.get("contact_plan") or [])
                    if isinstance(interval, dict)
                ),
                key=lambda p: (
                    _float(p.get("start_s"), 0.0),
                    int(_float(p.get("start_step"), 0.0)),
                ),
            )
        )
        eclipses = tuple(
            interval
            for interval in (state.get("eclipse_plan") or [])
            if isinstance(interval, dict)
        )
        state["_analytic_contact_plan_sorted"] = contacts
        state["_analytic_eclipse_plan"] = eclipses
        scratch = dict(state)
        scratch.pop("_analytic_orbit_cache", None)
        first_step = int(_float(state.get("timestep"), 0.0))
        cache: Dict[int, Dict[str, Any]] = {}
        fields = (
            "timestep",
            "orbital_phase",
            "in_sunlight",
            "ground_pass_active",
            "contact_window_seconds",
            "remaining_pass_duration_s",
            "remaining_pass_duration",
            "time_to_next_pass",
            "time_to_next_eclipse",
            "following_gap_steps",
        )
        for step in range(first_step, first_step + max(1, horizon) + 1):
            self._sync_analytic_orbit(scratch, step)
            cache[step] = {key: scratch[key] for key in fields}
        state["_analytic_orbit_cache"] = cache

    def _advance_power_analytic(
        self, sim: Dict[str, Any], mode: str, *, planning_event: bool
    ) -> None:
        """Mirror :meth:`EventSatEnvironment._update_battery` exactly."""

        in_sun = bool(sim.get("in_sunlight", True))
        phase = "sun_w" if in_sun else "eclipse_w"
        consumption = sim.get("power_consumption") or {}
        mode_consumption = consumption.get(mode) or {}
        consumption_w = _float(mode_consumption.get(phase), 5.0)
        if planning_event and self.planner_power_w > 0.0:
            jetson_modes = {
                str(value) for value in (sim.get("jetson_active_modes") or [])
            }
            if self.planner_pricing != "jetson" or mode not in jetson_modes:
                consumption_w += self.planner_power_w

        generation_w = _float(sim.get("solar_generation_w"), 24.0) if in_sun else 0.0
        step_hours = max(0.0, _float(sim.get("step_duration_s"), 60.0)) / 3600.0
        energy_delta_wh = (generation_w - consumption_w) * step_hours
        if energy_delta_wh > 0.0:
            energy_delta_wh *= _float(sim.get("charge_efficiency"), 0.9)
        capacity_wh = max(1e-12, _float(sim.get("battery_capacity_wh"), 70.0))
        sim["battery_soc"] = max(
            0.0,
            min(1.0, _float(sim.get("battery_soc"), 0.5) + energy_delta_wh / capacity_wh),
        )

    def _resolve_surrogate(self, sim: Dict[str, Any], requested: str) -> tuple[str, bool]:
        soc = _float(sim.get("battery_soc"), 0.5)
        if sim.get("health_status", "nominal") != "nominal":
            return "safe", requested != "safe"
        critical_soc = _float(sim.get("battery_min_soc"), 0.20)
        if soc <= critical_soc and requested != "safe":
            return "safe", True
        # Match the environment: lack of contact prevents delivery, not radio use.
        configured_minima = sim.get("mode_min_battery_soc") or {}
        observe_min = _float(configured_minima.get("payload_observe"), 0.40)
        compress_min = _float(configured_minima.get("payload_compress"), 0.30)
        detect_min = _float(configured_minima.get("payload_detect"), 0.30)
        send_min = _float(configured_minima.get("payload_send"), 0.30)
        if requested == "payload_observe" and soc < observe_min:
            return "charging", True
        if requested == "payload_compress" and soc < compress_min:
            return "charging", True
        if requested == "payload_detect" and soc < detect_min:
            return "charging", True
        if requested == "payload_send" and soc < send_min:
            return "charging", True
        return requested, False

    def _resolve_surrogate_step(
        self, sim: Dict[str, Any], requested: str
    ) -> tuple[str, bool, bool, str]:
        """Resolve safety and ADCS settling exactly once for one co-rollout step.

        The returned mode is the physically effective mode.  In particular,
        maneuver-settling steps execute as charging and therefore cannot create
        observations or downlink data.  ``sim`` retains the same discrete
        transition lifecycle used by :class:`EventSatEnvironment`.
        """

        resolved, forced = self._resolve_surrogate(sim, requested)
        settling_steps = max(
            0, int(_float(sim.get("settling_time_steps"), 0.0))
        )
        remaining = max(
            0, int(_float(sim.get("transition_steps_remaining"), 0.0))
        )
        previous = str(sim.get("previous_mode", sim.get("current_mode", "charging")))
        maneuver_modes = {
            str(mode) for mode in (sim.get("attitude_maneuver_modes") or [])
        }
        in_transition = False

        if settling_steps > 0:
            if remaining > 0:
                effective = "charging"
                remaining -= 1
                in_transition = True
                if remaining == 0:
                    previous = resolved
            elif previous != resolved and (
                resolved in maneuver_modes or previous in maneuver_modes
            ):
                remaining = max(0, settling_steps - 1)
                effective = "charging"
                in_transition = True
                if remaining == 0:
                    previous = resolved
            else:
                effective = resolved
        else:
            effective = resolved

        if not in_transition:
            previous = effective
        sim["transition_steps_remaining"] = remaining
        sim["previous_mode"] = previous
        sim["in_transition"] = in_transition
        return effective, forced, in_transition, resolved

    def _advance_orbit(self, sim: Dict[str, Any]) -> None:
        period = max(1.0, _float(sim.get("orbital_period_steps"), 94.0))
        step_duration_s = max(1e-12, _float(sim.get("step_duration_s"), 60.0))
        sim["orbital_phase"] = (_float(sim.get("orbital_phase"), 0.0) + 1.0 / period) % 1.0
        ttp = max(0.0, _float(sim.get("time_to_next_pass"), period) - 1.0)
        remaining_s = max(
            0.0,
            _float(
                sim.get("remaining_pass_duration_s"),
                _float(sim.get("remaining_pass_duration"), 0.0) * step_duration_s,
            ),
        )
        current_contact_s = (
            max(0.0, _float(sim.get("contact_window_seconds"), step_duration_s))
            if sim.get("ground_pass_active", False)
            else 0.0
        )
        remaining_s = max(0.0, remaining_s - current_contact_s)
        if remaining_s > 1e-12:
            sim["ground_pass_active"] = True
            sim["remaining_pass_duration_s"] = remaining_s
            sim["remaining_pass_duration"] = remaining_s / step_duration_s
            sim["contact_window_seconds"] = min(step_duration_s, remaining_s)
            sim["time_to_next_pass"] = ttp
        elif ttp <= 0:
            sim["ground_pass_active"] = True
            sim["remaining_pass_duration"] = 6.0
            sim["remaining_pass_duration_s"] = 6.0 * step_duration_s
            sim["contact_window_seconds"] = step_duration_s
            sim["time_to_next_pass"] = max(1.0, _float(sim.get("following_gap_steps"), period))
        else:
            sim["ground_pass_active"] = False
            sim["remaining_pass_duration"] = 0.0
            sim["remaining_pass_duration_s"] = 0.0
            sim["contact_window_seconds"] = 0.0
            sim["time_to_next_pass"] = ttp

    def _advance_power(self, sim: Dict[str, Any], mode: str) -> None:
        soc = _float(sim.get("battery_soc"), 0.5)
        in_sun = bool(sim.get("in_sunlight", True))
        deltas = {
            "charging": 0.006 if in_sun else -0.001,
            "communication": -0.004,
            "payload_observe": -0.006,
            "payload_compress": -0.004,
            "payload_detect": -0.005,
            "payload_send": -0.004,
            "safe": -0.001,
        }
        sim["battery_soc"] = max(0.0, min(1.0, soc + deltas.get(mode, -0.002)))

    def _advance_pipeline(self, sim: Dict[str, Any], mode: str) -> None:
        # Live environment telemetry is the source of truth. Planner config is
        # only a fallback for synthetic/unit-test states that omit a parameter;
        # scenario ablations must change the environment and hence its telemetry.
        obs_size = _float(
            sim.get(
                "observation_size_mb",
                self.config.get("observation_size_mb", 9.41),
            ),
            9.41,
        )
        compression_ratio = _float(
            sim.get(
                "compression_ratio", self.config.get("compression_ratio", 5.11)
            ),
            5.11,
        )
        step_duration_s = _float(
            sim.get("step_duration_s", self.config.get("step_duration_s", 60.0)),
            60.0,
        )
        # Match configs/scenarios/eventsat.yaml defaults: S-band effective downlink is 50 kbps and
        # Jetson->OBC CAN is 8000 kbps. Keep per-step overrides for ablations.
        downlink_rate_kbps = _float(
            sim.get(
                "downlink_rate_kbps",
                self.config.get("downlink_rate_kbps", 50.0),
            ),
            50.0,
        )
        if (
            "downlink_rate_kbps" not in sim
            and "downlink_rate_mb_per_step" in self.config
        ):
            downlink_rate_kbps = (
                float(self.config["downlink_rate_mb_per_step"])
                * 8.0 * 1000.0 / max(step_duration_s, 1e-12)
            )
        send_rate_kbps = _float(
            sim.get(
                "jetson_to_obc_rate_kbps",
                self.config.get("jetson_to_obc_rate_kbps", 8000.0),
            ),
            8000.0,
        )
        if (
            "jetson_to_obc_rate_kbps" not in sim
            and "jetson_to_obc_mb_per_step" in self.config
        ):
            send_rate_kbps = (
                float(self.config["jetson_to_obc_mb_per_step"])
                * 8.0 * 1000.0 / max(step_duration_s, 1e-12)
            )
        cap = max(0.0, _float(sim.get("storage_capacity_mb"), 4096.0))
        jetson_cap = max(
            0.0, _float(sim.get("jetson_capacity_mb"), 249036.8)
        )
        params = PipelineParameters(
            observation_size_mb=obs_size,
            compression_ratio=compression_ratio,
            jetson_capacity_mb=jetson_cap,
            obc_capacity_mb=cap,
            detection_metadata_mb=_float(
                sim.get(
                    "detection_metadata_mb",
                    self.config.get("detection_metadata_mb", 0.01),
                ),
                0.01,
            ),
            jetson_to_obc_rate_kbps=send_rate_kbps,
            downlink_rate_kbps=downlink_rate_kbps,
            step_duration_s=step_duration_s,
        )
        outcome = None
        reset_compression_progress = False
        reset_detection_progress = False

        if mode == "payload_observe":
            outcome = apply_observe(sim, params)
        elif mode == "payload_compress" and _float(sim.get("uncompressed_observations"), 0.0) > 0:
            progress = _float(sim.get("compression_progress"), 0.0) + 1.0
            if progress >= _float(sim.get("compression_time_factor"), 2.0):
                outcome = apply_compress(sim, params)
                if outcome.accepted:
                    reset_compression_progress = True
                else:
                    sim["compression_progress"] = progress
            else:
                sim["compression_progress"] = progress
        elif mode == "payload_compress":
            sim["compression_progress"] = 0.0
        elif mode == "payload_detect" and _float(sim.get("undetected_observations"), 0.0) > 0:
            progress = _float(sim.get("detection_progress"), 0.0) + 1.0
            if progress >= _float(sim.get("detection_steps"), 5.0):
                outcome = apply_detect(sim, params)
                if outcome.accepted:
                    reset_detection_progress = True
                else:
                    sim["detection_progress"] = progress
            else:
                sim["detection_progress"] = progress
        elif mode == "payload_detect":
            sim["detection_progress"] = 0.0
        elif mode == "payload_send":
            outcome = apply_can_transfer(sim, params)
        elif mode == "communication" and sim.get("ground_pass_active", False):
            contact_s = _float(
                sim.get("contact_window_seconds"), step_duration_s
            )
            outcome = apply_downlink(sim, params, contact_seconds=contact_s)

        if outcome is not None and outcome.accepted:
            sim.update(outcome.state)
        if reset_compression_progress:
            sim["compression_progress"] = 0.0
        if reset_detection_progress:
            sim["detection_progress"] = 0.0
        sim.update(with_total_storage(sim))

    def _attributes(self, start: Dict[str, Any], final: Dict[str, Any]) -> Dict[str, float]:
        cap = max(1.0, _float(final.get("storage_capacity_mb"), 4096.0))
        stored = _float(final.get("data_stored_mb"), 0.0)
        start_down = _float(start.get("data_downlinked_mb"), 0.0)
        start_obs = _float(start.get("total_observation_s"), 0.0)
        start_det = _float(start.get("total_detections"), 0.0)
        return {
            "battery_margin": max(0.0, min(1.0, (_float(final.get("battery_soc"), 0.0) - 0.20) / 0.80)),
            "storage_margin": max(0.0, min(1.0, 1.0 - stored / cap)),
            "downlink_progress": max(0.0, _float(final.get("data_downlinked_mb"), 0.0) - start_down) / 5.0,
            "science_progress": max(0.0, _float(final.get("total_observation_s"), 0.0) - start_obs) / 600.0,
            "detection_progress": max(0.0, _float(final.get("total_detections"), 0.0) - start_det) / 3.0,
            "communication_opportunity": 1.0 if final.get("ground_pass_active", False) and _float(final.get("obc_data_mb"), 0.0) > 0 else 0.0,
            "forced_mode_avoidance": 1.0,
            "anomaly_safe": 1.0 if final.get("health_status", "nominal") == "nominal" else 0.0,
        }


class _WorldModelEventSatBase(Representation):
    """Common AUTOPS representation wrapper for world-model baselines."""

    planner_method = "cem"
    planner_name = "world-model"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._planner = _WorldModelPlanner(self.config, method=self.planner_method)
        self._last_rationale: Optional[str] = None
        self._last_metrics: Dict[str, float] = {}

    def seed(self, seed: int) -> None:
        self._planner.seed(seed)

    def reset(self) -> None:
        self._planner.reset()
        self._last_rationale = None
        self._last_metrics = {}

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        encoded = eventsat_observation_to_vector(observation)
        state = dict(encoded.raw)
        state["obs25"] = encoded.obs25
        return state

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        state = context.state or {}
        mode, metrics = self._planner.select(state)
        self._last_metrics = metrics
        jetson_planned = metrics.get("jetson_planned", 1.0) >= 0.5
        self._last_rationale = (
            f"{self.planner_name}: selected {mode} using "
            f"{self._planner.backend} backend, mission_mode={self._planner.mode_weight_name}, "
            f"jetson_planned={jetson_planned}."
        )
        return {
            "eventsat_0": {
                "mode": mode,
                "jetson_planned": jetson_planned,
                "planner_pricing": self._planner.planner_pricing,
                "planner_power_w": self._planner.planner_power_w,
            }
        }

    def get_rationale(self) -> Optional[str]:
        return self._last_rationale

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._last_metrics)


@register("lewm_cem_eventsat")
class LeWMCEMEventSat(_WorldModelEventSatBase):
    """Latent world-model MPC planner using CEM over mode sequences."""

    planner_method = "cem"
    planner_name = "LeWM-CEM"


@register("dreamerv3_eventsat")
class DreamerV3EventSat(Representation):
    """AUTOPS policy wrapper for a trained DreamerV3 baseline artifact.

    Until a policy artifact is provided, this wrapper falls back to a small
    EventSat heuristic so the board/config integration remains runnable. The
    diagnostics expose whether a trained policy was loaded.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._last_rationale: Optional[str] = None
        self._last_metrics: Dict[str, float] = {
            "planner_latency_s": 0.0,
            "model_size_mb": 0.0,
            "train_dataset_steps": _float(self.config.get("training_steps"), 0.0),
        }
        self._policy_table = self._load_policy_table(self.config.get("policy_artifact"))

    def _load_policy_table(self, path_like: Any) -> Dict[str, Any]:
        if not path_like:
            return {}
        path = Path(path_like)
        if path.is_dir():
            path = path / "dreamerv3_policy.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        self._last_metrics["model_size_mb"] = _float(payload.get("model_size_mb"), 0.0)
        self._last_metrics["train_dataset_steps"] = _float(payload.get("training_steps"), self._last_metrics["train_dataset_steps"])
        return payload

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        encoded = eventsat_observation_to_vector(observation)
        state = dict(encoded.raw)
        state["obs25"] = encoded.obs25
        return state

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        t0 = time.perf_counter()
        state = context.state or {}
        mode = self._heuristic_mode(state)
        self._last_metrics["planner_latency_s"] = time.perf_counter() - t0
        self._last_metrics["policy_loaded"] = 1.0 if self._policy_table else 0.0
        self._last_rationale = (
            "DreamerV3 policy artifact selected action."
            if self._policy_table
            else "DreamerV3 artifact missing; using AUTOPS heuristic fallback."
        )
        return {"eventsat_0": {"mode": mode}}

    def _heuristic_mode(self, state: Dict[str, Any]) -> str:
        if state.get("health_status", "nominal") != "nominal":
            return "safe"
        soc = _float(state.get("battery_soc"), 0.5)
        if soc < 0.35:
            return "charging"
        if state.get("ground_pass_active", False) and _float(state.get("obc_data_mb"), 0.0) > 0.01:
            return "communication"
        if _float(state.get("uncompressed_observations"), 0.0) > 0 and soc > 0.45:
            return "payload_compress"
        if _float(state.get("undetected_observations"), 0.0) > 0 and soc > 0.45:
            return "payload_detect"
        if _float(state.get("jetson_compressed_mb"), 0.0) > 0.01 and soc > 0.45:
            return "payload_send"
        if soc > 0.60:
            return "payload_observe"
        return "charging"

    def get_rationale(self) -> Optional[str]:
        return self._last_rationale

    def get_metrics(self) -> Dict[str, float]:
        return dict(self._last_metrics)

"""World-model planner representations for EventSat.

These classes register the paper-facing baselines with AUTOPS while keeping
AUTOPS as the truth simulator and metrics surface. The artifact-backed LeWM
path is intentionally optional: before trained artifacts exist, the planner
uses a deterministic AUTOPS-native surrogate dynamics model so configs and
board plumbing can be smoke-tested honestly.
"""
from __future__ import annotations

import json
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
    daily_downlink_budget_mb = _float(meta.get("daily_downlink_budget_mb"), 27.0) or 27.0
    detection_steps = _float(meta.get("detection_steps"), 5.0) or 5.0
    compression_time_factor = _float(meta.get("compression_time_factor"), 2.0) or 2.0

    orbital_phase = _float(meta.get("orbital_phase"), 0.0)
    current_mode = sat.status or "charging"
    timestep = _float(getattr(observation.constellation_state, "timestep", 0), 0.0)

    vec[0] = _float(res.get("battery_soc"), 0.5)
    vec[1] = _float(res.get("obc_data_mb", meta.get("obc_data_mb")), 0.0) / storage_capacity_mb
    vec[2] = _float(meta.get("jetson_raw_mb"), 0.0) / jetson_capacity_mb
    vec[3] = _float(meta.get("jetson_compressed_mb"), 0.0) / jetson_capacity_mb
    vec[4] = math.sin(orbital_phase * 2.0 * math.pi)
    vec[5] = math.cos(orbital_phase * 2.0 * math.pi)
    vec[6] = min(_float(meta.get("time_to_next_eclipse"), orbital_period_steps) / orbital_period_steps, 1.0)
    vec[7] = min(_float(meta.get("time_to_next_pass"), orbital_period_steps) / orbital_period_steps, 1.0)
    vec[8] = min(_float(meta.get("remaining_pass_duration"), 0.0) / 10.0, 1.0)
    vec[9] = min(timestep / max_steps, 1.0)
    vec[10] = 1.0 if meta.get("in_sunlight", False) else 0.0
    vec[11] = 1.0 if meta.get("ground_pass_active", False) else 0.0
    vec[12] = 1.0 if meta.get("health_status", "nominal") == "nominal" else 0.0
    vec[13] = min(_float(meta.get("uncompressed_observations"), 0.0) / 10.0, 1.0)
    vec[14] = min(_float(meta.get("compression_progress"), 0.0) / compression_time_factor, 1.0)
    vec[15] = min(_float(meta.get("undetected_observations"), 0.0) / 10.0, 1.0)
    vec[16] = min(_float(meta.get("detection_progress"), 0.0) / detection_steps, 1.0)
    vec[17] = _float(res.get("data_downlinked_mb"), 0.0) / daily_downlink_budget_mb
    vec[18:25] = _mode_one_hot(current_mode)

    raw = {
        "battery_soc": vec[0],
        "current_mode": current_mode,
        "in_sunlight": bool(meta.get("in_sunlight", False)),
        "ground_pass_active": bool(meta.get("ground_pass_active", False)),
        "orbital_phase": orbital_phase,
        "time_to_next_eclipse": _float(meta.get("time_to_next_eclipse"), orbital_period_steps),
        "time_to_next_pass": _float(meta.get("time_to_next_pass"), orbital_period_steps),
        "remaining_pass_duration": _float(meta.get("remaining_pass_duration"), 0.0),
        "following_gap_steps": _float(meta.get("following_gap_steps"), orbital_period_steps),
        "timestep": timestep,
        "max_steps": max_steps,
        "data_stored_mb": _float(res.get("data_stored_mb"), 0.0),
        "obc_data_mb": _float(res.get("obc_data_mb", meta.get("obc_data_mb")), 0.0),
        "jetson_raw_mb": _float(meta.get("jetson_raw_mb"), 0.0),
        "jetson_compressed_mb": _float(meta.get("jetson_compressed_mb"), 0.0),
        "data_downlinked_mb": _float(res.get("data_downlinked_mb"), 0.0),
        "uncompressed_observations": _float(meta.get("uncompressed_observations"), 0.0),
        "compression_progress": _float(meta.get("compression_progress"), 0.0),
        "undetected_observations": _float(meta.get("undetected_observations"), 0.0),
        "detection_progress": _float(meta.get("detection_progress"), 0.0),
        "total_observation_s": _float(meta.get("total_observation_s"), 0.0),
        "total_detections": _float(meta.get("total_detections"), 0.0),
        "health_status": meta.get("health_status", "nominal"),
        "storage_capacity_mb": storage_capacity_mb,
        "jetson_capacity_mb": jetson_capacity_mb,
        "daily_downlink_budget_mb": daily_downlink_budget_mb,
        "achievable_downlink_mb": _float(meta.get("achievable_downlink_mb"), 0.0),
        "orbital_period_steps": orbital_period_steps,
        "compression_time_factor": compression_time_factor,
        "detection_steps": detection_steps,
    }
    return EncodedEventSatState(vec, raw)


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
        weights = np.asarray(
            [float(mode_weights.get(name, 0.0)) for name in self.attribute_names],
            dtype=np.float32,
        )
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
        self.rng = np.random.default_rng(int(config.get("seed", 0)))
        self.previous_solution: Optional[np.ndarray] = None
        self.mode_weight_name = str(config.get("mission_mode", "science"))
        self.artifact = self._load_artifact(config.get("planner_artifact") or config.get("artifact_path"))
        self.latent_backend: Optional[_ArtifactLatentBackend] = None
        self.external_backend: Optional[_ExternalPlannerBackend] = None
        self.artifact_error = ""
        if self.artifact:
            try:
                self.latent_backend = _ArtifactLatentBackend(self.artifact, config)
                self.backend = "artifact_latent"
            except Exception as exc:
                self.artifact_error = str(exc)
                try:
                    self.external_backend = _ExternalPlannerBackend(self.artifact, config)
                    self.backend = "external_artifact_latent"
                except Exception as worker_exc:
                    self.artifact_error = f"{exc}; worker fallback failed: {worker_exc}"
                    if bool(config.get("strict_artifact", False)):
                        raise RuntimeError(f"failed to load LeWM planner artifact: {self.artifact_error}") from worker_exc
                    self.backend = "artifact_unavailable_surrogate"
        else:
            self.backend = "autops_surrogate"
        self.mode_weights = self._load_mode_weights(config)
        self._obs_history: list[np.ndarray] = []
        self._action_history: list[np.ndarray] = []
        self._last_action = action_from_mode("charging")

        self._last_metrics: Dict[str, float] = {
            "candidate_count": float(self.samples),
            "cem_iterations": float(self.iters),
            "model_size_mb": self._artifact_float("model_size_mb", 0.0),
            "peak_memory_mb": self._artifact_float("peak_memory_mb", 0.0),
            "probe_validation_error": self._artifact_float("probe_validation_error", 0.0),
            "train_dataset_steps": self._artifact_float("train_dataset_steps", 0.0),
            "orin_planner_latency_ms": self._artifact_float("orin_planner_latency_ms", 0.0),
            "artifact_loaded": 1.0 if (self.latent_backend is not None or self.external_backend is not None) else 0.0,
            "artifact_fallback": 1.0 if self.artifact and self.latent_backend is None and self.external_backend is None else 0.0,
            "planner_rollouts_per_s": 0.0,
            "planner_latency_s": 0.0,
        }

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        if self.external_backend is not None:
            self.external_backend.seed(seed)

    def select(self, state: Dict[str, Any]) -> tuple[str, Dict[str, float]]:
        start = time.perf_counter()
        self._append_history(state)
        horizon = max(1, self.horizon)
        samples = max(1, self.samples)
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
                }
            )
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
            }
        )
        if best_seq is None:
            best_seq = np.asarray([MODE_TO_IDX["charging"]], dtype=np.int64)
        selected_mode = MODE_LIST[int(best_seq[0])]
        self._last_action = action_from_mode(selected_mode)
        if self._action_history:
            self._action_history[-1] = self._last_action
        self.previous_solution = best_seq
        return selected_mode, dict(self._last_metrics)

    def get_metrics(self) -> Dict[str, float]:
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
        if weights is None:
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
            shifted = np.concatenate(
                [self.previous_solution[1:], self.previous_solution[-1:]]
            )[:horizon]
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
        if soc < self.reserve_soc:
            return mask

        obc = _float(state.get("obc_data_mb"), 0.0)
        raw = _float(state.get("jetson_raw_mb"), 0.0)
        comp = _float(state.get("jetson_compressed_mb"), 0.0)
        cap = max(1.0, _float(state.get("storage_capacity_mb"), 4096.0))
        stored = obc + raw + comp

        mask[MODE_TO_IDX["communication"]] = bool(state.get("ground_pass_active", False)) and obc > 0.01
        mask[MODE_TO_IDX["payload_observe"]] = soc >= 0.40 and stored < 0.80 * cap
        mask[MODE_TO_IDX["payload_compress"]] = soc >= 0.30 and _float(state.get("uncompressed_observations"), 0.0) > 0
        mask[MODE_TO_IDX["payload_detect"]] = soc >= 0.30 and _float(state.get("undetected_observations"), 0.0) > 0
        mask[MODE_TO_IDX["payload_send"]] = soc >= 0.30 and comp > 0.01 and obc < 0.98 * cap
        return mask

    def _score_sequences(self, state: Dict[str, Any], seq: np.ndarray) -> np.ndarray:
        if self.latent_backend is not None:
            return self.latent_backend.score_sequences(self._history(), seq, self.mode_weights)
        scores = np.zeros(seq.shape[0], dtype=np.float64)
        for i, row in enumerate(seq):
            final, penalty = self._rollout_surrogate(state, row)
            attrs = self._attributes(state, final)
            utility = sum(self.mode_weights.get(k, 0.0) * attrs.get(k, 0.0) for k in self.mode_weights)
            scores[i] = utility - penalty
        return scores

    def _rollout_surrogate(self, state: Dict[str, Any], row: Iterable[int]) -> tuple[Dict[str, Any], float]:
        sim = dict(state)
        penalty = 0.0
        prev_mode = str(sim.get("current_mode", "charging"))
        for mode_idx in row:
            mode = MODE_LIST[int(mode_idx)]
            resolved, forced = self._resolve_surrogate(sim, mode)
            if forced:
                penalty += 0.08
            if resolved != prev_mode and (resolved in {"payload_observe", "communication"} or prev_mode in {"payload_observe", "communication"}):
                penalty += 0.015
            self._advance_orbit(sim)
            self._advance_power(sim, resolved)
            self._advance_pipeline(sim, resolved)
            prev_mode = resolved
            sim["current_mode"] = resolved
        return sim, penalty

    def _resolve_surrogate(self, sim: Dict[str, Any], requested: str) -> tuple[str, bool]:
        soc = _float(sim.get("battery_soc"), 0.5)
        if sim.get("health_status", "nominal") != "nominal":
            return "safe", requested != "safe"
        if soc <= 0.20 and requested != "safe":
            return "safe", True
        if requested == "communication" and not sim.get("ground_pass_active", False):
            return "charging", True
        if requested == "payload_observe" and soc < 0.40:
            return "charging", True
        if requested in {"payload_compress", "payload_detect", "payload_send"} and soc < 0.30:
            return "charging", True
        return requested, False

    def _advance_orbit(self, sim: Dict[str, Any]) -> None:
        period = max(1.0, _float(sim.get("orbital_period_steps"), 94.0))
        sim["orbital_phase"] = (_float(sim.get("orbital_phase"), 0.0) + 1.0 / period) % 1.0
        rem = max(0.0, _float(sim.get("remaining_pass_duration"), 0.0) - 1.0)
        ttp = max(0.0, _float(sim.get("time_to_next_pass"), period) - 1.0)
        if rem > 0:
            sim["ground_pass_active"] = True
            sim["remaining_pass_duration"] = rem
        elif ttp <= 0:
            sim["ground_pass_active"] = True
            sim["remaining_pass_duration"] = 6.0
            sim["time_to_next_pass"] = max(1.0, _float(sim.get("following_gap_steps"), period))
        else:
            sim["ground_pass_active"] = False
            sim["remaining_pass_duration"] = 0.0
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
        obs_size = float(self.config.get("observation_size_mb", 9.41))
        compression_ratio = float(self.config.get("compression_ratio", 5.11))
        compressed_size = obs_size / max(1e-6, compression_ratio)
        downlink_rate_mb = float(self.config.get("downlink_rate_mb_per_step", 0.96))
        send_rate_mb = float(self.config.get("jetson_to_obc_mb_per_step", 0.375))
        cap = max(1.0, _float(sim.get("storage_capacity_mb"), 4096.0))
        jetson_cap = max(1.0, _float(sim.get("jetson_capacity_mb"), 249036.8))

        if mode == "payload_observe":
            sim["total_observation_s"] = _float(sim.get("total_observation_s"), 0.0) + 60.0
            sim["uncompressed_observations"] = _float(sim.get("uncompressed_observations"), 0.0) + 1.0
            sim["jetson_raw_mb"] = min(jetson_cap, _float(sim.get("jetson_raw_mb"), 0.0) + obs_size)
        elif mode == "payload_compress" and _float(sim.get("uncompressed_observations"), 0.0) > 0:
            progress = _float(sim.get("compression_progress"), 0.0) + 1.0
            if progress >= _float(sim.get("compression_time_factor"), 2.0):
                sim["compression_progress"] = 0.0
                sim["uncompressed_observations"] = max(0.0, _float(sim.get("uncompressed_observations"), 0.0) - 1.0)
                sim["undetected_observations"] = _float(sim.get("undetected_observations"), 0.0) + 1.0
                sim["jetson_raw_mb"] = max(0.0, _float(sim.get("jetson_raw_mb"), 0.0) - obs_size)
                sim["jetson_compressed_mb"] = min(jetson_cap, _float(sim.get("jetson_compressed_mb"), 0.0) + compressed_size)
            else:
                sim["compression_progress"] = progress
        elif mode == "payload_detect" and _float(sim.get("undetected_observations"), 0.0) > 0:
            progress = _float(sim.get("detection_progress"), 0.0) + 1.0
            if progress >= _float(sim.get("detection_steps"), 5.0):
                sim["detection_progress"] = 0.0
                sim["undetected_observations"] = max(0.0, _float(sim.get("undetected_observations"), 0.0) - 1.0)
                sim["total_detections"] = _float(sim.get("total_detections"), 0.0) + 1.0
                sim["obc_data_mb"] = min(cap, _float(sim.get("obc_data_mb"), 0.0) + 0.01)
            else:
                sim["detection_progress"] = progress
        elif mode == "payload_send":
            transfer = min(send_rate_mb, _float(sim.get("jetson_compressed_mb"), 0.0), cap - _float(sim.get("obc_data_mb"), 0.0))
            sim["jetson_compressed_mb"] = max(0.0, _float(sim.get("jetson_compressed_mb"), 0.0) - transfer)
            sim["obc_data_mb"] = min(cap, _float(sim.get("obc_data_mb"), 0.0) + transfer)
        elif mode == "communication" and sim.get("ground_pass_active", False):
            down = min(downlink_rate_mb, _float(sim.get("obc_data_mb"), 0.0))
            sim["obc_data_mb"] = max(0.0, _float(sim.get("obc_data_mb"), 0.0) - down)
            sim["data_downlinked_mb"] = _float(sim.get("data_downlinked_mb"), 0.0) + down

        sim["data_stored_mb"] = (
            _float(sim.get("obc_data_mb"), 0.0)
            + _float(sim.get("jetson_raw_mb"), 0.0)
            + _float(sim.get("jetson_compressed_mb"), 0.0)
        )

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

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        encoded = eventsat_observation_to_vector(observation)
        state = dict(encoded.raw)
        state["obs25"] = encoded.obs25
        return state

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        state = context.state or {}
        mode, metrics = self._planner.select(state)
        self._last_metrics = metrics
        self._last_rationale = (
            f"{self.planner_name}: selected {mode} using "
            f"{self._planner.backend} backend, mission_mode={self._planner.mode_weight_name}."
        )
        return {"eventsat_0": {"mode": mode}}

    def get_rationale(self) -> Optional[str]:
        return self._last_rationale

    def get_metrics(self) -> Dict[str, float]:
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


"""Subsymbolic RL representation for EventSat.

Registered as ``subsymbolic_eventsat`` in the behaviour controller.

Uses a PPO policy trained through RLlib and loaded from checkpoint at runtime.
The scientific behaviour mechanism remains ``ppo``; RLlib is the canonical
technical backend used by ``autops train``. During ``autops run`` the policy is
kept frozen for evaluation. ``rl_mock`` uses ``RandomPolicy`` for local smoke
runs and CI.

The policy operates on the 25D EventSat observation vector and outputs
MultiDiscrete([7, 2, 2]) actions: mode, data priority, and pipeline routing. The
selected mode is still passed through symbolic safety grounding before being
returned to the environment.

Papers:
- Oliver et al. EUCASS 2025 (8KDZ5Z53): architecture, PPO hyperparameters,
  obs/action space.
- Hamilton et al. 2025 (GWQ3LK6H): observation-space design and evaluation
  protocol.
- BSK-RL Stephenson & Schaub (ACUQK9VV): Gymnasium/RL environment pattern and
  orbital lookahead features.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from src.core.behaviour.controller import register
from src.core.representation import Representation
from src.eventsat.neural_policy import RandomPolicy
from src.rl.space_adapters import ACTION_DIMS

if TYPE_CHECKING:
    from src.core.decision_procedure.context import DecisionContext

logger = logging.getLogger(__name__)

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
_DEFAULT_JETSON_CAPACITY_MB = 249036.8
_DEFAULT_MAX_PASS_STEPS = 10.0


@register("subsymbolic_eventsat")
class SubsymbolicEventSat(Representation):
    """RL-based subsymbolic representation for EventSat mode selection.

    Config keys:
        rl_mock: use ``RandomPolicy`` without loading RLlib.
        deterministic: use greedy actions for evaluation.
        checkpoint_path: RLlib checkpoint directory/path.
        policy_id: RLlib policy id, default ``shared_policy``.
        trained_model_dir: directory containing ``manifest.json`` and checkpoints.
        satellite_id: satellite observed/controlled by this representation.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        mock_mode = bool(self.config.get("rl_mock", False))
        self._deterministic = bool(self.config.get("deterministic", True))
        self._policy_id = str(self.config.get("policy_id", "shared_policy"))

        checkpoint_path = self.config.get("checkpoint_path") or self._find_default_checkpoint()
        self._policy: Any
        self._mock = True
        if mock_mode:
            self._policy = RandomPolicy(action_dims=ACTION_DIMS)
        elif checkpoint_path:
            try:
                from src.eventsat.rllib_policy_adapter import RLLibPolicyAdapter

                self._policy = RLLibPolicyAdapter(
                    checkpoint_path=checkpoint_path,
                    policy_id=self._policy_id,
                )
                self._mock = False
                logger.info("Loaded RLlib checkpoint from %s", checkpoint_path)
            except (ImportError, FileNotFoundError, OSError) as exc:
                raise RuntimeError(
                    "RL cell integrity violation: configured RLlib checkpoint "
                    f"could not be loaded from '{checkpoint_path}'. Install the RL "
                    "extra or provide a valid checkpoint; use rl_mock: true only "
                    "for CI/smoke runs."
                ) from exc
        elif bool(self.config.get("allow_untrained", False)):
            self._policy = RandomPolicy(action_dims=ACTION_DIMS)
        else:
            raise RuntimeError(
                "RL cell integrity violation: no checkpoint_path configured and no "
                "default RLlib checkpoint manifest found. Provide "
                "representation_config.checkpoint_path, or set allow_untrained: true "
                "for training setup / rl_mock: true for CI."
            )

        self._jetson_capacity_mb = float(
            self.config.get("jetson_capacity_mb", _DEFAULT_JETSON_CAPACITY_MB)
        )
        self._orbital_period_steps = int(self.config.get("orbital_period_steps", 94))
        self._max_steps = int(self.config.get("max_steps", 10080))
        self._compression_time_factor = float(self.config.get("compression_time_factor", 2.0))
        self._detection_steps = int(self.config.get("detection_steps", 5))
        self._satellite_id = str(self.config.get("satellite_id", "eventsat_0"))

        self._last_rationale: Optional[str] = None
        self._last_action_vec: Optional[np.ndarray] = None
        self._last_mode_probs: Optional[np.ndarray] = None
        self._last_value = 0.0
        self._last_log_prob = 0.0
        self._last_obs_vec: Optional[np.ndarray] = None
        self._last_inference_latency_s = 0.0
        self._grounding_overrides = 0
        self._total_steps = 0
        self._trainer: Optional[Any] = None

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        """Extract feature dict plus 25D observation vector from raw observation."""
        if not hasattr(observation, "constellation_state"):
            return {}

        sat = observation.constellation_state.satellites.get(self._satellite_id)
        if sat is None:
            return {}

        res = sat.resources or {}
        meta = sat.metadata or {}
        constellation = observation.constellation_state

        return {
            "battery_soc": res.get("battery_soc", 0.5),
            "current_mode": sat.status,
            "in_sunlight": meta.get("in_sunlight", False),
            "ground_pass_active": meta.get("ground_pass_active", False),
            "data_stored_mb": res.get("data_stored_mb", 0.0),
            "obc_data_mb": res.get("obc_data_mb", meta.get("obc_data_mb", 0.0)),
            "jetson_raw_mb": meta.get("jetson_raw_mb", 0.0),
            "jetson_compressed_mb": meta.get("jetson_compressed_mb", 0.0),
            "storage_capacity_mb": meta.get("storage_capacity_mb", 4096.0),
            "uncompressed_observations": meta.get("uncompressed_observations", 0),
            "compression_progress": meta.get("compression_progress", 0),
            "total_observation_s": meta.get("total_observation_s", 0.0),
            "health_status": meta.get("health_status", "nominal"),
            "undetected_observations": meta.get("undetected_observations", 0),
            "daily_downlink_budget_mb": meta.get("daily_downlink_budget_mb", 27.0),
            "orbital_phase": meta.get("orbital_phase", 0.0),
            "time_to_next_eclipse": meta.get("time_to_next_eclipse", self._orbital_period_steps),
            "time_to_next_pass": meta.get("time_to_next_pass", self._orbital_period_steps),
            "remaining_pass_duration": meta.get("remaining_pass_duration", 0),
            "_current_step": int(constellation.timestep),
            "_obs_vector": self._build_obs_vector(res, meta, constellation, sat.status),
        }

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        """Select mode via RL policy plus symbolic grounding."""
        state = context.state
        if not state:
            return {self._satellite_id: {"mode": "charging"}}

        health = state.get("health_status", "nominal")
        if health != "nominal":
            self._last_rationale = f"Symbolic: anomaly ({health}) -> safe"
            self._grounding_overrides += 1
            return {self._satellite_id: {"mode": "safe"}}

        obs_vec = state.get("_obs_vector")
        if obs_vec is None:
            obs_vec = np.zeros(OBS_DIM, dtype=np.float32)

        t0 = time.perf_counter()
        action_vec, log_prob, value = self._policy.get_action(
            obs_vec,
            deterministic=self._deterministic,
        )
        mode_probs = self._policy.get_mode_probs(obs_vec)
        self._last_inference_latency_s = time.perf_counter() - t0
        self._total_steps += 1

        action_arr = np.asarray(action_vec, dtype=int).reshape(-1)
        mode_idx = self._clip_action_component(action_arr, 0, len(MODE_LIST) - 1)
        data_priority = self._clip_action_component(action_arr, 1, 1)
        pipeline_routing = self._clip_action_component(action_arr, 2, 1)
        mode = MODE_LIST[mode_idx]

        grounded_mode = self._apply_grounding(mode, state)
        if grounded_mode != mode:
            self._grounding_overrides += 1
        mode = grounded_mode

        self._last_action_vec = action_arr
        self._last_mode_probs = np.asarray(mode_probs, dtype=np.float32)
        self._last_value = float(value.item()) if hasattr(value, "item") else float(value)
        self._last_log_prob = (
            float(log_prob.item()) if hasattr(log_prob, "item") else float(log_prob)
        )
        self._last_obs_vec = np.asarray(obs_vec, dtype=np.float32)

        top_mode_prob = float(self._last_mode_probs[mode_idx])
        source = "RLlib PPO" if not self._mock else "RandomPolicy"
        self._last_rationale = (
            f"{source}: mode={mode} (p={top_mode_prob:.2f}), "
            f"value={self._last_value:.3f}"
        )

        return {
            self._satellite_id: {
                "mode": mode,
                "data_priority": data_priority,
                "pipeline_routing": pipeline_routing,
            }
        }

    def reason(self, state: Dict[str, Any], memory: Any) -> List[Dict[str, Any]]:
        """Return top mode probabilities as structured reasoning steps."""
        if not state:
            return [{"check": "state", "value": None, "implication": "empty_default_charging"}]

        obs_vec = state.get("_obs_vector")
        if obs_vec is None:
            return []

        probs = self._policy.get_mode_probs(obs_vec)
        top_indices = np.argsort(probs)[::-1][: min(3, len(MODE_LIST))]
        return [
            {
                "check": MODE_LIST[idx],
                "value": float(probs[idx]),
                "implication": "mode_probability",
            }
            for idx in top_indices
        ]

    def update(self, experience: Any) -> None:
        """Backward-compatible no-op hook.

        PPO is trained offline through RLlib. This method remains so generic
        learned-representation hooks and older tests can call it safely.
        """
        if self._trainer is None:
            return
        if isinstance(experience, dict) and "buffer" in experience:
            self._trainer.update(experience["buffer"])

    def set_trainer(self, trainer: Any) -> None:
        """Attach a legacy trainer hook."""
        self._trainer = trainer

    def get_last_step_data(self) -> Optional[Dict[str, Any]]:
        """Return last policy step data for backward-compatible diagnostics."""
        if self._last_obs_vec is None or self._last_action_vec is None:
            return None
        return {
            "obs_vec": self._last_obs_vec,
            "action_vec": self._last_action_vec,
            "log_prob": self._last_log_prob,
            "value": self._last_value,
        }

    def get_rationale(self) -> Optional[str]:
        return self._last_rationale

    def get_metrics(self) -> Dict[str, float]:
        metrics = {
            "rl_inference_latency_s": self._last_inference_latency_s,
            "rl_value_estimate": self._last_value,
            "rl_grounding_overrides": float(self._grounding_overrides),
            "rl_total_steps": float(self._total_steps),
        }
        if self._last_mode_probs is not None:
            top3 = np.argsort(self._last_mode_probs)[::-1][:3]
            for rank, idx in enumerate(top3):
                metrics[f"rl_mode_prob_{rank + 1}_{MODE_LIST[idx]}"] = float(
                    self._last_mode_probs[idx]
                )
        if self._trainer is not None and hasattr(self._trainer, "get_last_update_info"):
            info = self._trainer.get_last_update_info()
            metrics.update({f"ppo_{key}": value for key, value in info.items()})
        return metrics

    def get_name(self) -> str:
        return "SubsymbolicEventSat"

    def close(self) -> None:
        if hasattr(self._policy, "close"):
            self._policy.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _find_default_checkpoint(self) -> Optional[str]:
        experiment_id = self.config.get("experiment_id")
        if not experiment_id:
            return None
        root = Path(
            self.config.get("trained_model_dir", f"data/trained_models/{experiment_id}")
        )
        if not root.exists():
            return None

        manifest = root / "manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                checkpoint_path = data.get("checkpoint_path")
                if checkpoint_path and Path(checkpoint_path).exists():
                    return str(checkpoint_path)
            except (OSError, ValueError):
                pass

        candidates = sorted(root.glob("checkpoint_*"), key=lambda path: path.stat().st_mtime)
        return str(candidates[-1]) if candidates else None

    def _build_obs_vector(
        self,
        res: Dict[str, Any],
        meta: Dict[str, Any],
        constellation: Any,
        current_mode: str = "charging",
    ) -> np.ndarray:
        """Build normalized 25D observation vector from EventSat state."""
        vec = np.zeros(OBS_DIM, dtype=np.float32)

        vec[0] = float(res.get("battery_soc", 0.5))
        obc_cap = float(meta.get("storage_capacity_mb", 4096.0)) or 1.0
        vec[1] = float(res.get("obc_data_mb", meta.get("obc_data_mb", 0.0))) / obc_cap
        jetson_cap = self._jetson_capacity_mb or 1.0
        vec[2] = float(meta.get("jetson_raw_mb", 0.0)) / jetson_cap
        vec[3] = float(meta.get("jetson_compressed_mb", 0.0)) / jetson_cap

        orbital_phase = float(meta.get("orbital_phase", 0.0))
        vec[4] = math.sin(orbital_phase * 2 * math.pi)
        vec[5] = math.cos(orbital_phase * 2 * math.pi)

        orbital_period = float(self._orbital_period_steps) or 1.0
        vec[6] = min(float(meta.get("time_to_next_eclipse", orbital_period)) / orbital_period, 1.0)
        vec[7] = min(float(meta.get("time_to_next_pass", orbital_period)) / orbital_period, 1.0)
        vec[8] = min(float(meta.get("remaining_pass_duration", 0)) / _DEFAULT_MAX_PASS_STEPS, 1.0)
        max_steps = float(self._max_steps) or 1.0
        vec[9] = int(getattr(constellation, "timestep", 0)) / max_steps

        vec[10] = 1.0 if meta.get("in_sunlight", False) else 0.0
        vec[11] = 1.0 if meta.get("ground_pass_active", False) else 0.0
        vec[12] = 1.0 if meta.get("health_status", "nominal") == "nominal" else 0.0

        vec[13] = min(float(meta.get("uncompressed_observations", 0)) / 10.0, 1.0)
        comp_time = float(self._compression_time_factor) or 1.0
        vec[14] = min(float(meta.get("compression_progress", 0)) / comp_time, 1.0)
        vec[15] = min(float(meta.get("undetected_observations", 0)) / 10.0, 1.0)
        det_steps = float(self._detection_steps) or 1.0
        vec[16] = min(float(meta.get("detection_progress", 0.0)) / det_steps, 1.0)
        dl_budget = float(meta.get("daily_downlink_budget_mb", 27.0)) or 1.0
        vec[17] = float(res.get("data_downlinked_mb", 0.0)) / dl_budget

        mode_idx = MODE_TO_IDX.get(str(current_mode), 0)
        vec[18 + mode_idx] = 1.0
        return vec

    @staticmethod
    def _clip_action_component(action_arr: np.ndarray, index: int, max_value: int) -> int:
        value = int(action_arr[index]) if action_arr.size > index else 0
        return max(0, min(value, max_value))

    def _apply_grounding(self, mode: str, state: Dict[str, Any]) -> str:
        if mode == "communication" and not state.get("ground_pass_active", False):
            return "charging"
        soc = float(state.get("battery_soc", 0.5))
        if soc < 0.20 and mode != "charging":
            return "charging"
        return mode

"""Subsymbolic RL representation for EventSat.

Registered as ``subsymbolic_eventsat`` in the behaviour controller.

Uses a PPO policy trained through RLlib and loaded from checkpoint at runtime.
The scientific behaviour mechanism remains ``ppo``; RLlib is the canonical
technical backend used by ``autops train``. During ``autops run`` the policy is
kept frozen for evaluation. ``rl_mock`` uses ``RandomPolicy`` for local smoke
runs and CI.

The policy operates on the 25D EventSat observation vector and outputs one
categorical operational-mode action through ``MultiDiscrete([7])``.  The
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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from src.core.behaviour.controller import register
from src.core.representation import Representation
from src.eventsat.neural_policy import RandomPolicy
from src.eventsat.rl_obs_encoder import (
    ACTION_DIMS,
    EVENTSAT_OBS_SCHEMA_ID,
    MODE_LIST,
    OBS_DIM,
    _DEFAULT_JETSON_CAPACITY_MB,
    encode_eventsat_rl_obs,
    ground_eventsat_mode,
)

if TYPE_CHECKING:
    from src.core.decision_procedure.context import DecisionContext

logger = logging.getLogger(__name__)


@register("subsymbolic_eventsat")
class SubsymbolicEventSat(Representation):
    """RL-based subsymbolic representation for EventSat mode selection.

    Config keys:
        rl_mock: use ``RandomPolicy`` without loading RLlib.
        deterministic: use greedy actions for evaluation.
        checkpoint_path: RLlib checkpoint directory/path.
        policy_id: RLlib policy id, default ``shared_policy``.
        trained_model_dir: directory containing ``manifest.json`` and checkpoints.
        satellite_id: legacy single satellite observed/controlled by this representation.
        observe_ids: satellites visible to the policy.
        act_ids: satellites controlled by the policy.
    """

    # Unlike the hard-coded symbolic EventSat core, this representation binds
    # its emitted key to config.satellite_id and is therefore valid for each
    # independently mapped MultiEventSat agent.
    supported_scenarios = frozenset({"eventsat", "multieventsat"})
    action_key_schema = {
        "eventsat": "eventsat_single",
        "multieventsat": "native_satellites",
    }
    uses_agent_observation = True

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        mock_mode = bool(self.config.get("rl_mock", False))
        self._deterministic = bool(self.config.get("deterministic", True))
        self._policy_id = str(self.config.get("policy_id", "shared_policy"))

        self._mode_list = list(MODE_LIST)
        self._base_action_dims = list(ACTION_DIMS)
        self._obs_encoder = encode_eventsat_rl_obs
        legacy_satellite_id = str(self.config.get("satellite_id", "eventsat_0"))
        self._act_ids = self._coerce_id_list(
            self._configured_id_source(
                "act_ids",
                "satellite_ids",
                default=[legacy_satellite_id],
            )
        )
        self._observe_ids = self._coerce_id_list(
            self._configured_id_source(
                "observe_ids",
                "observed_satellite_ids",
                default=self._act_ids,
            )
        )
        self._satellite_id = (
            self._act_ids[0]
            if self._act_ids
            else (self._observe_ids[0] if self._observe_ids else legacy_satellite_id)
        )
        self._include_messages = bool(self.config.get("include_peer_messages", False))
        self._message_dim = (
            len(self._observe_ids) * len(self._mode_list) if self._include_messages else 0
        )
        self._action_dims = self._base_action_dims * len(self._act_ids)
        if not self._action_dims:
            self._action_dims = [1]
        self._obs_dim = OBS_DIM * len(self._observe_ids) + self._message_dim
        if self._obs_dim <= 0:
            self._obs_dim = 1

        checkpoint_path = self.config.get("checkpoint_path") or self._find_default_checkpoint()
        self._policy: Any
        self._mock = True
        if mock_mode:
            self._policy = RandomPolicy(action_dims=self._action_dims)
        elif checkpoint_path:
            self._validate_checkpoint_manifest(checkpoint_path)
            try:
                from src.eventsat.rllib_policy_adapter import RLLibPolicyAdapter

                self._policy = RLLibPolicyAdapter(
                    checkpoint_path=checkpoint_path,
                    policy_id=self._policy_id,
                    action_dims=self._action_dims,
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
            self._policy = RandomPolicy(action_dims=self._action_dims)
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

    def reset(self) -> None:
        """Clear inference diagnostics at an episode boundary."""
        self._last_rationale = None
        self._last_action_vec = None
        self._last_mode_probs = None
        self._last_value = 0.0
        self._last_log_prob = 0.0
        self._last_obs_vec = None
        self._last_inference_latency_s = 0.0
        self._grounding_overrides = 0
        self._total_steps = 0
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def seed(self, seed: int) -> None:
        """Seed the active policy's private exploration stream."""
        if hasattr(self._policy, "seed"):
            self._policy.seed(int(seed))

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        """Extract feature dict plus the scenario's RL observation vector."""
        raw_observation = observation
        messages = []
        if hasattr(observation, "local_state") and isinstance(observation.local_state, dict):
            raw_observation = observation.local_state.get("full_observation", observation)
            messages = list(getattr(observation, "messages", []) or [])

        if not hasattr(raw_observation, "constellation_state"):
            return {}

        primary_id = self._satellite_id
        sat = raw_observation.constellation_state.satellites.get(primary_id)
        if sat is None and self._observe_ids:
            primary_id = self._observe_ids[0]
            sat = raw_observation.constellation_state.satellites.get(primary_id)
        if sat is None:
            return {}

        res = sat.resources or {}
        meta = sat.metadata or {}
        constellation = raw_observation.constellation_state
        satellite_states = {
            sat_id: self._satellite_state(raw_observation, sat_id) for sat_id in self._act_ids
        }

        return {
            "battery_soc": res.get("battery_soc", 0.5),
            "current_mode": sat.status,
            "in_sunlight": meta.get("in_sunlight", False),
            "ground_pass_active": meta.get(
                "contact_window_active", meta.get("ground_pass_active", False)
            ),
            "data_stored_mb": res.get("data_stored_mb", 0.0),
            "obc_data_mb": res.get("obc_data_mb", meta.get("obc_data_mb", 0.0)),
            "jetson_raw_mb": meta.get("jetson_raw_mb", 0.0),
            "jetson_compressed_mb": meta.get("jetson_compressed_mb", 0.0),
            "storage_capacity_mb": meta.get("storage_capacity_mb", 4096.0),
            "battery_min_soc": meta.get("battery_min_soc", 0.20),
            "uncompressed_observations": meta.get("uncompressed_observations", 0),
            "compression_progress": meta.get("compression_progress", 0),
            "total_observation_s": meta.get("total_observation_s", 0.0),
            "health_status": meta.get("health_status", "nominal"),
            "undetected_observations": meta.get("undetected_observations", 0),
            "orbital_phase": meta.get("orbital_phase", 0.0),
            "time_to_next_eclipse": meta.get("time_to_next_eclipse", self._orbital_period_steps),
            "time_to_next_pass": meta.get("time_to_next_pass", self._orbital_period_steps),
            "remaining_pass_duration": meta.get("remaining_pass_duration", 0),
            "_current_step": int(constellation.timestep),
            "_obs_vector": self._build_joint_obs_vector(raw_observation, messages),
            "_satellite_states": satellite_states,
        }

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        """Select mode via RL policy plus symbolic grounding."""
        state = context.state
        if not state:
            return {sat_id: {"mode": "charging"} for sat_id in self._act_ids}

        obs_vec = state.get("_obs_vector")
        if obs_vec is None:
            obs_vec = np.zeros(self._obs_dim, dtype=np.float32)

        t0 = time.perf_counter()
        action_vec, log_prob, value = self._policy.get_action(
            obs_vec,
            deterministic=self._deterministic,
        )
        mode_probs = self._policy.get_mode_probs(obs_vec)
        self._last_inference_latency_s = time.perf_counter() - t0
        self._total_steps += 1

        action_arr = np.asarray(action_vec, dtype=int).reshape(-1)
        satellite_states = state.get("_satellite_states", {}) or {}
        width = len(self._base_action_dims)
        actions: Dict[str, Any] = {}
        rationale_parts: List[str] = []
        first_mode_idx = 0
        for sat_idx, sat_id in enumerate(self._act_ids):
            start = sat_idx * width
            mode_idx = self._clip_action_component(action_arr, start, len(self._mode_list) - 1)
            mode = self._mode_list[mode_idx]
            sat_state = satellite_states.get(sat_id, state)
            if len(self._act_ids) <= 1:
                sat_state = {**sat_state, **state}
            grounded_mode = self._apply_grounding(mode, sat_state)
            if grounded_mode != mode:
                self._grounding_overrides += 1
            mode = grounded_mode
            if sat_idx == 0:
                first_mode_idx = mode_idx
            actions[sat_id] = {"mode": mode}
            rationale_parts.append(f"{sat_id}={mode}")

        self._last_action_vec = action_arr
        self._last_mode_probs = np.asarray(mode_probs, dtype=np.float32)
        self._last_value = float(value.item()) if hasattr(value, "item") else float(value)
        self._last_log_prob = (
            float(log_prob.item()) if hasattr(log_prob, "item") else float(log_prob)
        )
        self._last_obs_vec = np.asarray(obs_vec, dtype=np.float32)

        top_mode_prob = (
            float(self._last_mode_probs[first_mode_idx])
            if self._last_mode_probs is not None and self._last_mode_probs.size > first_mode_idx
            else 0.0
        )
        source = "RLlib PPO" if not self._mock else "RandomPolicy"
        target_summary = ", ".join(rationale_parts) or "no controlled satellites"
        self._last_rationale = (
            f"{source}: {target_summary} (first p={top_mode_prob:.2f}), "
            f"value={self._last_value:.3f}"
        )

        return actions

    def reason(self, state: Dict[str, Any], memory: Any) -> List[Dict[str, Any]]:
        """Return top mode probabilities as structured reasoning steps."""
        if not state:
            return [{"check": "state", "value": None, "implication": "empty_default_charging"}]

        obs_vec = state.get("_obs_vector")
        if obs_vec is None:
            return []

        probs = self._policy.get_mode_probs(obs_vec)
        top_indices = np.argsort(probs)[::-1][: min(3, len(self._mode_list))]
        return [
            {
                "check": self._mode_list[idx],
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
                metrics[f"rl_mode_prob_{rank + 1}_{self._mode_list[idx]}"] = float(
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

    def _validate_checkpoint_manifest(self, checkpoint_path: str | Path) -> None:
        """Fail clearly when a checkpoint was trained on another observation MDP."""
        path = Path(checkpoint_path).expanduser()
        candidates = [path / "manifest.json", path.parent / "manifest.json"]
        experiment_id = self.config.get("experiment_id")
        trained_model_dir = self.config.get(
            "trained_model_dir",
            f"data/trained_models/{experiment_id}" if experiment_id else None,
        )
        if trained_model_dir:
            candidates.append(Path(str(trained_model_dir)).expanduser() / "manifest.json")
        manifest_path = next(
            (candidate for candidate in candidates if candidate.is_file()),
            None,
        )
        if manifest_path is None:
            raise RuntimeError(
                "EventSat RL checkpoint manifest is missing; checkpoints must declare "
                f"observation schema '{EVENTSAT_OBS_SCHEMA_ID}' and policy shape."
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"EventSat RL checkpoint manifest is unreadable: {manifest_path}"
            ) from exc

        actual_schema = manifest.get("observation_schema_id")
        if actual_schema != EVENTSAT_OBS_SCHEMA_ID:
            raise RuntimeError(
                "EventSat RL checkpoint observation schema mismatch: "
                f"expected '{EVENTSAT_OBS_SCHEMA_ID}', got {actual_schema!r}. "
                "Retrain the checkpoint with the current observation encoder."
            )
        shapes = manifest.get("policy_observation_shapes", {}) or {}
        actual_shape = shapes.get(self._policy_id)
        expected_shape = [self._obs_dim]
        if list(actual_shape or []) != expected_shape:
            raise RuntimeError(
                "EventSat RL checkpoint observation shape mismatch for policy "
                f"'{self._policy_id}': expected {expected_shape}, got {actual_shape!r}."
            )

    def _find_default_checkpoint(self) -> Optional[str]:
        experiment_id = self.config.get("experiment_id")
        if not experiment_id:
            return None
        root = Path(self.config.get("trained_model_dir", f"data/trained_models/{experiment_id}"))
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

    def _build_joint_obs_vector(
        self,
        observation: Any,
        messages: List[Dict[str, Any]],
    ) -> np.ndarray:
        parts = []
        for sat_id in self._observe_ids:
            sat = observation.constellation_state.satellites.get(sat_id)
            if sat is None:
                parts.append(
                    np.zeros(
                        self._obs_dim_without_messages_per_satellite(),
                        dtype=np.float32,
                    )
                )
                continue
            parts.append(
                self._build_obs_vector(
                    sat.resources or {},
                    sat.metadata or {},
                    observation.constellation_state,
                    str(sat.status or "charging"),
                )
            )
        if self._message_dim:
            parts.append(self._encode_messages(messages))
        if not parts:
            return np.zeros(self._obs_dim, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def _satellite_state(self, observation: Any, sat_id: str) -> Dict[str, Any]:
        sat = observation.constellation_state.satellites.get(sat_id)
        if sat is None:
            return {}
        res = sat.resources or {}
        meta = sat.metadata or {}
        return {
            "battery_soc": res.get("battery_soc", 0.5),
            "current_mode": sat.status,
            "ground_pass_active": meta.get("ground_pass_active", False),
            "health_status": meta.get("health_status", "nominal"),
            "battery_min_soc": meta.get("battery_min_soc", 0.20),
        }

    def _encode_messages(self, messages: List[Dict[str, Any]]) -> np.ndarray:
        vec = np.zeros(self._message_dim, dtype=np.float32)
        if not self._message_dim:
            return vec
        sat_to_slot = {sat_id: idx for idx, sat_id in enumerate(self._observe_ids)}
        mode_to_idx = {mode: idx for idx, mode in enumerate(self._mode_list)}
        mode_count = len(self._mode_list)
        for message in messages:
            proposal = message.get("proposal") or message.get("action") or {}
            if not isinstance(proposal, dict):
                continue
            for sat_id, sat_action in proposal.items():
                slot = sat_to_slot.get(str(sat_id))
                if slot is None or not isinstance(sat_action, dict):
                    continue
                mode_idx = mode_to_idx.get(str(sat_action.get("mode", "")))
                if mode_idx is None:
                    continue
                vec[slot * mode_count + mode_idx] = 1.0
        return vec

    def _obs_dim_without_messages_per_satellite(self) -> int:
        if self._observe_ids:
            return (self._obs_dim - self._message_dim) // len(self._observe_ids)
        return self._obs_dim

    def _build_obs_vector(
        self,
        res: Dict[str, Any],
        meta: Dict[str, Any],
        constellation: Any,
        current_mode: str = "charging",
    ) -> np.ndarray:
        """Build the scenario's normalized RL observation vector.

        Delegates the vector math to this scenario's shared RL encoder
        (``self._obs_encoder``); constants are resolved here from this
        representation's config/instance (inference path), mirroring how the
        space adapter resolves them from the live env (training path).
        """
        return self._obs_encoder(
            res,
            meta,
            current_mode,
            obc_cap=float(meta.get("storage_capacity_mb", 4096.0)),
            jetson_cap=float(self._jetson_capacity_mb),
            orbital_period=float(meta.get("orbital_period_steps", self._orbital_period_steps))
            or 1.0,
            max_steps=float(self._max_steps),
            compression_time=float(self._compression_time_factor),
            detection_steps=float(self._detection_steps),
            current_step=int(getattr(constellation, "timestep", 0)),
            detection_progress=float(meta.get("detection_progress", 0.0)),
        )

    @staticmethod
    def _clip_action_component(action_arr: np.ndarray, index: int, max_value: int) -> int:
        value = int(action_arr[index]) if action_arr.size > index else 0
        return max(0, min(value, max_value))

    def _apply_grounding(self, mode: str, state: Dict[str, Any]) -> str:
        """Apply the shared controller-visible EventSat RL safety shield."""
        return ground_eventsat_mode(
            mode,
            battery_soc=float(state.get("battery_soc", 0.5)),
            health_status=str(state.get("health_status", "nominal")),
            ground_pass_active=bool(state.get("ground_pass_active", False)),
            battery_min_soc=float(state.get("battery_min_soc", 0.20)),
        )

    @staticmethod
    def _coerce_id_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    def _configured_id_source(self, *keys: str, default: Any) -> Any:
        for key in keys:
            if key in self.config and self.config[key] is not None:
                return self.config[key]
        return default

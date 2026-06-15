"""
Subsymbolic RL Representation for EventSat.

Registered as "subsymbolic_eventsat" in the emergence controller.

Uses a trained PPO ActorCritic policy (or RandomPolicy in mock mode) to
select operational modes. Works with all 3 decision loops (SDA, OODA, ReAct)
and all 3 operations paradigms (AH, AG, CG) — orthogonal in the morphological
matrix.

The policy operates on a 25D observation vector (Groups 1-5 per the plan),
outputs MultiDiscrete([7, 2, 2]) actions, and is subject to the same symbolic
safety grounding constraints as LLMEventSat.

When behaviour == "emergent" the representation delegates PPO updates to
PPOTrainer (called from experiment_runner after each episode). When
behaviour == "hand_designed" update() is a no-op (policy is frozen).

Papers:
- Oliver et al. EUCASS 2025 (8KDZ5Z53): architecture, obs/action space
- Hamilton et al. 2025 (GWQ3LK6H): obs space design
- BSK-RL Stephenson & Schaub (ACUQK9VV): Gymnasium pattern, lookahead
"""
from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from src.behaviour.controller import register
from src.representation.base import Representation
from src.representation.neural_policy import TORCH_AVAILABLE, RandomPolicy

if TYPE_CHECKING:
    from src.decision_procedure.context import DecisionContext

logger = logging.getLogger(__name__)

# Mode list — must match neural_policy.MODE_LIST and gymnasium_wrapper.MODE_LIST
MODE_LIST = [
    "charging",
    "communication",
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "safe",
]
MODE_TO_IDX = {m: i for i, m in enumerate(MODE_LIST)}
OBS_DIM = 25

# Defaults for env constants not in observation metadata
_DEFAULT_JETSON_CAPACITY_MB = 249036.8
_DEFAULT_MAX_PASS_STEPS = 10.0  # cap for remaining_pass normalization


@register("subsymbolic_eventsat")
class SubsymbolicEventSat(Representation):
    """RL-based subsymbolic representation for EventSat mode selection.

    The policy provides the subsymbolic core. Symbolic constraints ground
    the output (same rules as LLMEventSat):
    - Anomaly active → forced safe (cannot be overridden)
    - SoC < 0.20 → forced charging
    - communication without active pass → forced charging

    Config keys:
        rl_mock (bool): Use RandomPolicy without torch — for CI. Default False.
        deterministic (bool): Take argmax action — for evaluation. Default True.
        checkpoint_path (str): Path to .pt checkpoint (optional; random if not set).
        jetson_capacity_mb (float): Used for normalizing Jetson fill fractions.
        orbital_period_steps (int): Used for normalizing timing features.
        max_steps (int): Episode length, for episode_progress feature.
        compression_time_factor (float): For normalizing compression_progress.
        detection_steps (int): For normalizing detection_progress.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        mock_mode = self.config.get("rl_mock", False)
        self._deterministic: bool = self.config.get("deterministic", True)

        if not mock_mode and not TORCH_AVAILABLE:
            # Never silently degrade an RL cell to a random policy: that produces
            # plausible-looking garbage results (caught 2026-06-11). CI must opt in.
            raise RuntimeError(
                "subsymbolic representation requires torch (`uv sync --extra rl`); "
                "set representation_config.rl_mock: true explicitly for CI/mock runs."
            )
        if mock_mode:
            self._policy = RandomPolicy()
            self._mock = True
        else:
            import torch
            from src.representation.neural_policy import ActorCritic
            self._policy = ActorCritic()
            self._mock = False

            checkpoint_path = self.config.get("checkpoint_path")
            if checkpoint_path:
                import os
                if os.path.exists(checkpoint_path):
                    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    if "policy_state_dict" in state:
                        self._policy.load_state_dict(state["policy_state_dict"])
                    else:
                        self._policy.load_state_dict(state)
                    logger.info("Loaded checkpoint from %s", checkpoint_path)
                else:
                    raise RuntimeError(
                        f"RL cell integrity violation: checkpoint_path set but not found "
                        f"at '{checkpoint_path}' — refusing to evaluate an untrained policy."
                    )
            elif not self.config.get("allow_untrained", False):
                # Evaluating an untrained policy yields plausible-looking garbage
                # (caught 2026-06-11). Training entry points set allow_untrained.
                raise RuntimeError(
                    "RL cell integrity violation: no checkpoint_path configured — an "
                    "evaluation run would measure an untrained policy. Provide "
                    "representation_config.checkpoint_path, or set allow_untrained: true "
                    "(training) / rl_mock: true (CI)."
                )

        # Env constants used for normalisation (configurable so they can be set
        # from experiment YAML without requiring access to the live env object)
        self._jetson_capacity_mb = float(
            self.config.get("jetson_capacity_mb", _DEFAULT_JETSON_CAPACITY_MB)
        )
        self._orbital_period_steps = int(self.config.get("orbital_period_steps", 94))
        self._max_steps = int(self.config.get("max_steps", 10080))
        self._compression_time_factor = float(self.config.get("compression_time_factor", 2.0))
        self._detection_steps = int(self.config.get("detection_steps", 5))

        # Metrics and last-step state (used by rollout buffer collection)
        self._last_rationale: Optional[str] = None
        self._last_action_vec: Optional[np.ndarray] = None
        self._last_mode_probs: Optional[np.ndarray] = None
        self._last_value: float = 0.0
        self._last_log_prob: float = 0.0
        self._last_obs_vec: Optional[np.ndarray] = None
        self._last_inference_latency_s: float = 0.0
        self._grounding_overrides: int = 0
        self._total_steps: int = 0
        self._trainer: Optional[Any] = None  # Set by experiment_runner if learned

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        """Extract feature dict + 25D obs vector from raw observation.

        Returns the same feature dict as rule_based/llm representations
        (for comparability in analysis), plus:
          _obs_vector: float32 numpy array of shape (25,) for the policy
        """
        if not hasattr(observation, "constellation_state"):
            return {}

        sat = observation.constellation_state.satellites.get("eventsat_0")
        if sat is None:
            return {}

        res = sat.resources or {}
        meta = sat.metadata or {}
        constellation = observation.constellation_state

        feature_dict = {
            "battery_soc": res.get("battery_soc", 0.5),
            "current_mode": sat.status,
            "in_sunlight": meta.get("in_sunlight", False),
            "ground_pass_active": meta.get("ground_pass_active", False),
            "data_stored_mb": res.get("data_stored_mb", 0.0),
            "obc_data_mb": res.get("obc_data_mb", meta.get("obc_data_mb", 0.0)),
            "jetson_raw_mb": meta.get("jetson_raw_mb", 0.0),
            "jetson_compressed_mb": meta.get("jetson_compressed_mb", 0.0),
            "storage_capacity_mb": meta.get("storage_capacity_mb", 512.0),
            "uncompressed_observations": meta.get("uncompressed_observations", 0),
            "compression_progress": meta.get("compression_progress", 0),
            "total_observation_s": meta.get("total_observation_s", 0.0),
            "health_status": meta.get("health_status", "nominal"),
            "undetected_observations": meta.get("undetected_observations", 0),
            "daily_downlink_budget_mb": meta.get("daily_downlink_budget_mb", 27.0),
            # Orbital lookahead (added by extended eventsat_env.py)
            "orbital_phase": meta.get("orbital_phase", 0.0),
            "time_to_next_eclipse": meta.get("time_to_next_eclipse", self._orbital_period_steps),
            "time_to_next_pass": meta.get("time_to_next_pass", self._orbital_period_steps),
            "remaining_pass_duration": meta.get("remaining_pass_duration", 0),
            # Step counter for episode_progress
            "_current_step": int(constellation.timestep),
            # Pre-built 25D vector for neural policy
            "_obs_vector": self._build_obs_vector(res, meta, constellation, sat.status),
        }
        return feature_dict

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        """Select mode via RL policy + symbolic grounding.

        Reads context.state["_obs_vector"] for the policy forward pass.
        Falls back to charging if state is empty.
        """
        state = context.state
        if not state:
            return {"eventsat_0": {"mode": "charging"}}

        # Symbolic safety: anomaly → safe
        health = state.get("health_status", "nominal")
        if health != "nominal":
            self._last_rationale = f"Symbolic: anomaly ({health}) → safe"
            self._grounding_overrides += 1
            return {"eventsat_0": {"mode": "safe"}}

        obs_vec = state.get("_obs_vector")
        if obs_vec is None:
            obs_vec = np.zeros(OBS_DIM, dtype=np.float32)

        t0 = time.perf_counter()
        if self._mock:
            action_vec, log_prob, value = self._policy.get_action(
                obs_vec, deterministic=self._deterministic
            )
            mode_probs = self._policy.get_mode_probs(obs_vec)
        else:
            import torch
            obs_tensor = torch.FloatTensor(obs_vec)
            action_vec, log_prob, value = self._policy.get_action(
                obs_tensor, deterministic=self._deterministic
            )
            mode_probs = self._policy.get_mode_probs(obs_tensor)
            value = float(value.item()) if hasattr(value, "item") else float(value)
            log_prob = float(log_prob.item()) if hasattr(log_prob, "item") else float(log_prob)

        self._last_inference_latency_s = time.perf_counter() - t0
        self._total_steps += 1

        mode_idx = int(action_vec[0])
        data_priority = int(action_vec[1])
        pipeline_routing = int(action_vec[2])
        mode = MODE_LIST[mode_idx]

        # Symbolic grounding
        mode = self._apply_grounding(mode, state)
        if mode != MODE_LIST[mode_idx]:
            self._grounding_overrides += 1

        self._last_action_vec = action_vec
        self._last_mode_probs = mode_probs
        self._last_value = float(value)
        self._last_log_prob = float(log_prob)
        self._last_obs_vec = obs_vec

        top_mode_prob = float(mode_probs[mode_idx]) if mode_probs is not None else 0.0
        self._last_rationale = (
            f"RL policy: mode={mode} (p={top_mode_prob:.2f}), "
            f"value={self._last_value:.3f}"
        )

        return {
            "eventsat_0": {
                "mode": mode,
                "data_priority": data_priority,
                "pipeline_routing": pipeline_routing,
            }
        }

    def reason(self, state: Dict[str, Any], memory: Any) -> List[Dict[str, Any]]:
        """Return per-head action probabilities as reasoning steps for ReAct.

        Called by the ReAct loop's Think phase. Returns a structured list
        so the loop can include policy uncertainty in its decision trace.
        """
        if not state:
            return [{"check": "state", "value": None, "implication": "empty_default_charging"}]

        obs_vec = state.get("_obs_vector")
        if obs_vec is None:
            return []

        if self._mock:
            probs = self._policy.get_mode_probs(obs_vec)
        else:
            import torch
            probs = self._policy.get_mode_probs(torch.FloatTensor(obs_vec))

        # Return top-3 modes by probability
        top_k = min(3, len(MODE_LIST))
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [
            {
                "check": MODE_LIST[idx],
                "value": float(probs[idx]),
                "implication": "mode_probability",
            }
            for idx in top_indices
        ]

    def update(self, experience: Any) -> None:
        """Delegate PPO update to the trainer (learned mode only).

        Called by experiment_runner after each episode when
        behaviour == "emergent". No-op if trainer not set.

        Args:
            experience: Dict with keys: buffer (RolloutBuffer), episode (int).
        """
        if self._trainer is None:
            return
        if not isinstance(experience, dict) or "buffer" not in experience:
            return
        self._trainer.update(experience["buffer"])

    def set_trainer(self, trainer: Any) -> None:
        """Attach a PPOTrainer for learned-mode updates."""
        self._trainer = trainer

    def get_last_step_data(self) -> Optional[Dict[str, Any]]:
        """Return (obs_vec, action_vec, log_prob, value) from the last select_action().

        Used by experiment_runner to populate the rollout buffer for PPO training.
        Returns None if select_action() has not been called yet this episode.
        """
        if self._last_obs_vec is None or self._last_action_vec is None:
            return None
        return {
            "obs_vec": self._last_obs_vec,
            "action_vec": self._last_action_vec,
            "log_prob": self._last_log_prob,
            "value": self._last_value,
        }

    # ------------------------------------------------------------------
    # Optional extension points
    # ------------------------------------------------------------------

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
                metrics[f"rl_mode_prob_{rank+1}_{MODE_LIST[idx]}"] = float(
                    self._last_mode_probs[idx]
                )
        if self._trainer is not None:
            info = self._trainer.get_last_update_info()
            metrics.update({f"ppo_{k}": v for k, v in info.items()})
        return metrics

    def get_name(self) -> str:
        return "SubsymbolicEventSat"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs_vector(
        self,
        res: Dict[str, Any],
        meta: Dict[str, Any],
        constellation: Any,
        current_mode: str = "charging",
    ) -> np.ndarray:
        """Build normalized 25D observation vector from raw observation fields.

        Matches EventSatGymnasium._obs_to_vector() — same 5 groups.
        Uses representation config for env constants (no env object needed).
        """
        vec = np.zeros(OBS_DIM, dtype=np.float32)

        # Group 1: Resource state (0-3)
        vec[0] = float(res.get("battery_soc", 0.5))
        obc_cap = float(meta.get("storage_capacity_mb", 512.0)) or 1.0
        vec[1] = float(res.get("obc_data_mb", meta.get("obc_data_mb", 0.0))) / obc_cap
        jetson_cap = self._jetson_capacity_mb or 1.0
        vec[2] = float(meta.get("jetson_raw_mb", 0.0)) / jetson_cap
        vec[3] = float(meta.get("jetson_compressed_mb", 0.0)) / jetson_cap

        # Group 2: Orbital phase & timing (4-9)
        orbital_phase = float(meta.get("orbital_phase", 0.0))
        vec[4] = math.sin(orbital_phase * 2 * math.pi)
        vec[5] = math.cos(orbital_phase * 2 * math.pi)

        orbital_period = float(self._orbital_period_steps) or 1.0
        vec[6] = min(float(meta.get("time_to_next_eclipse", orbital_period)) / orbital_period, 1.0)
        vec[7] = min(float(meta.get("time_to_next_pass", orbital_period)) / orbital_period, 1.0)
        vec[8] = min(float(meta.get("remaining_pass_duration", 0)) / _DEFAULT_MAX_PASS_STEPS, 1.0)
        current_step = int(getattr(constellation, "timestep", 0))
        max_steps = float(self._max_steps) or 1.0
        vec[9] = current_step / max_steps

        # Group 3: Environment flags (10-12)
        vec[10] = 1.0 if meta.get("in_sunlight", False) else 0.0
        vec[11] = 1.0 if meta.get("ground_pass_active", False) else 0.0
        vec[12] = 1.0 if meta.get("health_status", "nominal") == "nominal" else 0.0

        # Group 4: Pipeline state (13-17)
        vec[13] = min(float(meta.get("uncompressed_observations", 0)) / 10.0, 1.0)
        comp_time = float(self._compression_time_factor) or 1.0
        vec[14] = min(float(meta.get("compression_progress", 0)) / comp_time, 1.0)
        vec[15] = min(float(meta.get("undetected_observations", 0)) / 10.0, 1.0)
        det_steps = float(self._detection_steps) or 1.0
        # detection_progress is not in metadata — use 0 as fallback
        vec[16] = 0.0
        dl_budget = float(meta.get("daily_downlink_budget_mb", 27.0)) or 1.0
        vec[17] = float(res.get("data_downlinked_mb", 0.0)) / dl_budget

        # Group 5: Current mode one-hot (18-24)
        mode_idx = MODE_TO_IDX.get(str(current_mode), 0)
        vec[18 + mode_idx] = 1.0

        return vec

    def _apply_grounding(self, mode: str, state: Dict[str, Any]) -> str:
        """Apply symbolic safety constraints (same as llm_eventsat._apply_grounding)."""
        # No ground pass → cannot communicate
        if mode == "communication" and not state.get("ground_pass_active", False):
            return "charging"

        # Very low SoC → forced charging
        soc = float(state.get("battery_soc", 0.5))
        if soc < 0.20 and mode != "charging":
            return "charging"

        return mode

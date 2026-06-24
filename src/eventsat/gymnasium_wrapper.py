"""
Gymnasium Wrapper for EventSat Environment.

Wraps EventSatEnvironment as a standard gymnasium.Env with:
- observation_space: Box(25,) normalized to [0, 1]
- action_space: Discrete(7) operational modes

The observation vector is a 25D normalized feature vector designed per the
EUCASS 2025 paper (Oliver et al.) and Hamilton et al. 2025 observation space
ablation study:

  Group 1 — Resource state (4D): battery_soc, obc_fill, jetson_raw_fill, jetson_compressed_fill
  Group 2 — Orbital phase & timing (6D): sin/cos orbital_phase, time_to_eclipse, time_to_pass,
             remaining_pass_duration, episode_progress
  Group 3 — Environment flags (3D): in_sunlight, ground_pass_active, health_nominal
  Group 4 — Pipeline state (5D): uncompressed_obs, compression_progress, undetected_obs,
             detection_progress, downlink_utilization
  Group 5 — Current mode one-hot (7D)

Symbolic safety constraints (same as llm_eventsat._apply_grounding):
  - Anomaly active → forced safe mode (cannot be overridden by RL policy)
  - SoC < 0.20 → forced charging
  - Communication without active pass → forced charging

The reward is a scalar sum of the EventSatRewardFunction output, enabling
standard PPO training without modifications to the existing reward function.

Papers:
- Oliver et al. EUCASS 2025 (8KDZ5Z53): observation/action space design
- Hamilton et al. 2025 (GWQ3LK6H): task-relevant sensor ablation
- BSK-RL Stephenson & Schaub (ACUQK9VV): Gymnasium wrapper pattern,
  OpportunityProperties lookahead, Eclipse timing features
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Gymnasium import — optional to allow tests without gym installed
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None  # type: ignore
    spaces = None  # type: ignore

from src.eventsat.env import EventSatEnvironment

# Mode index mapping — must match VALID_MODES order used in the one-hot encoding
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


class EventSatGymnasium:
    """Gymnasium-compatible wrapper for EventSatEnvironment.

    Provides standard gym.Env interface for PPO training. Uses duck typing
    so it can be imported even when gymnasium is not installed (raises
    ImportError only on instantiation).

    Args:
        env_config: Config dict passed to EventSatEnvironment. Must include
            scenario_config and all standard EventSat parameters.
        max_episode_steps: Override max_steps from env_config.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: Dict[str, Any], max_episode_steps: Optional[int] = None) -> None:
        if not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "gymnasium is required for EventSatGymnasium. "
                "Install with: uv sync --extra rl"
            )
        if max_episode_steps is not None:
            env_config = dict(env_config)
            env_config["max_steps"] = max_episode_steps

        self._env = EventSatEnvironment(env_config)

        # Observation space: 25D float32, nominally [0, 1]
        # Some features (downlink_utilization, orbital timing) can exceed 1.0;
        # clipped to [0, 2] for safety.
        obs_low = np.zeros(OBS_DIM, dtype=np.float32)
        obs_high = np.ones(OBS_DIM, dtype=np.float32) * 2.0
        obs_low[4:6] = -1.0  # sin/cos orbital phase
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        # Action space: Discrete(7) operational modes
        self.action_space = spaces.Discrete(len(MODE_LIST))

        self._current_step: int = 0
        self._max_steps: int = self._env.max_steps

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return (obs_vector, info)."""
        if seed is not None:
            self.action_space.seed(seed)
        self._current_step = 0
        obs = self._env.reset(seed=seed)
        obs_vec = self._obs_to_vector(obs)
        return obs_vec, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply action and return (obs, reward, terminated, truncated, info).

        Args:
            action: integer mode index from Discrete(7)

        Returns:
            obs_vec: 25D float32 observation vector
            reward: scalar reward
            terminated: True when episode done
            truncated: False (no time limit beyond max_steps)
            info: dict with step metadata
        """
        action_arr = np.asarray(action, dtype=int)
        mode_idx = int(action_arr.item() if action_arr.shape == () else action_arr.reshape(-1)[0])

        # Build env action dict with symbolic grounding
        mode = self._apply_symbolic_grounding(mode_idx)

        env_action = {"eventsat_0": {"mode": mode}}

        result = self._env.step(env_action)
        self._current_step += 1

        obs_vec = self._obs_to_vector(result.observation)
        reward = float(sum(result.rewards.values()))
        terminated = result.done
        truncated = False

        return obs_vec, reward, terminated, truncated, result.info

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _obs_to_vector(self, obs: Any) -> np.ndarray:
        """Encode EnvironmentObservation into 25D normalized float32 vector.

        Implements the 5-group observation space from the plan:
          Group 1 (4D): resource fill fractions
          Group 2 (6D): orbital phase + timing lookahead
          Group 3 (3D): binary environment flags
          Group 4 (5D): pipeline state
          Group 5 (7D): current mode one-hot

        Design grounded in: Oliver et al. EUCASS 2025 (orbital params + resources),
        Hamilton et al. 2025 (task-relevant sensors), BSK-RL (eclipse/pass lookahead).
        """
        vec = np.zeros(OBS_DIM, dtype=np.float32)

        env = self._env
        sat = obs.constellation_state.satellites.get("eventsat_0")
        if sat is None:
            return vec

        res = sat.resources or {}
        meta = sat.metadata or {}

        # ------ Group 1: Resource state (indices 0-3) ------
        vec[0] = float(res.get("battery_soc", 0.5))
        obc_cap = meta.get("storage_capacity_mb", env.storage_capacity_mb) or 1.0
        vec[1] = float(res.get("obc_data_mb", 0.0)) / obc_cap
        jetson_cap = env.jetson_capacity_mb or 1.0
        vec[2] = float(meta.get("jetson_raw_mb", 0.0)) / jetson_cap
        vec[3] = float(meta.get("jetson_compressed_mb", 0.0)) / jetson_cap

        # ------ Group 2: Orbital phase & timing (indices 4-9) ------
        orbital_phase = float(meta.get("orbital_phase", 0.0))
        vec[4] = math.sin(orbital_phase * 2 * math.pi)
        vec[5] = math.cos(orbital_phase * 2 * math.pi)

        orbital_period = float(env.orbital_period_steps) or 1.0
        vec[6] = min(float(meta.get("time_to_next_eclipse", orbital_period)) / orbital_period, 1.0)
        vec[7] = min(float(meta.get("time_to_next_pass", orbital_period)) / orbital_period, 1.0)
        max_pass = 10.0  # typical max pass ~422s / 60s ≈ 7 steps; cap at 10
        vec[8] = min(float(meta.get("remaining_pass_duration", 0)) / max_pass, 1.0)
        max_steps = float(env.max_steps) or 1.0
        vec[9] = float(env.current_step) / max_steps

        # ------ Group 3: Environment flags (indices 10-12) ------
        vec[10] = 1.0 if meta.get("in_sunlight", False) else 0.0
        vec[11] = 1.0 if meta.get("ground_pass_active", False) else 0.0
        vec[12] = 1.0 if meta.get("health_status", "nominal") == "nominal" else 0.0

        # ------ Group 4: Pipeline state (indices 13-17) ------
        vec[13] = min(float(meta.get("uncompressed_observations", 0)) / 10.0, 1.0)
        comp_time = float(env.compression_time_factor) or 1.0
        vec[14] = min(float(meta.get("compression_progress", 0)) / comp_time, 1.0)
        vec[15] = min(float(meta.get("undetected_observations", 0)) / 10.0, 1.0)
        det_steps = float(env.detection_steps) or 1.0
        vec[16] = min(float(env.detection_progress) / det_steps, 1.0)
        dl_budget = float(meta.get("daily_downlink_budget_mb", 27.0)) or 1.0
        vec[17] = float(res.get("data_downlinked_mb", 0.0)) / dl_budget

        # ------ Group 5: Current mode one-hot (indices 18-24) ------
        current_mode = sat.status or "charging"
        mode_idx = MODE_TO_IDX.get(current_mode, 0)
        vec[18 + mode_idx] = 1.0

        return vec

    # ------------------------------------------------------------------
    # Symbolic grounding (same constraints as llm_eventsat._apply_grounding)
    # ------------------------------------------------------------------

    def _apply_symbolic_grounding(self, mode_idx: int) -> str:
        """Apply hard safety constraints to RL action before passing to env.

        These are the same constraints enforced by the environment's
        _resolve_mode() — applying them here makes the wrapper self-contained
        and avoids wasted env steps on always-overridden actions.
        """
        mode = MODE_LIST[mode_idx]
        env = self._env

        # Anomaly → safe (highest priority)
        if env.active_anomaly is not None:
            return "safe"

        # Very low SoC → forced charging
        if env.battery_soc < 0.20 and mode != "safe":
            return "charging"

        # No ground pass → cannot communicate
        if mode == "communication" and not env._is_ground_pass_active():
            return "charging"

        return mode

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def unwrapped(self) -> "EventSatGymnasium":
        return self

    def get_env(self) -> EventSatEnvironment:
        """Return the underlying EventSatEnvironment for direct access."""
        return self._env

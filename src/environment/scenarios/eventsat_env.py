"""
EventSat Environment -- Single-satellite operations simulation.

Physics model (sourced from PDR Nov 2025, ADCS thesis, and Part I Proposal Mar 2026):

- P1: Multi-step compression pipeline
      PDR: compression takes ~2x observation time (compression_time_factor = 2.0)
- P2: Mode transition overhead
      ADCS thesis: 135s attitude settling for maneuver modes (payload_observe, communication)
- P3: 3-pool data pipeline: Jetson raw → Jetson compressed → OBC → S-band downlink
      Measured data: 9.41 MB raw/obs (6.64 MB/42.36 s x 60 s), 1.84 MB compressed/obs (5.11:1)
      50 kbps Jetson→OBC internal bus (PDR)
- P4: Detection mode (payload_detect)
      PDR Section 3.2.3: CV inference on Jetson after compression; 5 min per observation.
      Produces small detection metadata (~0.01 MB) written to OBC.
      Tracked via detection_progress counter (like compression_progress).
- Thermal model removed (heat dissipation design in progress; not a constraint)
- Orbit at 400 km (Proposal Section 13: below 415 km preferred)

Pipeline backpressure:
  daily_downlink_budget_mb (configurable, default 27 MB/day from PDR Section 3.2.3 with GSaaS)
  is exposed in observation metadata so the agent can throttle observation.

Pipeline efficiency metric:
  total_pass_duration_s tracks cumulative ground contact time → max_achievable_downlink_mb
  = total_pass_duration_s x S-band rate. Published in step info for EventSatMetricsCollector.

Operational modes (VALID_MODES):
  charging, communication, payload_observe, payload_compress, payload_detect,
  payload_send, safe

Data transfer (payload_send):
  RS-485 one-way link: Jetson actively transmits, OBC listens.
  50 kbps (PDR) = 0.375 MB per 60s step.
  Agent must explicitly select this mode; transfer is NOT automatic.
  Detection metadata (0.01 MB) is written directly to OBC as part of detection
  completion — negligible at 50 kbps (< 2s to transmit).
"""
from __future__ import annotations
import logging, random
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from src.environment.satellite_env import (
    ConstellationState, EnvironmentObservation,
    SatelliteEnvironment, SatelliteState, StepResult,
)
from src.environment.orbital.context import OrbitalContext, compute_orbital_context
from src.environment.rewards import EventSatRewardFunction

logger = logging.getLogger(__name__)
VALID_MODES = {"charging", "communication", "payload_observe", "payload_compress", "payload_detect", "payload_send", "safe"}


class EventSatEnvironment(SatelliteEnvironment):
    """Single-satellite EventSat environment."""

    def __init__(self, config):
        self.config = config
        self.scenario = {}
        self._load_scenario(config)
        self.step_duration_s = config.get("step_duration_s", 60.0)
        self.max_steps = config.get("max_steps", 10080)
        self.current_step = 0

        # Orbit
        orb = self.scenario.get("orbit", {})
        self.orbital_period_s = orb.get("orbital_period_s", 5676)
        self.eclipse_fraction = orb.get("eclipse_fraction", 0.36)
        self.orbital_period_steps = max(1, int(self.orbital_period_s / self.step_duration_s))

        # Power
        pwr = self.scenario.get("power", {})
        solar_cfg = pwr.get("solar_panels", {})
        generation_peak_w = solar_cfg.get("generation_peak_w", solar_cfg.get("generation_sun_w", 24.0))
        panel_efficiency_factor = solar_cfg.get("panel_efficiency_factor", 1.0)
        self.solar_generation_w = generation_peak_w * panel_efficiency_factor
        bat = pwr.get("battery", {})
        self.battery_capacity_wh = bat.get("capacity_wh", 70.0)
        self.initial_soc = bat.get("initial_soc", 0.8)
        self.min_soc = bat.get("min_soc", 0.2)
        self.max_soc = bat.get("max_soc", 1.0)
        self.charge_efficiency = bat.get("charge_efficiency", 0.9)
        self.consumption = pwr.get("consumption", {})
        # Onboard-compute (Jetson) power overhead: a *Jetson-based* onboard core
        # (subsymbolic / hybrid onboard, paradigms AO/AH) needs the Jetson powered on
        # to run per-step inference. Jetson-on draws ~7 W (inference itself is
        # microseconds → no extra compute energy). This is ADDED to modes where the
        # Jetson would otherwise be off (charging, communication, safe). It is
        # NOT added during the Jetson-on payload modes (observe — the event camera hangs
        # off the Jetson — compress / detect / send), whose
        # per-mode consumption already includes the working Jetson — there the inference
        # simply runs before the task, no double-count. Symbolic onboard runs on the OBC
        # (sub-watt) and ground paradigms decide on the ground → no overhead. Set
        # per-episode by the runner via `onboard_compute_active`.
        self.onboard_compute_w = pwr.get("onboard_compute_w", 7.0)
        self.onboard_compute_active = False
        # Modes whose consumption already includes the working Jetson (no overhead added).
        self.jetson_active_modes = set(pwr.get("jetson_active_modes", [
            "payload_observe", "payload_compress", "payload_detect", "payload_send",
        ]))

        # Storage (3-pool pipeline)
        stor = self.scenario.get("storage", {})
        self.storage_capacity_mb = stor.get("obc_capacity_mb", 4*1024)       # OBC capacity, 50 % of 8 GB = 4096 MB
        self.jetson_capacity_mb = stor.get("jetson_capacity_mb", 249036.8)    # Jetson storage, 95% of 256 GB = 249036.8 MB
        self.observation_size_mb = stor.get("observation_size_mb", 9.41)  # Raw data size per observation (PDR measurement)
        # P3: compression ratio 
        self.compression_ratio = stor.get("compression_ratio", 1)  # For compatibility, but PDR measurement: 6.64 MB raw → 1.84 MB compressed = 5.11
        # P3: Jetson→OBC transfer rate (RS-485, 50 kbps per PDR)
        self.jetson_to_obc_rate_kbps = stor.get("jetson_to_obc_rate_kbps", 50)

        # Communications
        comm = self.scenario.get("communications", {})
        self.downlink_rate_kbps = comm.get("sband", {}).get("downlink_rate_kbps", 128)

        # Mode constraints
        modes_cfg = self.scenario.get("modes", {}).get("constraints", {})
        self.observe_min_soc = modes_cfg.get("payload_observe", {}).get("min_battery_soc", 0.4)
        self.compress_min_soc = modes_cfg.get("payload_compress", {}).get("min_battery_soc", 0.3)
        self.detect_min_soc = modes_cfg.get("payload_detect", {}).get("min_battery_soc", 0.3)
        self.send_min_soc = modes_cfg.get("payload_send", {}).get("min_battery_soc", 0.3)

        # P1: Compression pipeline (PDR: compression_time_factor = 2.0)
        payload_cfg = self.scenario.get("payload", {})
        self.compression_time_factor = payload_cfg.get("compression_time_factor", 2.0)
        # Detection pipeline: 5 min per observation (CV inference on Jetson)
        detection_time_s = payload_cfg.get("detection_time_s", 300.0)
        self.detection_steps = max(1, int(detection_time_s / self.step_duration_s))
        self.detection_metadata_mb = 0.01  # Small metadata output per detection

        # Communications: configurable daily downlink budget (PDR: 27 MB with GSaaS)
        passes_cfg = comm.get("passes", {})
        self.daily_downlink_budget_mb = passes_cfg.get("daily_downlink_budget_mb", 27.0)

        # P2: Mode transition overhead (ADCS thesis: 135s settling time)
        trans_cfg = self.scenario.get("modes", {}).get("transition_overhead", {})
        settling_s = trans_cfg.get("settling_time_s", 0.0)
        self.settling_time_steps = max(0, int(settling_s / self.step_duration_s))
        self.attitude_maneuver_modes = set(trans_cfg.get("attitude_maneuver_modes", []))

        # Anomaly injection
        self.anomaly_prob = config.get("anomaly_prob", 0.001)
        # When True, anomaly recovery requires a ground pass (conventional ops).
        # When False, onboard FDIR clears the anomaly once the countdown expires.
        self.anomaly_requires_ground_pass = config.get("anomaly_requires_ground_pass", True)

        # State (initialised in reset())
        self.battery_soc = self.initial_soc
        # 3-pool storage
        self.jetson_raw_mb = 0.0
        self.jetson_compressed_mb = 0.0
        self.obc_data_mb = 0.0
        self.data_stored_mb = 0.0       # total = jetson_raw + jetson_compressed + obc
        self.data_downlinked_mb = 0.0
        self.total_raw_captured_mb = 0.0
        self.obc_raw_equivalent_mb = 0.0
        self.downlink_raw_equivalent_mb = 0.0
        self.uncompressed_observations = 0
        self.total_observation_s = 0.0
        self.current_mode = "charging"
        # P1: compression progress counter
        self.compression_progress = 0
        # Detection state
        self.detection_progress = 0
        self.undetected_observations = 0
        self.total_detections = 0
        # C4: Track cumulative ground pass duration for pipeline efficiency
        self.total_pass_duration_s = 0.0
        # P2: transition state
        self.transition_steps_remaining = 0
        self.previous_mode = "charging"
        # Misc
        self._orbital_ctx: Optional[OrbitalContext] = None
        self._episode_orbit: Dict[str, Any] = {}
        self.active_anomaly = None
        # RL sub-actions (set in step(), initialised here for get_observation() safety)
        self._data_priority: int = 0
        self._pipeline_routing: int = 0
        self.forced_safe_steps = 0
        # Dedicated RNG for anomaly injection — isolated from the global stream
        # so that different recovery timings between ops paradigms don't desync
        # anomaly injection across architectures. Seeded per episode in reset().
        self._anomaly_rng: random.Random = random.Random()
        self.episode_reward = 0.0
        self._step_metrics = {}

        # Reward function (Individual Negative, from autops-rl)
        reward_cfg = config.get("reward_config", {})
        self.reward_fn = EventSatRewardFunction(reward_cfg)

        # Mission targets (scaled to episode length)
        objectives = self.scenario.get("objectives", {})
        mission_days = objectives.get("mission_duration_days", 90.0)
        episode_days = (self.max_steps * self.step_duration_s) / 86400.0
        target_scale = episode_days / mission_days if mission_days > 0 else 1.0
        self.obs_target_hours = objectives.get("total_observation_hours", 2.0) * target_scale
        self.dl_target_mb = objectives.get("min_downlinked_data_mb", 221.0) * target_scale

    def _load_scenario(self, config):
        scenario_path = config.get("scenario_config") or config.get("scenario_file")
        if scenario_path and Path(scenario_path).exists():
            with open(scenario_path) as f:
                self.scenario = yaml.safe_load(f) or {}
        else:
            self.scenario = config.get("scenario_params", {})

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.current_step = 0
        self.battery_soc = self.initial_soc
        # 3-pool storage reset
        self.jetson_raw_mb = 0.0
        self.jetson_compressed_mb = 0.0
        self.obc_data_mb = 0.0
        self.data_stored_mb = 0.0
        self.data_downlinked_mb = 0.0
        self.total_raw_captured_mb = 0.0
        self.obc_raw_equivalent_mb = 0.0
        self.downlink_raw_equivalent_mb = 0.0
        self.uncompressed_observations = 0
        self.total_observation_s = 0.0
        self.current_mode = "charging"
        self.compression_progress = 0
        self.detection_progress = 0
        self.undetected_observations = 0
        self.total_detections = 0
        self.total_pass_duration_s = 0.0
        self.transition_steps_remaining = 0
        self.previous_mode = "charging"
        self.active_anomaly = None
        self.forced_safe_steps = 0
        self._data_priority = 0
        self._pipeline_routing = 0
        self._anomaly_rng = random.Random(seed * 131 + 7919 if seed is not None else None)
        self.episode_reward = 0.0
        self._step_metrics = {}

        # Launch lottery: randomize RAAN, ArgP, TA per episode to simulate
        # rideshare insertion uncertainty (no onboard propulsion post-deployment).
        # Draws happen first so the RNG ordering is deterministic per seed.
        orbit_config = dict(self.scenario.get("orbit", {}))
        if orbit_config.get("launch_lottery", False):
            orbit_config["raan_deg"] = random.uniform(0, 360)
            orbit_config["arg_perigee_deg"] = random.uniform(0, 360)
            orbit_config["true_anomaly_deg"] = random.uniform(0, 360)
            logger.debug(
                "Launch lottery: RAAN=%.1f, ArgP=%.1f, TA=%.1f",
                orbit_config["raan_deg"],
                orbit_config["arg_perigee_deg"],
                orbit_config["true_anomaly_deg"],
            )

        from src.environment.orbital.propagator import is_available as _orekit_available
        self._orbital_ctx = compute_orbital_context(
            orbit_config=orbit_config,
            comms_config=self.scenario.get("communications", {}),
            step_s=self.step_duration_s,
            total_steps=self.max_steps,
            require_orekit=_orekit_available(),
        )
        # Persist the *actual* per-episode orbit (incl. lottery draws) and the
        # resulting pass schedule so analysis/figures reproduce the exact run
        # without having to replay the RNG draw order. See get_episode_orbit().
        self._episode_orbit = dict(orbit_config)
        return self.get_observation()

    def step(self, actions):
        sat_action = actions.get("eventsat_0", {})
        if isinstance(sat_action, dict):
            requested_mode = sat_action.get("mode", "charging")
            # RL sub-actions (MultiDiscrete): ignored by symbolic/LLM representations
            self._data_priority = int(sat_action.get("data_priority", 0))       # 0=normal, 1=urgent
            self._pipeline_routing = int(sat_action.get("pipeline_routing", 0)) # 0=compress_first, 1=detect_first
        else:
            requested_mode = "charging"
            self._data_priority = 0
            self._pipeline_routing = 0
        resolved_mode = self._resolve_mode(requested_mode)
        forced = resolved_mode != requested_mode

        # P2: Mode transition overhead
        in_transition = False
        if self.settling_time_steps > 0:
            if self.transition_steps_remaining > 0:
                # Already mid-transition: execute as charging (non-productive)
                effective_mode = "charging"
                self.transition_steps_remaining -= 1
                in_transition = True
                # On the last transition step, mark previous_mode as the target
                # so the next step doesn't re-trigger the transition
                if self.transition_steps_remaining == 0:
                    self.previous_mode = resolved_mode
            elif self._requires_attitude_maneuver(self.previous_mode, resolved_mode):
                # New transition needed: start it, first step is non-productive
                self.transition_steps_remaining = max(0, self.settling_time_steps - 1)
                effective_mode = "charging"
                in_transition = True
            else:
                effective_mode = resolved_mode
        else:
            effective_mode = resolved_mode

        in_sun = self._is_in_sunlight()
        pass_active = self._is_ground_pass_active()
        prev_soc = self.battery_soc

        self._update_battery(effective_mode, in_sun)
        # C4: Track cumulative ground *contact* time (sub-timestep accurate) for
        # pipeline efficiency / max-achievable downlink.
        if pass_active:
            self.total_pass_duration_s += self._contact_seconds()
        reward, action_info = self._apply_mode_effects(effective_mode, in_sun, pass_active)
        anomaly_event = self._maybe_inject_anomaly()

        self.current_mode = effective_mode
        self.current_step += 1
        self.episode_reward += reward

        # Update previous_mode only when not transitioning
        if not in_transition:
            self.previous_mode = effective_mode

        # Update total data_stored_mb (for metrics/rewards backward compat)
        self.data_stored_mb = self.jetson_raw_mb + self.jetson_compressed_mb + self.obc_data_mb

        self._step_metrics = {
            "reward": reward,
            "battery_soc": self.battery_soc,
            "prev_battery_soc": prev_soc,
            "data_stored_mb": self.data_stored_mb,
            "jetson_raw_mb": self.jetson_raw_mb,
            "jetson_compressed_mb": self.jetson_compressed_mb,
            "obc_data_mb": self.obc_data_mb,
            "data_downlinked_mb": self.data_downlinked_mb,
            "total_raw_captured_mb": self.total_raw_captured_mb,
            "obc_raw_equivalent_mb": self.obc_raw_equivalent_mb,
            "downlink_raw_equivalent_mb": self.downlink_raw_equivalent_mb,
            "in_sunlight": float(in_sun),
            "ground_pass_active": float(pass_active),
            "forced_mode": float(forced),
            "in_transition": in_transition,
            "anomaly": anomaly_event,
            "anomaly_forced_safe": float(self.active_anomaly is not None),
            "observation_hours": self.total_observation_s / 3600.0,
            "total_detections": self.total_detections,
            "undetected_observations": self.undetected_observations,
            "total_pass_duration_s": self.total_pass_duration_s,
            "max_achievable_downlink_mb": self.total_pass_duration_s * (self.downlink_rate_kbps / 8.0 / 1000.0),
        }
        return StepResult(
            observation=self.get_observation(),
            rewards={"total": reward},
            done=self.is_done(),
            info={
                "resolved_mode": effective_mode,
                "requested_mode": requested_mode,
                "forced": forced,
                # Pre-transition safety classification: resolved_mode BEFORE the
                # transition/settling mask (which reports effective_mode="charging"
                # during a forced-safe step's settling window). M-05/M-13 key off this
                # so an anomaly/critical-battery safe step is scored as a safety
                # override, never as a charging constraint violation.
                "safety_safe": float(resolved_mode == "safe"),
                "anomaly": anomaly_event,
                **action_info,           # per-step values (e.g. data_downlinked_mb per step)
                **self._step_metrics,    # cumulative values overwrite — data_downlinked_mb is always cumulative here
                "step_downlinked_mb": action_info.get("data_downlinked_mb", 0.0),
            },
        )

    def get_observation(self):
        in_sun = self._is_in_sunlight()
        pass_active = self._is_ground_pass_active()
        orbital_lookahead = self._compute_orbital_lookahead()
        sat = SatelliteState(
            satellite_id="eventsat_0",
            position=[0.0, 0.0, 500.0],
            velocity=[0.0, 0.0, 0.0],
            resources={
                "battery_soc": self.battery_soc,
                "data_stored_mb": self.data_stored_mb,
                "obc_data_mb": self.obc_data_mb,
                "data_downlinked_mb": self.data_downlinked_mb,
            },
            status=self.current_mode,
            metadata={
                "in_sunlight": in_sun,
                "ground_pass_active": pass_active,
                "uncompressed_observations": self.uncompressed_observations,
                "compression_progress": self.compression_progress,
                "total_observation_s": self.total_observation_s,
                "storage_capacity_mb": self.storage_capacity_mb,
                "jetson_raw_mb": self.jetson_raw_mb,
                "jetson_compressed_mb": self.jetson_compressed_mb,
                "obc_data_mb": self.obc_data_mb,
                "health_status": "nominal" if self.active_anomaly is None else self.active_anomaly,
                "undetected_observations": self.undetected_observations,
                "daily_downlink_budget_mb": self.daily_downlink_budget_mb,
                # Physical capacity the planner can actually downlink at the next pass
                # (50 kbps × next-pass contact seconds) — replaces the 27 MB heuristic.
                "achievable_downlink_mb": self._next_pass_capacity_mb(),
                # Orbital lookahead (RL observation space Groups 2, BSK-RL pattern)
                **orbital_lookahead,
            },
        )
        constellation = ConstellationState(
            timestep=self.current_step,
            epoch_seconds=self.current_step * self.step_duration_s,
            satellites={"eventsat_0": sat},
            global_info={"max_steps": self.max_steps},
        )
        tasks = self._generate_tasks(in_sun, pass_active)
        return EnvironmentObservation(
            constellation_state=constellation,
            tasks=tasks,
            events=[{"type": self.active_anomaly}] if self.active_anomaly else [],
        )

    def get_metrics(self):
        return {
            **self._step_metrics,
            "episode_reward": self.episode_reward,
            "total_observation_hours": self.total_observation_s / 3600.0,
            "total_downlinked_mb": self.data_downlinked_mb,
        }

    def get_episode_orbit(self) -> Dict[str, Any]:
        """Return the actual orbital elements used this episode.

        Includes the launch-lottery draws (RAAN/ArgP/TA) plus the fixed
        inclination/altitude/epoch from the scenario. Persisting these lets
        analysis and ground-track figures reproduce the exact orbit that was
        simulated, instead of re-deriving the RNG draw order (fragile — any
        change to ``reset()`` draw ordering silently desyncs figures).
        """
        return dict(self._episode_orbit)

    def get_ground_passes(self) -> List[Dict[str, Any]]:
        """Return this episode's ground-pass schedule as serializable dicts."""
        from dataclasses import asdict

        if self._orbital_ctx is None:
            return []
        return [asdict(gp) for gp in self._orbital_ctx.ground_passes]

    def is_done(self):
        return self.current_step >= self.max_steps

    def _is_in_sunlight(self):
        if self._orbital_ctx is None:
            return True
        return self._orbital_ctx.is_in_sunlight(self.current_step)

    def _is_ground_pass_active(self):
        if self._orbital_ctx is None:
            return False
        return self._orbital_ctx.is_ground_pass_active(self.current_step)

    def _contact_seconds(self):
        """Seconds of ground contact within the current step (≤ step_duration_s).
        Sub-timestep accurate: a short pass credits only its actual contact time."""
        if self._orbital_ctx is None:
            return 0.0
        return self._orbital_ctx.contact_seconds(self.current_step)

    def _next_pass_capacity_mb(self):
        """Physically achievable downlink at the next ground pass = downlink rate ×
        that pass's contact seconds. The capacity a planner can actually deliver for
        the gap it is scheduling (replaces the flat daily-budget heuristic)."""
        if self._orbital_ctx is None:
            return 0.0
        contact_s = self._orbital_ctx.next_pass_contact_s(self.current_step)
        return self.downlink_rate_kbps / 8.0 * contact_s / 1000.0

    def _requires_attitude_maneuver(self, from_mode: str, to_mode: str) -> bool:
        """Return True if switching from_mode→to_mode requires attitude settling (P2)."""
        if self.settling_time_steps == 0:
            return False
        if from_mode == to_mode:
            return False
        return to_mode in self.attitude_maneuver_modes or from_mode in self.attitude_maneuver_modes

    def _resolve_mode(self, requested):
        if requested not in VALID_MODES:
            requested = "charging"
        # Environment-enforced safe mode during active anomaly — agent cannot override.
        # Recovery requires a ground pass (see _maybe_inject_anomaly).
        if self.active_anomaly is not None:
            return "safe"
        if self.battery_soc <= self.min_soc and requested != "safe":
            return "safe"
        if requested == "payload_observe" and self.battery_soc < self.observe_min_soc:
            return "charging"
        if requested == "payload_compress" and self.battery_soc < self.compress_min_soc:
            return "charging"
        if requested == "payload_detect" and self.battery_soc < self.detect_min_soc:
            return "charging"
        if requested == "payload_send" and self.battery_soc < self.send_min_soc:
            return "charging"
        if requested == "communication" and not self._is_ground_pass_active():
            return "charging"
        return requested

    def _update_battery(self, mode, in_sun):
        phase = "sun_w" if in_sun else "eclipse_w"
        consumption_w = self.consumption.get(mode, {}).get(phase, 5.0)
        # A Jetson-based onboard core keeps the Jetson powered (+~7W), added to modes
        # where it would otherwise be off; the Jetson-compute modes already include it.
        if self.onboard_compute_active and mode not in self.jetson_active_modes:
            consumption_w += self.onboard_compute_w
        generation_w = self.solar_generation_w if in_sun else 0.0
        net_power_w = generation_w - consumption_w
        energy_delta_wh = net_power_w * (self.step_duration_s / 3600.0)
        if energy_delta_wh > 0:
            energy_delta_wh *= self.charge_efficiency
        soc_delta = energy_delta_wh / self.battery_capacity_wh
        # Hard cap at 1.0 regardless of max_soc config — SoC is a fraction
        self.battery_soc = max(0.0, min(1.0, self.battery_soc + soc_delta))

    def _transfer_jetson_to_obc(self):
        """P3: Jetson→OBC transfer via CAN bus (~1 MB/s; rate from config).

        No longer a bottleneck — the binding constraint is the OBC→S-band
        transmitter (50 kbps). Only called when agent selects payload_send mode.
        Returns the amount of data actually transferred (MB).
        """
        if self.jetson_compressed_mb <= 0.0:
            return 0.0
        transfer_mb = (self.jetson_to_obc_rate_kbps / 8.0) * (self.step_duration_s / 1000.0)
        transfer_mb = min(transfer_mb, self.jetson_compressed_mb)
        space_on_obc = self.storage_capacity_mb - self.obc_data_mb
        actual_transfer = min(transfer_mb, max(0.0, space_on_obc))
        self.jetson_compressed_mb -= actual_transfer
        self.obc_data_mb += actual_transfer
        self.obc_raw_equivalent_mb += actual_transfer * self.compression_ratio
        return actual_transfer

    def _apply_mode_effects(self, mode, in_sun, pass_active):
        """Apply state transitions and compute structured reward."""
        action_info = {"pass_active": pass_active}
        storage_overflow = False

        # Reset progress counters if agent switches away mid-process (penalizes thrashing)
        if mode != "payload_compress" and self.previous_mode == "payload_compress":
            self.compression_progress = 0
        if mode != "payload_detect" and self.previous_mode == "payload_detect":
            self.detection_progress = 0

        # --- State transitions ---
        if mode == "payload_observe":
            self.total_observation_s += self.step_duration_s
            self.uncompressed_observations += 1
            # P3: raw data goes to Jetson storage
            self.jetson_raw_mb += self.observation_size_mb
            self.total_raw_captured_mb += self.observation_size_mb
            if self.jetson_raw_mb > self.jetson_capacity_mb:
                self.jetson_raw_mb = self.jetson_capacity_mb
                storage_overflow = True
            action_info["storage_overflow"] = storage_overflow

        elif mode == "payload_compress":
            # RL sub-action: detect_first → if detection backlog exists and no compression backlog,
            # advance detection instead (agent wants to prioritize detection pipeline)
            if (
                getattr(self, "_pipeline_routing", 0) == 1
                and self.undetected_observations > 0
                and self.uncompressed_observations == 0
            ):
                # Redirect: treat as payload_detect step (detect_first routing)
                self.detection_progress += 1
                if self.detection_progress >= self.detection_steps:
                    self.undetected_observations -= 1
                    self.obc_data_mb += self.detection_metadata_mb
                    self.total_detections += 1
                    self.detection_progress = 0
                    action_info["detection_completed"] = True
                else:
                    action_info["detection_in_progress"] = True
                action_info["pipeline_routed_to_detect"] = True
                had_data = True
            else:
                had_data = self.uncompressed_observations > 0
            if had_data and not action_info.get("pipeline_routed_to_detect"):
                # P1: multi-step compression
                self.compression_progress += 1
                if self.compression_progress >= self.compression_time_factor:
                    # Compression complete: move from Jetson raw → Jetson compressed
                    self.uncompressed_observations -= 1
                    compressed_size = self.observation_size_mb / self.compression_ratio
                    self.jetson_raw_mb = max(0.0, self.jetson_raw_mb - self.observation_size_mb)
                    self.jetson_compressed_mb += compressed_size
                    self.compression_progress = 0
                    self.undetected_observations += 1
                    action_info["compression_completed"] = True
                else:
                    action_info["compression_in_progress"] = True
            else:
                self.compression_progress = 0
            action_info["had_data_to_compress"] = had_data

        elif mode == "payload_detect":
            # RL sub-action: compress_first → if compression backlog exists and no detection backlog,
            # advance compression instead
            if (
                getattr(self, "_pipeline_routing", 0) == 0
                and self.uncompressed_observations > 0
                and self.undetected_observations == 0
            ):
                # Redirect: treat as payload_compress step (compress_first routing)
                self.compression_progress += 1
                if self.compression_progress >= self.compression_time_factor:
                    self.uncompressed_observations -= 1
                    compressed_size = self.observation_size_mb / self.compression_ratio
                    self.jetson_raw_mb = max(0.0, self.jetson_raw_mb - self.observation_size_mb)
                    self.jetson_compressed_mb += compressed_size
                    self.compression_progress = 0
                    self.undetected_observations += 1
                    action_info["compression_completed"] = True
                else:
                    action_info["compression_in_progress"] = True
                action_info["pipeline_routed_to_compress"] = True
                had_data = True
            else:
                had_data = self.undetected_observations > 0
            if had_data and not action_info.get("pipeline_routed_to_compress"):
                self.detection_progress += 1
                if self.detection_progress >= self.detection_steps:
                    # Detection complete: produce small metadata → OBC
                    self.undetected_observations -= 1
                    self.obc_data_mb += self.detection_metadata_mb
                    self.total_detections += 1
                    self.detection_progress = 0
                    action_info["detection_completed"] = True
                else:
                    action_info["detection_in_progress"] = True
            else:
                self.detection_progress = 0
            action_info["had_data_to_detect"] = had_data

        elif mode == "payload_send":
            # RS-485: Jetson actively transmits compressed data to OBC
            had_data = self.jetson_compressed_mb > 0
            sent_mb = self._transfer_jetson_to_obc() if had_data else 0.0
            action_info["had_data_to_send"] = had_data
            action_info["data_sent_mb"] = sent_mb

        elif mode == "communication":
            if pass_active:
                # P3: downlink from OBC at the S-band protocol rate, over only the
                # seconds actually in contact this step (short passes downlink less).
                contact_s = self._contact_seconds()
                dl_mb = (self.downlink_rate_kbps / 8.0) * (contact_s / 1000.0)
                # RL sub-action: urgent data_priority → 1.5x downlink chunk
                if getattr(self, "_data_priority", 0) == 1:
                    dl_mb *= 1.5
                actual_dl = min(dl_mb, self.obc_data_mb)
                raw_equivalent_dl = 0.0
                if self.obc_data_mb > 0:
                    raw_equiv_fraction = self.obc_raw_equivalent_mb / self.obc_data_mb
                    raw_equivalent_dl = min(self.obc_raw_equivalent_mb, actual_dl * raw_equiv_fraction)
                self.obc_data_mb -= actual_dl
                self.obc_raw_equivalent_mb = max(0.0, self.obc_raw_equivalent_mb - raw_equivalent_dl)
                self.downlink_raw_equivalent_mb += raw_equivalent_dl
                self.data_downlinked_mb += actual_dl
                action_info["data_downlinked_mb"] = actual_dl
                action_info["data_priority_urgent"] = getattr(self, "_data_priority", 0) == 1

        # --- Reward computation ---
        obs_hours = self.total_observation_s / 3600.0
        is_final = (self.current_step + 1) >= self.max_steps
        # Use OBC data for storage resource penalty (OBC is the downlink bottleneck)
        total_data = self.jetson_raw_mb + self.jetson_compressed_mb + self.obc_data_mb

        reward = self.reward_fn.compute(
            mode=mode,
            battery_soc=self.battery_soc,
            data_stored_mb=total_data,
            storage_capacity_mb=self.storage_capacity_mb,
            action_info=action_info,
            obs_hours=obs_hours,
            downlinked_mb=self.data_downlinked_mb,
            obs_target_hours=self.obs_target_hours,
            downlink_target_mb=self.dl_target_mb,
            episode_step=self.current_step,
            max_steps=self.max_steps,
            is_final_step=is_final,
        )
        return reward, action_info

    def _maybe_inject_anomaly(self):
        if self.active_anomaly is not None:
            self.forced_safe_steps -= 1
            # Recovery: minimum countdown must expire.
            # Conventional ops also require a ground pass (flight controller
            # sends the resume command). Autonomous ops clear via onboard FDIR.
            countdown_done = self.forced_safe_steps <= 0
            can_recover = countdown_done and (
                not self.anomaly_requires_ground_pass
                or self._is_ground_pass_active()
            )
            if can_recover:
                recovery_method = (
                    "ground contact" if self.anomaly_requires_ground_pass
                    else "onboard FDIR"
                )
                logger.info(
                    "Anomaly '%s' cleared via %s at step %d",
                    self.active_anomaly, recovery_method, self.current_step,
                )
                self.active_anomaly = None
            return None
        if self._anomaly_rng.random() < self.anomaly_prob:
            self.active_anomaly = "thermal_warning"
            self.forced_safe_steps = self._anomaly_rng.randint(3, 10)
            logger.info(
                "Anomaly injected: %s (min %d steps) at step %d",
                self.active_anomaly, self.forced_safe_steps, self.current_step,
            )
            return self.active_anomaly
        return None

    def _compute_time_to_next_event(self, current_step: int, intervals) -> int:
        """Return steps until the next interval starts (BSK-RL OpportunityProperties pattern).

        Searches for the nearest interval whose start_step > current_step.
        Returns orbital_period_steps if no future event is found.
        """
        min_steps = self.orbital_period_steps
        for interval in intervals:
            if interval.start_step > current_step:
                gap = interval.start_step - current_step
                if gap < min_steps:
                    min_steps = gap
        return min_steps

    def _compute_orbital_lookahead(self) -> dict:
        """Compute orbital lookahead features for the RL observation space (Group 2).

        Based on BSK-RL Eclipse/OpportunityProperties patterns and EUCASS 2025
        orbital phase encoding. All values are in steps (unnormalized); the
        Gymnasium wrapper normalizes them to [0, 1].

        Returns dict with keys: orbital_phase, time_to_next_eclipse,
        time_to_next_pass, remaining_pass_duration, following_gap_steps.
        """
        step = self.current_step
        orbital_period_steps = self.orbital_period_steps

        # Orbital phase ∈ [0, 1): position within current orbit
        orbital_phase = (step % orbital_period_steps) / orbital_period_steps

        following_gap_steps = orbital_period_steps
        if self._orbital_ctx is not None:
            # Eclipse lookahead
            # If currently in eclipse, next eclipse = next eclipse entry after current one
            time_to_next_eclipse = self._compute_time_to_next_event(step, self._orbital_ctx.eclipses)

            # Ground pass lookahead
            future_passes = sorted(
                (p for p in self._orbital_ctx.ground_passes if p.start_step > step),
                key=lambda p: p.start_step,
            )
            current_pass = self._orbital_ctx.get_current_pass(step)
            if current_pass is not None:
                remaining_pass_duration = max(0, current_pass.end_step - step)
                time_to_next_pass = (
                    future_passes[0].start_step - step
                    if future_passes else orbital_period_steps
                )
            else:
                remaining_pass_duration = 0
                time_to_next_pass = self._compute_time_to_next_event(step, self._orbital_ctx.ground_passes)
            # Gap AFTER the next contact (next-pass end → subsequent-pass start):
            # the window a schedule uploaded at the next pass actually covers.
            # Pass prediction is deterministic ground-segment capability, so
            # ConventionalGround's one-pass-delayed planner may know it.
            if len(future_passes) >= 2:
                following_gap_steps = max(
                    1, future_passes[1].start_step - future_passes[0].end_step
                )
        else:
            # _orbital_ctx is None only before the first reset(); return safe defaults
            time_to_next_eclipse = orbital_period_steps
            time_to_next_pass = orbital_period_steps
            remaining_pass_duration = 0

        return {
            "orbital_phase": orbital_phase,
            "time_to_next_eclipse": time_to_next_eclipse,
            "time_to_next_pass": time_to_next_pass,
            "remaining_pass_duration": remaining_pass_duration,
            "following_gap_steps": following_gap_steps,
        }

    def _generate_tasks(self, in_sun, pass_active):
        tasks = []
        if self.battery_soc < 0.4:
            tasks.append({"type": "manage_power", "priority": "high", "detail": "low_battery"})
        # P3: downlink task only when OBC has data
        if pass_active and self.obc_data_mb > 0:
            tasks.append({"type": "schedule_downlink", "priority": "high", "detail": "pass_active"})
        if self.data_stored_mb > self.storage_capacity_mb * 0.7:
            tasks.append({"type": "schedule_downlink", "priority": "medium", "detail": "storage_pressure"})
        if self.uncompressed_observations >= 3:
            tasks.append({"type": "calibrate_payload", "priority": "medium", "detail": "compress_backlog"})
        if self.battery_soc > 0.6 and self.data_stored_mb < self.storage_capacity_mb * 0.8:
            tasks.append({"type": "schedule_observation", "priority": "normal", "detail": "ready"})
        if self.active_anomaly:
            tasks.append({"type": "respond_to_anomaly", "priority": "critical", "detail": self.active_anomaly})
        return tasks

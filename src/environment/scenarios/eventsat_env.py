"""
EventSat Environment -- Single-satellite operations simulation.
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

logger = logging.getLogger(__name__)
VALID_MODES = {"charging", "communication", "payload_observe", "payload_compress", "safe"}


class EventSatEnvironment(SatelliteEnvironment):
    """Single-satellite EventSat environment."""

    def __init__(self, config):
        self.config = config
        self.scenario = {}
        self._load_scenario(config)
        self.step_duration_s = config.get("step_duration_s", 60.0)
        self.max_steps = config.get("max_steps", 10080)
        self.current_step = 0
        orb = self.scenario.get("orbit", {})
        self.orbital_period_s = orb.get("orbital_period_s", 5676)
        self.eclipse_fraction = orb.get("eclipse_fraction", 0.36)
        self.orbital_period_steps = max(1, int(self.orbital_period_s / self.step_duration_s))
        pwr = self.scenario.get("power", {})
        self.solar_generation_w = pwr.get("solar_panels", {}).get("generation_sun_w", 24.0)
        bat = pwr.get("battery", {})
        self.battery_capacity_wh = bat.get("capacity_wh", 40.0)
        self.initial_soc = bat.get("initial_soc", 0.8)
        self.min_soc = bat.get("min_soc", 0.2)
        self.max_soc = bat.get("max_soc", 1.0)
        self.charge_efficiency = bat.get("charge_efficiency", 0.9)
        self.consumption = pwr.get("consumption", {})
        stor = self.scenario.get("storage", {})
        self.storage_capacity_mb = stor.get("obc_capacity_mb", 512.0)
        self.observation_size_mb = stor.get("observation_size_mb", 2.0)
        comm = self.scenario.get("communications", {})
        self.downlink_rate_kbps = comm.get("sband", {}).get("downlink_rate_kbps", 128)
        modes_cfg = self.scenario.get("modes", {}).get("constraints", {})
        self.observe_min_soc = modes_cfg.get("payload_observe", {}).get("min_battery_soc", 0.4)
        self.compress_min_soc = modes_cfg.get("payload_compress", {}).get("min_battery_soc", 0.3)
        self.anomaly_prob = config.get("anomaly_prob", 0.001)
        self.battery_soc = self.initial_soc
        self.data_stored_mb = 0.0
        self.data_downlinked_mb = 0.0
        self.uncompressed_observations = 0
        self.total_observation_s = 0.0
        self.current_mode = "charging"
        self._orbital_ctx: Optional[OrbitalContext] = None
        self.active_anomaly = None
        self.forced_safe_steps = 0
        self.episode_reward = 0.0
        self._step_metrics = {}

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
        self.data_stored_mb = 0.0
        self.data_downlinked_mb = 0.0
        self.uncompressed_observations = 0
        self.total_observation_s = 0.0
        self.current_mode = "charging"
        self.active_anomaly = None
        self.forced_safe_steps = 0
        self.episode_reward = 0.0
        self._step_metrics = {}
        self._orbital_ctx = compute_orbital_context(
            orbit_config=self.scenario.get("orbit", {}),
            comms_config=self.scenario.get("communications", {}),
            step_s=self.step_duration_s,
            total_steps=self.max_steps,
        )
        return self.get_observation()

    def step(self, actions):
        sat_action = actions.get("eventsat_0", {})
        requested_mode = sat_action.get("mode", "charging") if isinstance(sat_action, dict) else "charging"
        resolved_mode = self._resolve_mode(requested_mode)
        forced = resolved_mode != requested_mode
        in_sun = self._is_in_sunlight()
        pass_active = self._is_ground_pass_active()
        self._update_battery(resolved_mode, in_sun)
        reward = self._apply_mode_effects(resolved_mode, in_sun, pass_active)
        anomaly_event = self._maybe_inject_anomaly()
        self.current_mode = resolved_mode
        self.current_step += 1
        self.episode_reward += reward
        self._step_metrics = {
            "reward": reward,
            "battery_soc": self.battery_soc,
            "data_stored_mb": self.data_stored_mb,
            "data_downlinked_mb": self.data_downlinked_mb,
            "in_sunlight": float(in_sun),
            "ground_pass_active": float(pass_active),
            "forced_mode": float(forced),
            "anomaly": float(anomaly_event is not None),
            "observation_hours": self.total_observation_s / 3600.0,
        }
        return StepResult(
            observation=self.get_observation(),
            rewards={"total": reward},
            done=self.is_done(),
            info={
                "resolved_mode": resolved_mode,
                "requested_mode": requested_mode,
                "forced": forced,
                "anomaly": anomaly_event,
                **self._step_metrics,
            },
        )

    def get_observation(self):
        in_sun = self._is_in_sunlight()
        pass_active = self._is_ground_pass_active()
        sat = SatelliteState(
            satellite_id="eventsat_0",
            position=[0.0, 0.0, 500.0],
            velocity=[0.0, 0.0, 0.0],
            resources={
                "battery_soc": self.battery_soc,
                "data_stored_mb": self.data_stored_mb,
                "data_downlinked_mb": self.data_downlinked_mb,
            },
            status=self.current_mode,
            metadata={
                "in_sunlight": in_sun,
                "ground_pass_active": pass_active,
                "uncompressed_observations": self.uncompressed_observations,
                "total_observation_s": self.total_observation_s,
                "storage_capacity_mb": self.storage_capacity_mb,
                "health_status": "nominal" if self.active_anomaly is None else self.active_anomaly,
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

    def is_done(self):
        return self.current_step >= self.max_steps

    def _is_in_sunlight(self):
        if self._orbital_ctx is not None:
            return self._orbital_ctx.is_in_sunlight(self.current_step)
        # Legacy fallback (should not happen after reset)
        phase = (self.current_step % self.orbital_period_steps) / self.orbital_period_steps
        return phase >= self.eclipse_fraction

    def _is_ground_pass_active(self):
        if self._orbital_ctx is not None:
            return self._orbital_ctx.is_ground_pass_active(self.current_step)
        return False

    def _resolve_mode(self, requested):
        if requested not in VALID_MODES:
            requested = "charging"
        if self.battery_soc <= self.min_soc and requested != "safe":
            return "safe"
        if requested == "payload_observe" and self.battery_soc < self.observe_min_soc:
            return "charging"
        if requested == "payload_compress" and self.battery_soc < self.compress_min_soc:
            return "charging"
        if requested == "communication" and not self._is_ground_pass_active():
            return "charging"
        return requested

    def _update_battery(self, mode, in_sun):
        phase = "sun_w" if in_sun else "eclipse_w"
        consumption_w = self.consumption.get(mode, {}).get(phase, 5.0)
        generation_w = self.solar_generation_w if in_sun else 0.0
        net_power_w = generation_w - consumption_w
        energy_delta_wh = net_power_w * (self.step_duration_s / 3600.0)
        if energy_delta_wh > 0:
            energy_delta_wh *= self.charge_efficiency
        soc_delta = energy_delta_wh / self.battery_capacity_wh
        self.battery_soc = max(0.0, min(self.max_soc, self.battery_soc + soc_delta))

    def _apply_mode_effects(self, mode, in_sun, pass_active):
        reward = 0.0
        if mode == "payload_observe":
            self.total_observation_s += self.step_duration_s
            self.uncompressed_observations += 1
            self.data_stored_mb += self.observation_size_mb
            reward += 1.0
            if self.data_stored_mb > self.storage_capacity_mb:
                self.data_stored_mb = self.storage_capacity_mb
                reward -= 0.5
        elif mode == "payload_compress":
            if self.uncompressed_observations > 0:
                self.uncompressed_observations -= 1
                reward += 0.5
            else:
                reward -= 0.1
        elif mode == "communication":
            if pass_active:
                dl_mb = (self.downlink_rate_kbps / 8.0) * (self.step_duration_s / 1000.0)
                actual_dl = min(dl_mb, self.data_stored_mb)
                self.data_stored_mb -= actual_dl
                self.data_downlinked_mb += actual_dl
                reward += actual_dl
            else:
                reward -= 0.2
        elif mode == "charging":
            reward += 0.1 if in_sun else 0.05
        elif mode == "safe":
            reward -= 0.3
        if self.battery_soc < 0.3:
            reward -= 0.2 * (0.3 - self.battery_soc)
        obs_hours = self.total_observation_s / 3600.0
        target = self.scenario.get("objectives", {}).get("total_observation_hours", 2.0)
        if obs_hours >= target:
            reward += 0.1
        return reward

    def _maybe_inject_anomaly(self):
        if self.active_anomaly is not None:
            self.forced_safe_steps -= 1
            if self.forced_safe_steps <= 0:
                self.active_anomaly = None
            return None
        if random.random() < self.anomaly_prob:
            self.active_anomaly = "thermal_warning"
            self.forced_safe_steps = random.randint(3, 10)
            return self.active_anomaly
        return None

    def _generate_tasks(self, in_sun, pass_active):
        tasks = []
        if self.battery_soc < 0.4:
            tasks.append({"type": "manage_power", "priority": "high", "detail": "low_battery"})
        if pass_active and self.data_stored_mb > 0:
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

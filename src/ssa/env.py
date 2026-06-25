"""SSA constellation environment."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml

from src.core.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteState,
    StepResult,
)
from src.eventsat.multieventsat_env import MultiEventsatEnv
from src.orbital.isl import ISLConfig, is_isl_feasible
from src.ssa.rewards import SSARewardFunction
from src.ssa.targets import (
    DetectionAccess,
    RSOTarget,
    detect_targets_in_fov,
    generate_sso_catalog,
    propagated_catalog_positions_km,
)


SSA_MODES = [
    "charging",
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "communication",
    "isl_share",
    "safe",
]


class SSAEnvironment(MultiEventsatEnv):
    """Multi-EventSat plus RSO detection, ISL sharing, and ground archive."""

    def __init__(self, config: Dict[str, Any]) -> None:
        scenario_defaults = self._load_ssa_scenario_defaults(config)
        self.ssa_config = {**dict(scenario_defaults.get("ssa", {})), **dict(config.get("ssa", {}))}
        self.targets_config = {
            **dict(scenario_defaults.get("targets", {})),
            **dict(config.get("targets", {})),
        }
        self.isl_config_block = {
            **dict(scenario_defaults.get("isl", {})),
            **dict(config.get("isl", {})),
        }
        self.ground_config = {
            **dict(scenario_defaults.get("ground_station", {})),
            **dict(config.get("ground_station", {})),
        }
        satellite_positions = {
            **dict(scenario_defaults.get("satellite_positions_km", {})),
            **dict(config.get("satellite_positions_km", {})),
        }
        self._configured_sat_positions = {
            str(k): tuple(float(x) for x in v)
            for k, v in satellite_positions.items()
        }
        super().__init__(config)

        self.reward_fn = SSARewardFunction(config.get("reward_config", {}))
        self.mode_list = list(SSA_MODES)
        self.mode_to_index = {mode: idx for idx, mode in enumerate(self.mode_list)}
        self.target_count = int(self.targets_config.get("count", 6))
        self.fov_half_angle_deg = float(self.targets_config.get("fov_half_angle_deg", 5.0))
        self.max_detection_range_km = self.targets_config.get("max_range_km")
        if self.max_detection_range_km is not None:
            self.max_detection_range_km = float(self.max_detection_range_km)
        self.catalog_seed = self.targets_config.get("seed")
        self.prefer_orekit_targets = bool(self.targets_config.get("prefer_orekit", False))
        self.fixed_target_positions = self._parse_fixed_target_positions(
            self.targets_config.get("fixed_positions_km")
        )
        if self.fixed_target_positions:
            self.target_count = len(self.fixed_target_positions)
        self.target_ids = [f"rso_{idx}" for idx in range(self.target_count)]
        self.target_index = {target_id: idx for idx, target_id in enumerate(self.target_ids)}
        self.targets: list[RSOTarget] = []

        self.isl_config = ISLConfig(**{
            key: value for key, value in self.isl_config_block.items()
            if key in ISLConfig.__dataclass_fields__
        })
        self.ground_always_visible = bool(self.ground_config.get("always_visible", False))

        self.detection_matrix: list[list[int]] = []
        self.onboard_estimates: dict[str, dict[str, dict[str, Any]]] = {}
        self.ground_archive: dict[str, list[dict[str, Any]]] = {}
        self._undelivered_records: dict[str, list[dict[str, Any]]] = {}
        self._last_observed_step: dict[str, int] = {}
        self.duplicate_observations = 0
        self.successful_observations = 0
        self.total_observation_records = 0
        self.isl_attempts = 0
        self.isl_successes = 0
        self.last_step_detections: dict[str, list[str]] = {}
        self.last_step_downlinked_records = 0


    @staticmethod
    def _load_ssa_scenario_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
        scenario_path = config.get("scenario_config") or config.get("scenario_file")
        if not scenario_path:
            return {}
        path = Path(str(scenario_path))
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            scenario = yaml.safe_load(handle) or {}
        return scenario if isinstance(scenario, dict) else {}

    def reset(self, seed: int | None = None) -> EnvironmentObservation:
        obs = super().reset(seed=seed)
        catalog_seed = self.catalog_seed if self.catalog_seed is not None else seed
        self.targets = generate_sso_catalog(
            self.target_count,
            seed=None if catalog_seed is None else int(catalog_seed),
            altitude_range_km=tuple(self.targets_config.get("altitude_range_km", (600.0, 900.0))),
            eccentricity_max=float(self.targets_config.get("eccentricity_max", 0.001)),
            inclination_range_deg=tuple(self.targets_config.get("inclination_range_deg", (97.0, 99.0))),
            object_size_m=float(self.targets_config.get("object_size_m", 1.0)),
        )
        if self.fixed_target_positions:
            self.target_ids = list(self.fixed_target_positions.keys())
            self.target_index = {target_id: idx for idx, target_id in enumerate(self.target_ids)}
        else:
            self.target_ids = [target.object_id for target in self.targets]
            self.target_index = {target_id: idx for idx, target_id in enumerate(self.target_ids)}

        self.detection_matrix = [
            [0 for _ in range(self.target_count)] for _ in range(self.constellation_size)
        ]
        self.onboard_estimates = {sat_id: {} for sat_id in self._sat_ids}
        self.ground_archive = {target_id: [] for target_id in self.target_ids}
        self._undelivered_records = {sat_id: [] for sat_id in self._sat_ids}
        self._last_observed_step = {}
        self.duplicate_observations = 0
        self.successful_observations = 0
        self.total_observation_records = 0
        self.isl_attempts = 0
        self.isl_successes = 0
        self.last_step_detections = {sat_id: [] for sat_id in self._sat_ids}
        self.last_step_downlinked_records = 0
        return self.get_observation()

    def step(self, actions: Dict[str, Any]) -> StepResult:
        decoded = self._decode_actions(actions)
        event_actions = {
            sat_id: {"mode": ("charging" if mode == "isl_share" else mode)}
            for sat_id, mode in decoded.items()
        }
        base_result = super().step(event_actions)
        per_sat_info = dict(base_result.info.get("per_satellite", {}))

        self.last_step_detections = {sat_id: [] for sat_id in self._sat_ids}
        self.last_step_downlinked_records = 0
        target_positions = self._target_positions()

        for sat_id, requested_mode in decoded.items():
            info = per_sat_info.get(sat_id, {})
            productive_observe = (
                requested_mode == "payload_observe"
                and info.get("resolved_mode") == "payload_observe"
                and not bool(info.get("in_transition", False))
            )
            if productive_observe:
                accesses = detect_targets_in_fov(
                    self._satellite_position(sat_id, base_result.observation),
                    target_positions,
                    fov_half_angle_deg=self.fov_half_angle_deg,
                    max_range_km=self.max_detection_range_km,
                )
                for access in accesses:
                    self._record_detection(sat_id, access)

        self._apply_isl_shares(decoded, base_result.observation)
        self._apply_ground_downlinks(decoded, per_sat_info)

        info = dict(base_result.info)
        info.update(self._ssa_info(decoded))
        rewards = self.reward_fn.compute_rewards(
            base_result.rewards,
            {"_global": {"delivered_coverage": self.delivered_coverage}},
        )
        return StepResult(
            observation=self.get_observation(),
            rewards=rewards,
            done=base_result.done,
            truncated=base_result.truncated,
            info=info,
        )

    def get_observation(self) -> EnvironmentObservation:
        obs = super().get_observation()
        target_positions = self._target_positions() if self.target_ids else {}
        satellites: dict[str, SatelliteState] = {}
        for sat_id, sat in obs.constellation_state.satellites.items():
            position = self._satellite_position(sat_id, obs)
            visible = detect_targets_in_fov(
                position,
                target_positions,
                fov_half_angle_deg=self.fov_half_angle_deg,
                max_range_km=self.max_detection_range_km,
            ) if target_positions else []
            metadata = dict(sat.metadata)
            sat_idx = self._sat_index(sat_id)
            metadata.update({
                "ssa_detection_matrix": deepcopy(self.detection_matrix),
                "ssa_detection_row": list(self.detection_matrix[sat_idx]) if self.detection_matrix else [],
                "ssa_known_objects": sorted(self.onboard_estimates.get(sat_id, {})),
                "ssa_delivered_objects": sorted(self.delivered_object_ids),
                "visible_rso_ids": [item.object_id for item in visible],
                "visible_rso_count": len(visible),
                "ssa_onboard_coverage": self.onboard_coverage,
                "ssa_delivered_coverage": self.delivered_coverage,
            })
            satellites[sat_id] = SatelliteState(
                satellite_id=sat_id,
                position=list(position),
                velocity=list(sat.velocity),
                resources=dict(sat.resources),
                status=sat.status,
                metadata=metadata,
            )

        tasks = list(obs.tasks)
        for sat_id, sat in satellites.items():
            for access in detect_targets_in_fov(
                sat.position,
                target_positions,
                fov_half_angle_deg=self.fov_half_angle_deg,
                max_range_km=self.max_detection_range_km,
            ) if target_positions else []:
                tasks.append({
                    "type": "observe_rso",
                    "satellite_id": sat_id,
                    "object_id": access.object_id,
                    "quality": access.quality,
                })

        return EnvironmentObservation(
            constellation_state=ConstellationState(
                timestep=obs.constellation_state.timestep,
                epoch_seconds=obs.constellation_state.epoch_seconds,
                satellites=satellites,
                global_info={
                    **dict(obs.constellation_state.global_info),
                    "ssa_detection_matrix": deepcopy(self.detection_matrix),
                    "ssa_target_count": self.target_count,
                    "ssa_onboard_coverage": self.onboard_coverage,
                    "ssa_delivered_coverage": self.delivered_coverage,
                    "ssa_ground_archive_records": sum(len(v) for v in self.ground_archive.values()),
                },
            ),
            tasks=tasks,
            events=obs.events,
        )

    @property
    def delivered_object_ids(self) -> set[str]:
        return {object_id for object_id, records in self.ground_archive.items() if records}

    @property
    def onboard_coverage(self) -> float:
        if self.target_count <= 0:
            return 0.0
        known = {
            target_id
            for sat_estimates in self.onboard_estimates.values()
            for target_id in sat_estimates
        }
        return len(known) / self.target_count

    @property
    def delivered_coverage(self) -> float:
        return len(self.delivered_object_ids) / self.target_count if self.target_count else 0.0

    def get_metrics(self) -> Dict[str, float]:
        return {
            **super().get_metrics(),
            **self._ssa_metrics(),
        }

    def _decode_actions(self, actions: Dict[str, Any]) -> dict[str, str]:
        decoded: dict[str, str] = {}
        for sat_id in self._sat_ids:
            action = actions.get(sat_id, {}) if isinstance(actions, dict) else {}
            mode = "charging"
            if isinstance(action, dict):
                raw_mode = action.get("mode")
                if raw_mode in self.mode_to_index:
                    mode = str(raw_mode)
                elif "mode_one_hot" in action:
                    mode = self._decode_one_hot(action.get("mode_one_hot"))
                elif "one_hot" in action:
                    mode = self._decode_one_hot(action.get("one_hot"))
            elif isinstance(action, (list, tuple)):
                mode = self._decode_one_hot(action)
            decoded[sat_id] = mode
        return decoded

    def _decode_one_hot(self, values: Any) -> str:
        try:
            seq = [int(v) for v in values]
        except TypeError:
            return "charging"
        if len(seq) != len(self.mode_list):
            return "charging"
        if sum(1 for value in seq if value == 1) != 1:
            return "charging"
        return self.mode_list[seq.index(1)]

    def _record_detection(self, sat_id: str, access: DetectionAccess) -> None:
        target_idx = self.target_index.get(access.object_id)
        if target_idx is None:
            return
        sat_idx = self._sat_index(sat_id)
        already_known = any(row[target_idx] for row in self.detection_matrix)
        if already_known:
            self.duplicate_observations += 1
        else:
            self.successful_observations += 1
        self.total_observation_records += 1
        self.detection_matrix[sat_idx][target_idx] = 1
        record = {
            "object_id": access.object_id,
            "satellite_id": sat_id,
            "position_km": list(access.position_km),
            "time_step": self.current_step,
            "quality": float(access.quality),
        }
        current = self.onboard_estimates[sat_id].get(access.object_id)
        if current is None or float(record["quality"]) > float(current.get("quality", 0.0)):
            self.onboard_estimates[sat_id][access.object_id] = record
        self._undelivered_records[sat_id].append(record)
        self._last_observed_step[access.object_id] = self.current_step
        self.last_step_detections.setdefault(sat_id, []).append(access.object_id)

    def _apply_isl_shares(self, modes: Mapping[str, str], observation: EnvironmentObservation) -> None:
        for src_id, mode in modes.items():
            if mode != "isl_share":
                continue
            for dst_id in self._sat_ids:
                if dst_id == src_id:
                    continue
                self.isl_attempts += 1
                dst_idle = modes.get(dst_id, "charging") in {"charging", "safe", "isl_share"}
                if not is_isl_feasible(
                    self._satellite_position(src_id, observation),
                    self._satellite_position(dst_id, observation),
                    endpoint_a_idle=True,
                    endpoint_b_idle=dst_idle,
                    config=self.isl_config,
                ):
                    continue
                self._merge_satellite_knowledge(src_id, dst_id)
                self.isl_successes += 1

    def _merge_satellite_knowledge(self, src_id: str, dst_id: str) -> None:
        src_idx = self._sat_index(src_id)
        dst_idx = self._sat_index(dst_id)
        for target_id, src_record in self.onboard_estimates.get(src_id, {}).items():
            target_idx = self.target_index[target_id]
            self.detection_matrix[dst_idx][target_idx] = max(
                self.detection_matrix[dst_idx][target_idx],
                self.detection_matrix[src_idx][target_idx],
            )
            dst_record = self.onboard_estimates[dst_id].get(target_id)
            if dst_record is None or float(src_record.get("quality", 0.0)) > float(dst_record.get("quality", 0.0)):
                self.onboard_estimates[dst_id][target_id] = deepcopy(src_record)

    def _apply_ground_downlinks(self, modes: Mapping[str, str], per_sat_info: Mapping[str, Any]) -> None:
        for sat_id, mode in modes.items():
            if mode != "communication":
                continue
            pass_active = bool(per_sat_info.get(sat_id, {}).get("ground_pass_active", False))
            if not (pass_active or self.ground_always_visible):
                continue
            records = list(self._undelivered_records.get(sat_id, []))
            for record in records:
                self.ground_archive.setdefault(record["object_id"], []).append(deepcopy(record))
            self.last_step_downlinked_records += len(records)
            self._undelivered_records[sat_id] = []

    def _ssa_info(self, modes: Mapping[str, str]) -> dict[str, Any]:
        metrics = self._ssa_metrics()
        return {
            **metrics,
            "ssa_requested_modes": dict(modes),
            "ssa_detection_matrix": deepcopy(self.detection_matrix),
            "ssa_last_step_detections": deepcopy(self.last_step_detections),
            "ssa_step_downlinked_records": float(self.last_step_downlinked_records),
        }

    def _ssa_metrics(self) -> dict[str, float]:
        duplicate_rate = (
            self.duplicate_observations / self.total_observation_records
            if self.total_observation_records else 0.0
        )
        revisit_ages = [
            self.current_step - step for step in self._last_observed_step.values()
        ]
        mean_revisit = sum(revisit_ages) / len(revisit_ages) if revisit_ages else 0.0
        connectivity = self.isl_successes / self.isl_attempts if self.isl_attempts else 0.0
        known_objects = {
            target_id
            for sat_estimates in self.onboard_estimates.values()
            for target_id in sat_estimates
        }
        return {
            "ssa_onboard_coverage": self.onboard_coverage,
            "ssa_delivered_coverage": self.delivered_coverage,
            "ssa_known_objects": float(len(known_objects)),
            "ssa_delivered_objects": float(len(self.delivered_object_ids)),
            "successful_observations": float(self.successful_observations),
            "duplicate_observations": float(self.duplicate_observations),
            "duplicate_observation_rate": duplicate_rate,
            "mean_revisit_steps": mean_revisit,
            "isl_connectivity": connectivity,
        }

    def _target_positions(self) -> dict[str, tuple[float, float, float]]:
        if self.fixed_target_positions:
            return dict(self.fixed_target_positions)
        return propagated_catalog_positions_km(
            self.targets,
            self.current_step * self.step_duration_s,
            prefer_orekit=self.prefer_orekit_targets,
        )

    def _parse_fixed_target_positions(self, raw: Any) -> dict[str, tuple[float, float, float]]:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return {str(k): tuple(float(x) for x in v) for k, v in raw.items()}
        out = {}
        for idx, value in enumerate(raw):
            out[f"rso_{idx}"] = tuple(float(x) for x in value)
        return out

    def _satellite_position(
        self, sat_id: str, observation: EnvironmentObservation | None = None
    ) -> tuple[float, float, float]:
        if sat_id in self._configured_sat_positions:
            return self._configured_sat_positions[sat_id]
        if observation is not None and sat_id in observation.constellation_state.satellites:
            return tuple(float(x) for x in observation.constellation_state.satellites[sat_id].position)
        return (0.0, 0.0, 500.0)

    def _sat_index(self, sat_id: str) -> int:
        try:
            return self._sat_ids.index(sat_id)
        except ValueError as exc:
            raise KeyError(f"unknown satellite id {sat_id!r}") from exc

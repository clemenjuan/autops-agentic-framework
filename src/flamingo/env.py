"""
Flamingo-lite multi-satellite SSA environment.

This is the smallest runnable constellation scenario for opening the
Organisation axis. It intentionally models only the coordination bottleneck:
multiple satellites see time-varying RSO opportunities and can waste capacity by
duplicating observations or requesting infeasible targets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.core.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteEnvironment,
    SatelliteState,
    StepResult,
)


@dataclass(frozen=True)
class RSOTarget:
    """Resident-space-object target used by the Flamingo-lite scheduler."""

    target_id: str
    priority: float
    phase_offset: int


class FlamingoEnvironment(SatelliteEnvironment):
    """Small deterministic SSA scheduling environment.

    Actions are keyed by satellite id and may contain ``target_id``. A successful
    observation requires the requested target to be visible to that satellite at
    the current step and not already observed by another satellite in the same
    step.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.scenario = self._load_scenario(config)
        step_seconds = config.get("step_duration_s", config.get("timestep_seconds", 60))
        merged = {
            "constellation_size": config.get(
                "constellation_size",
                self.scenario.get("constellation_size", 3),
            ),
            "timestep_seconds": step_seconds,
            "max_steps": config.get("max_steps", self.scenario.get("max_steps", 1440)),
        }
        super().__init__(merged)

        self.visibility_period_steps = int(
            self.scenario.get("visibility_period_steps", 12)
        )
        self.visibility_window_steps = int(
            self.scenario.get("visibility_window_steps", 3)
        )
        # How strongly each satellite's visibility window is phase-shifted from
        # its neighbours. 1 (default) de-synchronises the constellation so every
        # satellite sees a different target each step — there is no contention
        # and any sensible planner deconflicts trivially. 0 makes every
        # satellite see the *same* windows, so several satellites compete for the
        # same high-priority RSO: this is the coordination bottleneck that
        # separates organisation topologies (an uncoordinated org wastes capacity
        # on duplicates, a coordinated one spreads across targets).
        self.satellite_phase_shift = int(
            self.scenario.get("satellite_phase_shift", 1)
        )
        self.observation_data_mb = float(self.scenario.get("observation_data_mb", 1.0))
        self.ground_pass_active = bool(self.scenario.get("ground_pass_active", True))

        # Per-episode RSO catalog. Deterministic by default (reproducible smoke
        # tests); when ``stochastic`` is set, each episode draws a fresh catalog
        # from a seeded RNG so repeated episodes form a real distribution and
        # organisation comparisons get error bars. Paired seeds (every org runs
        # the same config.seed) hand each org the *same* catalog per episode, so
        # the organisation comparison stays a fair within-instance contrast.
        catalog = self.scenario.get("targets", {})
        self.target_count = int(catalog.get("count", 6))
        self.target_priorities = [
            float(p) for p in catalog.get("priorities", [3.0, 2.0, 1.0])
        ]
        self.stochastic = bool(self.scenario.get("stochastic", False))
        self._rng = random.Random(0)
        self.targets = self._build_targets(None)

        self.satellite_ids = [
            f"flamingo_{idx}" for idx in range(self.constellation_size)
        ]

        self.successful_observations = 0
        self.duplicate_observations = 0
        self.constraint_violations = 0
        self.total_attempts = 0
        self.total_utility = 0.0
        self._last_observed_step: Dict[str, Optional[int]] = {
            target.target_id: None for target in self.targets
        }

    def reset(self, seed: Optional[int] = None) -> EnvironmentObservation:
        self.current_step = 0
        # Seed the per-episode RNG (the runner passes config.seed + episode_id,
        # so seeds are paired across organisations). Draw a fresh catalog only in
        # stochastic mode; the deterministic default keeps its fixed catalog.
        self._rng = random.Random(0 if seed is None else seed)
        if self.stochastic:
            self.targets = self._build_targets(self._rng)
        self.successful_observations = 0
        self.duplicate_observations = 0
        self.constraint_violations = 0
        self.total_attempts = 0
        self.total_utility = 0.0
        self._last_observed_step = {
            target.target_id: None for target in self.targets
        }
        return self.get_observation()

    def step(self, actions: Dict[str, Any]) -> StepResult:
        observed_this_step: set[str] = set()
        step_duplicates = 0
        step_constraint_violations = 0
        step_utility = 0.0

        for sat_id in self.satellite_ids:
            action = actions.get(sat_id, {}) or {}
            target_id = action.get("target_id")
            if target_id in (None, "", "idle"):
                continue

            self.total_attempts += 1
            target = self._target_by_id(target_id)
            if target is None or not self._is_visible(sat_id, target_id):
                self.constraint_violations += 1
                step_constraint_violations += 1
                continue

            if target_id in observed_this_step:
                self.duplicate_observations += 1
                step_duplicates += 1
                continue

            observed_this_step.add(target_id)
            self.successful_observations += 1
            self._last_observed_step[target_id] = self.current_step
            step_utility += target.priority

        self.total_utility += step_utility
        self.current_step += 1
        observation = self.get_observation()
        return StepResult(
            observation=observation,
            rewards={"utility": step_utility},
            done=False,
            truncated=self.is_done(),
            info={
                "observed_targets": sorted(observed_this_step),
                "step_successful_observations": float(len(observed_this_step)),
                "step_duplicate_observations": float(step_duplicates),
                "step_constraint_violations": float(step_constraint_violations),
                "duplicates": float(self.duplicate_observations),
                "constraint_violations": float(self.constraint_violations),
                "metrics": self.get_metrics(),
            },
        )

    def get_observation(self) -> EnvironmentObservation:
        satellites = {
            sat_id: self._satellite_state(sat_id)
            for sat_id in self.satellite_ids
        }
        return EnvironmentObservation(
            constellation_state=ConstellationState(
                timestep=self.current_step,
                epoch_seconds=self.current_step * self.timestep_seconds,
                satellites=satellites,
                global_info={
                    "num_targets": len(self.targets),
                    "successful_observations": self.successful_observations,
                },
            ),
            tasks=self._visible_tasks(),
            events=[],
        )

    def get_metrics(self) -> Dict[str, float]:
        observed_targets = sum(
            1 for step in self._last_observed_step.values() if step is not None
        )
        duplicate_rate = (
            self.duplicate_observations / self.total_attempts
            if self.total_attempts else 0.0
        )
        violation_rate = (
            self.constraint_violations / self.total_attempts
            if self.total_attempts else 0.0
        )
        revisit_ages = [
            self.current_step - step
            for step in self._last_observed_step.values()
            if step is not None
        ]
        mean_revisit_steps = (
            sum(revisit_ages) / len(revisit_ages) if revisit_ages else 0.0
        )
        return {
            "utility": self.total_utility,
            "successful_observations": float(self.successful_observations),
            "coverage_rate": observed_targets / len(self.targets) if self.targets else 0.0,
            "duplicate_observation_rate": duplicate_rate,
            "constraint_violation_rate": violation_rate,
            "mean_revisit_steps": mean_revisit_steps,
        }

    def _satellite_state(self, sat_id: str) -> SatelliteState:
        idx = self.satellite_ids.index(sat_id)
        visible_targets = [
            target.target_id
            for target in self.targets
            if self._is_visible(sat_id, target.target_id)
        ]
        return SatelliteState(
            satellite_id=sat_id,
            position=[float(idx), 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            resources={"data_capacity_mb": self.observation_data_mb},
            status="nominal",
            metadata={
                "visible_targets": visible_targets,
                "ground_pass_active": self.ground_pass_active,
            },
        )

    def _visible_tasks(self) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        for sat_id in self.satellite_ids:
            for target in self.targets:
                if self._is_visible(sat_id, target.target_id):
                    tasks.append(
                        {
                            "satellite_id": sat_id,
                            "target_id": target.target_id,
                            "priority": target.priority,
                        }
                    )
        return tasks

    def _is_visible(self, sat_id: str, target_id: str) -> bool:
        target = self._target_by_id(target_id)
        if target is None:
            return False
        sat_idx = self.satellite_ids.index(sat_id)
        phase = (
            self.current_step
            + sat_idx * self.satellite_phase_shift
            + target.phase_offset
        )
        return (phase % self.visibility_period_steps) < self.visibility_window_steps

    def _target_by_id(self, target_id: str) -> Optional[RSOTarget]:
        for target in self.targets:
            if target.target_id == target_id:
                return target
        return None

    @staticmethod
    def _load_scenario(config: Dict[str, Any]) -> Dict[str, Any]:
        path = config.get("scenario_file") or config.get("scenario_config")
        if path and Path(path).exists():
            with open(path, encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        return dict(config.get("scenario_params", {}))

    def _build_targets(self, rng: Optional[random.Random]) -> List[RSOTarget]:
        """Build the RSO catalog.

        With ``rng`` (stochastic mode) each target draws a random visibility
        phase and a priority sampled from the configured set, so every episode is
        a different SSA instance. Without ``rng`` the catalog is the fixed
        deterministic layout (phase_offset = index, priorities cycled).
        """
        priorities = self.target_priorities or [1.0]
        targets: List[RSOTarget] = []
        for idx in range(self.target_count):
            if rng is not None:
                phase_offset = rng.randrange(self.visibility_period_steps)
                priority = float(rng.choice(priorities))
            else:
                phase_offset = idx
                priority = float(priorities[idx % len(priorities)])
            targets.append(
                RSOTarget(
                    target_id=f"rso_{idx}",
                    priority=priority,
                    phase_offset=phase_offset,
                )
            )
        return targets


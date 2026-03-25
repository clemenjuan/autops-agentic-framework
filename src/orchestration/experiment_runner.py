"""
Experiment Runner — Main Orchestrator.

Configuration-driven experiment execution with comprehensive logging
and reproducibility. Loads a YAML configuration, initialises all
components, executes episodes with metrics collection, and saves
results with full provenance.

Usage::

    runner = ExperimentRunner("configs/experiments/my_experiment.yaml")
    stats = runner.run()
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.orchestration.config_loader import ExperimentConfig, load_config

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates a single experiment from configuration to results.

    Attributes:
        config: Validated experiment configuration.
        output_dir: Path where results and logs are saved.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: ExperimentConfig | None = None,
    ) -> None:
        """Initialise the experiment runner.

        Provide either a path to a YAML config file or a pre-built
        :class:`ExperimentConfig` object.

        Args:
            config_path: Path to the YAML configuration file.
            config: Pre-built configuration object (takes precedence).

        Raises:
            ValueError: If neither ``config_path`` nor ``config`` is supplied.
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            raise ValueError("Provide either config_path or config.")

        self.output_dir = Path(self.config.output_dir)

        # Component placeholders — populated in ``_initialize_components``
        self._environment: Any = None
        self._organization: Any = None
        self._decision_loops: Dict[str, Any] = {}  # agent_id → loop
        self._memory: Any = None
        self._metrics_collector: Any = None
        self._operations_paradigm: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full experiment.

        Returns:
            Dictionary containing experiment statistics and metadata.
        """
        self._setup_logging()
        self._set_seeds(self.config.seed)

        logger.info(
            "Starting experiment '%s' — %d episodes, seed=%d",
            self.config.experiment_id,
            self.config.num_episodes,
            self.config.seed,
        )

        self._initialize_components()

        all_episode_metrics: List[Dict[str, Any]] = []

        for episode in range(self.config.num_episodes):
            logger.info("Episode %d / %d", episode + 1, self.config.num_episodes)
            episode_result = self._run_episode(episode)
            all_episode_metrics.append(episode_result)

            if self.config.save_checkpoints:
                self._save_checkpoint(episode, episode_result)

        results = self._compile_results(all_episode_metrics)
        self._save_results(results)

        logger.info("Experiment '%s' complete.", self.config.experiment_id)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        """Configure logging for the experiment."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.output_dir / "experiment.log"

        # Root logger for the experiment
        exp_logger = logging.getLogger("src")
        exp_logger.setLevel(getattr(logging, self.config.log_level, logging.INFO))

        # File handler
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        fh.setFormatter(formatter)
        exp_logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, self.config.log_level, logging.INFO))
        ch.setFormatter(formatter)
        exp_logger.addHandler(ch)

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """Set random seeds for reproducibility.

        Args:
            seed: Integer seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        # torch seed would go here if using subsymbolic representations
        logger.debug("Random seeds set to %d", seed)

    def _initialize_components(self) -> None:
        """Instantiate all experiment components from configuration.

        This is the integration point where the morphological matrix
        dimensions are wired together. Each component factory will be
        plugged in as implementations are added.

        Raises:
            NotImplementedError: Until concrete component factories exist.
        """
        logger.info(
            "Initialising components — org=%s, loop=%s, repr=%s, emergence=%s, ops=%s",
            self.config.agent_organization,
            self.config.decision_loop,
            self.config.representation,
            self.config.emergence_mode,
            self.config.operations_paradigm,
        )

        # ----------------------------------------------------------
        # Component instantiation stubs
        # Replace with factory calls as implementations are added.
        # ----------------------------------------------------------

        # Environment
        self._environment = self._create_environment()

        # Memory (fixed design)
        self._memory = self._create_memory()

        # Agent organization
        self._organization = self._create_organization()

        # Decision loops (one per agent)
        self._decision_loops = self._create_decision_loops()

        # Operations paradigm (5th dimension)
        self._operations_paradigm = self._create_operations_paradigm()

        # Metrics collector
        self._metrics_collector = self._create_metrics_collector()

        logger.info("All components initialised.")

    def _create_environment(self) -> Any:
        """Factory for the satellite environment."""
        scenario = self.config.environment.scenario
        env_cfg = {
            "step_duration_s": self.config.environment.timestep_seconds,
            "max_steps": self.config.max_steps,
            **self.config.environment.scenario_config,
        }

        if scenario == "eventsat":
            from src.environment.scenarios.eventsat_env import EventSatEnvironment
            # Tell the environment whether the operations paradigm allows
            # onboard anomaly recovery (autonomous) vs requiring ground contact.
            env_cfg["anomaly_requires_ground_pass"] = (
                self.config.operations_paradigm != "autonomous_hybrid"
            )
            return EventSatEnvironment(config=env_cfg)

        logger.warning("Unknown scenario '%s', returning None.", scenario)
        return None

    def _create_memory(self) -> Any:
        """Factory for the fixed memory system.

        Returns:
            An initialised ``FixedMemory`` instance.
        """
        from src.memory.fixed_memory import FixedMemory

        return FixedMemory(config=self.config.memory_config)

    def _create_organization(self) -> Any:
        """Factory for the agent organization.

        Returns:
            An initialised ``AgentOrganization`` subclass.
        """
        from src.agent_organization.centralized import CentralizedOrganization
        from src.agent_organization.distributed import DistributedOrganization
        from src.agent_organization.hierarchical import HierarchicalOrganization

        org_map = {
            "centralized": CentralizedOrganization,
            "hierarchical": HierarchicalOrganization,
            "distributed": DistributedOrganization,
        }

        org_cls = org_map.get(self.config.agent_organization)
        if org_cls is None:
            raise ValueError(
                f"Unknown agent_organization: '{self.config.agent_organization}'"
            )

        org = org_cls(config=self.config.agent_organization_config)
        org.initialize(
            constellation_size=self.config.environment.constellation_size,
        )
        return org

    def _create_decision_loops(self):
        """Factory for decision loop instances (one per agent)."""
        from src.emergence.controller import EmergenceController
        import src.representation.rule_based_eventsat  # register representations
        import src.representation.schedule_based_eventsat  # register schedule planner
        import src.representation.conventional_schedule_eventsat  # register human schedule planner
        emergence = EmergenceController(config=self.config.emergence_config)
        repr_type = self.config.representation_config.get('type', 'rule_based_eventsat')
        representation = emergence.get_representation(
            repr_type=repr_type,
            repr_config=self.config.representation_config,
        )
        # Seed stochastic representations for reproducibility
        if hasattr(representation, "seed"):
            representation.seed(self.config.seed)
        loop_type = self.config.decision_loop
        if loop_type == 'sda':
            from src.decision_loop.sda_loop import SDALoop
            loop_cls = SDALoop
        elif loop_type == 'ooda':
            from src.decision_loop.ooda_loop import OODALoop
            loop_cls = OODALoop
        elif loop_type == 'react':
            from src.decision_loop.react_loop import ReActLoop
            loop_cls = ReActLoop
        else:
            raise ValueError(f"Unknown decision_loop: '{loop_type}'")
        agents = self._organization.get_agents() if self._organization else ['central_agent']
        loops = {}
        for agent_id in agents:
            loops[agent_id] = loop_cls(
                config=self.config.decision_loop_config,
                representation=representation,
            )
        return loops

    def _create_operations_paradigm(self) -> Any:
        """Factory for the operations paradigm (5th morphological matrix dimension)."""
        paradigm_type = self.config.operations_paradigm
        paradigm_config = self.config.operations_paradigm_config

        if paradigm_type == "autonomous_hybrid":
            from src.operations.autonomous_hybrid import AutonomousHybrid
            return AutonomousHybrid(config=paradigm_config)
        elif paradigm_type == "autonomous_ground":
            from src.operations.autonomous_ground import AutonomousGround
            return AutonomousGround(config=paradigm_config)
        elif paradigm_type == "conventional_ground":
            from src.operations.conventional_ground import ConventionalGround
            return ConventionalGround(config=paradigm_config)
        else:
            logger.warning(
                "Unknown operations_paradigm '%s', falling back to autonomous_hybrid.",
                paradigm_type,
            )
            from src.operations.autonomous_hybrid import AutonomousHybrid
            return AutonomousHybrid(config=paradigm_config)

    def _create_metrics_collector(self) -> Any:
        """Factory for the metrics collector."""
        scenario = self.config.environment.scenario
        if scenario == "eventsat":
            from src.orchestration.eventsat_metrics import EventSatMetricsCollector
            metrics_cfg = self.config.metrics.model_dump()
            # Pass environment parameters needed for energy/utility computation
            metrics_cfg["max_steps"] = self.config.max_steps
            metrics_cfg["step_duration_s"] = self.config.environment.timestep_seconds
            if self._environment is not None and hasattr(self._environment, "battery_capacity_wh"):
                metrics_cfg["battery_capacity_wh"] = self._environment.battery_capacity_wh
            return EventSatMetricsCollector(config=metrics_cfg)
        logger.warning("No metrics collector for scenario '%s'.", scenario)
        return None

    def _run_episode(self, episode_id: int) -> Dict[str, Any]:
        """Execute a single episode.

        Args:
            episode_id: Episode index.

        Returns:
            Dictionary of episode-level metrics and info.
        """
        episode_start = time.perf_counter()

        # --- Reset phase ---
        if self._environment is not None:
            observation = self._environment.reset(seed=self.config.seed + episode_id)
        else:
            observation = None

        if self._memory is not None:
            self._memory.reset()

        for loop in self._decision_loops.values():
            loop.reset()

        if self._operations_paradigm is not None:
            self._operations_paradigm.reset()

        # --- Step loop ---
        max_steps = self.config.max_steps
        step_data: List[Dict[str, Any]] = []

        for step in range(max_steps):
            step_info = self._run_step(step, observation)
            step_data.append(step_info)

            # Check termination
            if self._environment is not None and self._environment.is_done():
                break
            # Update observation for next step
            observation = step_info.get("observation")

        episode_duration = time.perf_counter() - episode_start

        # --- Finalise episode metrics ---
        episode_metrics = None
        if self._metrics_collector is not None:
            episode_metrics = self._metrics_collector.finalise_episode(episode_id)

        return {
            "episode_id": episode_id,
            "num_steps": len(step_data),
            "wall_clock_seconds": episode_duration,
            "episode_metrics": episode_metrics,
            "steps": step_data,
        }

    def _run_step(self, step: int, observation: Any) -> Dict[str, Any]:
        """Execute a single simulation step.

        The canonical flow is:
        1. Organization distributes observation to agents.
        2. Each agent's decision loop produces an action.
        3. Organization collects actions.
        4. Environment executes actions and returns results.
        5. Metrics are collected.

        Args:
            step: Current step index.
            observation: Current environment observation.

        Returns:
            Dictionary of step-level data.
        """
        step_start = time.perf_counter()

        # 0. Determine ground pass status (needed by operations paradigm)
        ground_pass_active = False
        if observation is not None:
            for sat in observation.constellation_state.satellites.values():
                if sat.metadata.get("ground_pass_active", False):
                    ground_pass_active = True
                    break

        # 1. Filter observation through operations paradigm
        filtered_observation = observation
        if self._operations_paradigm is not None and observation is not None:
            filtered_observation = self._operations_paradigm.filter_observation(
                observation, step
            )

        # 2. Distribute observations
        if self._organization is not None and filtered_observation is not None:
            agent_obs = self._organization.distribute_observation(filtered_observation)
        else:
            agent_obs = {}

        # 3. Decision loops (timed for latency metric)
        agent_actions = {}
        decision_metrics: Dict[str, Any] = {}
        for agent_id, loop in self._decision_loops.items():
            obs = agent_obs.get(agent_id)
            t0 = time.perf_counter()
            action, self._memory = loop.process(obs, self._memory)
            decision_latency = time.perf_counter() - t0
            from src.agent_organization.base import AgentAction

            agent_actions[agent_id] = AgentAction(agent_id=agent_id, action=action)
            # Collect decision loop metrics (latency, rationale, etc.)
            loop_metrics = loop.get_metrics() if hasattr(loop, "get_metrics") else {}
            decision_metrics = {
                "decision_latency_s": decision_latency,
                "has_rationale": loop_metrics.get("has_rationale", False),
                **loop_metrics,
            }

        # 4. Collect actions
        if self._organization is not None:
            env_actions = self._organization.collect_actions(agent_actions)
        else:
            env_actions = {}

        # 5. Process actions through operations paradigm (may buffer/gate)
        if self._operations_paradigm is not None:
            env_actions = self._operations_paradigm.process_action(
                env_actions, step, ground_pass_active
            )

        # 6. Environment step
        rewards: Dict[str, float] = {}
        info: Dict[str, Any] = {}
        new_observation = observation

        if self._environment is not None:
            step_result = self._environment.step(env_actions)
            new_observation = step_result.observation
            rewards = step_result.rewards
            info = step_result.info

        # 7. Update ground knowledge on downlink (communication mode during pass)
        if (
            self._operations_paradigm is not None
            and ground_pass_active
            and info.get("resolved_mode") == "communication"
        ):
            self._operations_paradigm.update_ground_knowledge(
                new_observation, step
            )

        step_duration = time.perf_counter() - step_start

        # 8. Record metrics through the collector pipeline
        if self._metrics_collector is not None:
            self._metrics_collector.record_step(
                timestep=step,
                wall_clock_seconds=step_duration,
                env_state=new_observation,
                actions=env_actions,
                rewards=rewards,
                info=info,
                decision_metrics=decision_metrics,
            )

        return {
            "step": step,
            "wall_clock_seconds": step_duration,
            "rewards": rewards,
            "info": info,
            "observation": new_observation,
        }

    def _compile_results(
        self,
        all_episode_metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compile final experiment results.

        Args:
            all_episode_metrics: List of per-episode result dicts.

        Returns:
            Full results dictionary with configuration provenance.
        """
        # Finalise experiment-level statistics
        experiment_statistics = None
        if self._metrics_collector is not None:
            stats = self._metrics_collector.finalise_experiment(
                self.config.experiment_id
            )
            # P7: Record Scale & Complexity metadata
            complexity_map = {
                "centralized": 0,
                "hierarchical": 1,
                "distributed": 2,
            }
            stats.metadata = {
                "constellation_size": self.config.environment.constellation_size,
                "complexity_index": complexity_map.get(
                    self.config.agent_organization, 0
                ),
                "agent_organization": self.config.agent_organization,
                "decision_loop": self.config.decision_loop,
                "representation": self.config.representation,
                "emergence_mode": self.config.emergence_mode,
                "operations_paradigm": self.config.operations_paradigm,
            }
            experiment_statistics = stats

        return {
            "experiment_id": self.config.experiment_id,
            "description": self.config.description,
            "config": self.config.model_dump(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_episodes": len(all_episode_metrics),
            "experiment_statistics": experiment_statistics,
            "episodes": all_episode_metrics,
        }

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save experiment results to disk.

        Args:
            results: Full results dictionary.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results_file = self.output_dir / "results.json"

        # Remove non-serialisable observation objects from step data
        serialisable = self._make_serialisable(results)

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, default=str)

        logger.info("Results saved to %s", results_file)

        # Also save a copy of the configuration
        config_copy = self.output_dir / "config.json"
        with open(config_copy, "w", encoding="utf-8") as f:
            json.dump(self.config.model_dump(), f, indent=2, default=str)

    def _save_checkpoint(
        self,
        episode_id: int,
        episode_result: Dict[str, Any],
    ) -> None:
        """Save a checkpoint after an episode.

        Args:
            episode_id: Episode index.
            episode_result: Episode result dictionary.
        """
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"episode_{episode_id:04d}.json"

        serialisable = self._make_serialisable(episode_result)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, default=str)

    @staticmethod
    def _make_serialisable(obj: Any) -> Any:
        """Recursively convert an object to a JSON-serialisable form.

        Strips non-serialisable entries (e.g. observation data classes)
        by converting them to their ``__dict__`` or string representation.
        """
        if isinstance(obj, dict):
            return {
                k: ExperimentRunner._make_serialisable(v) for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [ExperimentRunner._make_serialisable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {
                k: ExperimentRunner._make_serialisable(v)
                for k, v in dataclasses.asdict(obj).items()
            }
        elif hasattr(obj, "__dict__"):
            return ExperimentRunner._make_serialisable(obj.__dict__)
        else:
            return str(obj)

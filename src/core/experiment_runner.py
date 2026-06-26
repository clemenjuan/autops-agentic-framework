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

from src.core.config_loader import ExperimentConfig, load_config

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates a single experiment from configuration to results.

    Attributes:
        config: Validated experiment configuration.
        output_dir: Path where results and logs are saved.

    Class constants:
        TELEMETRY_SAMPLE_EPISODES: how many leading episodes keep a compact,
            downsampled per-step telemetry block in results.json (for graphs /
            the Episode inspector / presentations). The full per-step trace is
            never kept in results.json (it ballooned the file to multi-GB); this
            block is small (~1.5k points × a handful of scalar fields) and, unlike
            decisions_ep*.jsonl, is written regardless of log level.
        TELEMETRY_MAX_POINTS: downsample target per episode.
    """

    TELEMETRY_SAMPLE_EPISODES: int = 3
    TELEMETRY_MAX_POINTS: int = 1500

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
        self._ground_planner_loops: Dict[str, Any] = {}  # AH only: agent_id → ground-planner loop
        self._memory: Any = None
        self._metrics_collector: Any = None
        self._operations_paradigm: Any = None
        # Active reasoning representations. Multi-satellite RL uses one primary
        # representation per agent so each policy is bound to the right satellite.
        self._representation: Any = None
        self._representations: Dict[str, Any] = {}
        self._rollout_buffer: Any = None  # legacy in-process PPO hook; RLlib trains offline
        # Decision trace writer (active when log_level == DEBUG)
        self._decisions_file: Any = None
        # Optional full world-model trace writer for offline LeWM/Dreamer data.
        self._world_model_trace: Any = None

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

            if self.config.save_checkpoints:
                self._save_checkpoint(episode, episode_result)

            # Bound resident memory for long / parallel runs. The raw per-step
            # data (full ConstellationState observations + the per-step metric
            # list) is stripped at save anyway and is only needed in-memory for
            # the telemetry-sample episodes (and the tests that inspect the first
            # few). Drop it eagerly for the rest instead of holding every
            # episode's GB-scale steps until the end. Clearing
            # episode_metrics.step_metrics in place also frees the metrics
            # collector's shared reference (same object).
            if episode >= self.TELEMETRY_SAMPLE_EPISODES:
                episode_result.pop("steps", None)
                em = episode_result.get("episode_metrics")
                if (
                    em is not None
                    and dataclasses.is_dataclass(em)
                    and not isinstance(em, type)
                ):
                    em.step_metrics = []

            all_episode_metrics.append(episode_result)

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
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        logger.debug("Random seeds set to %d", seed)

    def _initialize_components(self) -> None:
        """Instantiate all experiment components from configuration.

        This is the integration point where the configured benchmark selectors
        are wired into concrete components.
        """
        logger.info(
            "Initialising components — org=%s, decision=%s, repr=%s, behaviour=%s, ops=%s",
            self.config.agent_organization,
            self.config.decision_procedure,
            self.config.representation,
            self.config.behaviour,
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

        # Operations paradigm
        self._operations_paradigm = self._create_operations_paradigm()

        # Anomaly recovery mode: onboard paradigms (AO/AH) clear anomalies via
        # onboard FDIR once the countdown expires; ground paradigms (AG/CG) need a
        # ground pass for the resume command. The paradigm's
        # can_self_recover_anomaly() is the single source of truth.
        if (
            self._environment is not None
            and self._operations_paradigm is not None
            and hasattr(self._environment, "anomaly_requires_ground_pass")
        ):
            self._environment.anomaly_requires_ground_pass = (
                not self._operations_paradigm.can_self_recover_anomaly()
            )

        # A Jetson-based onboard core (subsymbolic/hybrid onboard, AO/AH) keeps the
        # Jetson powered every step → extra power draw. Symbolic onboard rules run on
        # the OBC (no overhead); ground paradigms decide on the ground (no overhead).
        if self._environment is not None and hasattr(
            self._environment, "onboard_compute_active"
        ):
            self._environment.onboard_compute_active = self.config.onboard_uses_jetson

        # Metrics collector
        self._metrics_collector = self._create_metrics_collector()

        logger.info("All components initialised.")

    def _create_environment(self) -> Any:
        """Factory for the satellite environment."""
        scenario = self.config.environment.scenario
        env_cfg = {
            "constellation_size": self.config.environment.constellation_size,
            "step_duration_s": self.config.environment.timestep_seconds,
            "max_steps": self.config.max_steps,
            **self.config.environment.scenario_config,
        }

        if scenario == "eventsat":
            from src.eventsat.env import EventSatEnvironment
            # anomaly_requires_ground_pass is derived from the paradigm's
            # can_self_recover_anomaly() capability once the paradigm is built
            # (see initialise()); the env default holds until then.
            return EventSatEnvironment(config=env_cfg)

        if scenario == "multieventsat":
            from src.eventsat.multieventsat_env import MultiEventsatEnv
            return MultiEventsatEnv(config=env_cfg)

        if scenario == "ssa":
            from src.ssa.env import SSAEnvironment
            return SSAEnvironment(config=env_cfg)

        logger.warning("Unknown scenario '%s', returning None.", scenario)
        return None

    def _create_memory(self) -> Any:
        """Factory for the agent memory system.

        Returns a ``WritableMemory`` for ``writable_coala``
        configs and a ``FixedMemory`` for everything else. This runner is
        the single source of truth for the memory object — it is injected
        into every ``DecisionContext`` by the decision loops, so the
        representations' own internal memory is only a fallback. See the
        "Memory invariant" exception in CLAUDE.md: the writable semantic +
        episodic stores persist across episodes within a run (``reset()``
        deliberately keeps them) to enable CoALA learning.

        Returns:
            An initialised ``WritableMemory`` or ``FixedMemory`` instance.
        """
        mechanism = self.config.behaviour_config.get("mechanism")
        if mechanism == "writable_coala":
            from src.core.memory.writable_memory import WritableMemory

            return WritableMemory(config=self.config.memory_config)

        from src.core.memory.fixed_memory import FixedMemory

        return FixedMemory(config=self.config.memory_config)

    def _create_organization(self) -> Any:
        """Factory for the agent organization.

        Returns:
            An initialised ``AgentOrganization`` subclass.
        """
        from src.core.organization.single_agent_system import SingleAgentSystem
        from src.core.organization.centralized_mas import CentralizedMAS
        from src.core.organization.decentralized_mas import DecentralizedMAS
        from src.core.organization.independent_mas import IndependentMAS
        from src.core.organization.hybrid_mas import HybridMAS

        org_map = {
            "sas": SingleAgentSystem,
            "centralized_mas": CentralizedMAS,
            "decentralized_mas": DecentralizedMAS,
            "independent_mas": IndependentMAS,
            "hybrid_mas": HybridMAS,
        }

        # All five Kim et al. (2025) organisations are implemented; SSA is the
        # constellation-scale scenario that exercises the organisation axis.
        org_cls = org_map.get(self.config.agent_organization)
        if org_cls is None:
            raise ValueError(
                f"Unknown agent_organization: '{self.config.agent_organization}'"
            )

        org_config = dict(self.config.agent_organization_config)
        if self.config.agent_organization == "independent_mas":
            prefixes = {
                "multieventsat": "sat",
                "ssa": "sat",
                "eventsat": "eventsat",
            }
            prefix = prefixes.get(self.config.environment.scenario)
            if prefix is not None:
                org_config.setdefault("satellite_prefix", prefix)

        org = org_cls(config=org_config)
        org.initialize(
            constellation_size=self.config.environment.constellation_size,
        )
        return org

    def _create_decision_loops(self):
        """Factory for decision loop instances (one per agent)."""
        from src.core.behaviour.controller import BehaviourController
        import src.eventsat.symbolic  # register representations
        import src.eventsat.schedule_symbolic  # register schedule planner
        import src.eventsat.conventional  # register human schedule planner
        import src.eventsat.llm  # register LLM hybrid representation
        import src.eventsat.rl  # register RL subsymbolic representation
        import src.eventsat.agentic  # register agentic hybrid representation
        import src.eventsat.placeholders  # register placeholder cells/schedulers
        import src.eventsat.llm_scheduler  # register the real single-shot LLM ground planners (hllm-s/llm-s)
        import src.eventsat.agentic_scheduler  # register the real agentic LLM ground planners (hllm-a/llm-a)
        import src.eventsat.world_model  # register LeWM-CEM and DreamerV3 baselines
        import src.ssa.symbolic  # register SSA symbolic planner
        behaviour_factory = BehaviourController(config=self.config.behaviour_config)

        def with_runtime_defaults(base_config: Dict[str, Any]) -> Dict[str, Any]:
            repr_config = dict(base_config)
            repr_config.setdefault("experiment_id", self.config.experiment_id)
            repr_config.setdefault("behaviour_config", self.config.behaviour_config)
            repr_config.setdefault("max_steps", self.config.max_steps)
            return repr_config

        # Primary per-step core: the onboard core for paradigms with an onboard
        # slot (AO/AH), else the ground planner (AG/CG run their planner at passes).
        primary_repr_config = with_runtime_defaults(self.config.onboard_representation_config)
        repr_type = (
            primary_repr_config.get("type")
            or self.config.resolved_onboard_type
            or self.config.resolved_representation_type
        )
        loop_type = self.config.decision_procedure
        if loop_type != "sda":
            raise ValueError(
                f"Unsupported decision_procedure: '{loop_type}'. "
                "The EventSat benchmark now uses SDA only."
            )
        from src.core.decision_procedure.sda_loop import SDALoop
        loop_cls = SDALoop
        agents = self._organization.get_agents() if self._organization else ["central_agent"]

        if (
            self._organization is not None
            and (
                self.config.environment.scenario == "multieventsat"
                or (
                    self.config.environment.scenario == "ssa"
                    and self.config.agent_organization == "independent_mas"
                )
            )
        ):
            from src.core.organization.base import validate_agent_satellite_mapping

            validate_agent_satellite_mapping(
                self._organization,
                self._environment,
                self.config.environment.constellation_size,
                self.config.environment.scenario,
            )

        satellite_ids = set()
        if self._organization is not None:
            for agent_id in agents:
                satellite_ids.add(self._organization.satellite_for_agent(agent_id))
        use_per_agent_representations = len(satellite_ids) > 1

        loops = {}
        self._representations = {}
        self._representation = None

        if use_per_agent_representations and self._organization is not None:
            for agent_id in agents:
                agent_repr_config = dict(primary_repr_config)
                agent_repr_config["satellite_id"] = self._organization.satellite_for_agent(agent_id)
                representation = behaviour_factory.get_representation(
                    repr_type=repr_type,
                    repr_config=agent_repr_config,
                )
                if hasattr(representation, "seed"):
                    representation.seed(self.config.seed)
                self._representations[agent_id] = representation
                if self._representation is None:
                    self._representation = representation
                loops[agent_id] = loop_cls(
                    config=self.config.decision_procedure_config,
                    representation=representation,
                )
        else:
            representation = behaviour_factory.get_representation(
                repr_type=repr_type,
                repr_config=primary_repr_config,
            )
            if hasattr(representation, "seed"):
                representation.seed(self.config.seed)
            self._representation = representation
            for agent_id in agents:
                self._representations[agent_id] = representation
                loops[agent_id] = loop_cls(
                    config=self.config.decision_procedure_config,
                    representation=representation,
                )

        # Dual-slot AH: build the ground-planner core (runs at passes on the stale
        # view to refresh the uplinked plan; onboard loop above runs every step).
        self._ground_planner_loops = {}
        if (
            self.config.operations_paradigm == "autonomous_hybrid"
            and self.config.resolved_ground_planner_type is not None
        ):
            ground_repr_config = with_runtime_defaults(self.config.ground_representation_config)
            shared_gp_rep = None
            if not use_per_agent_representations:
                shared_gp_rep = behaviour_factory.get_representation(
                    repr_type=self.config.resolved_ground_planner_type,
                    repr_config=ground_repr_config,
                )
                if hasattr(shared_gp_rep, "seed"):
                    shared_gp_rep.seed(self.config.seed)
            for agent_id in agents:
                gp_rep = shared_gp_rep
                if use_per_agent_representations and self._organization is not None:
                    agent_ground_config = dict(ground_repr_config)
                    agent_ground_config["satellite_id"] = self._organization.satellite_for_agent(agent_id)
                    gp_rep = behaviour_factory.get_representation(
                        repr_type=self.config.resolved_ground_planner_type,
                        repr_config=agent_ground_config,
                    )
                    if hasattr(gp_rep, "seed"):
                        gp_rep.seed(self.config.seed)
                self._ground_planner_loops[agent_id] = loop_cls(
                    config=self.config.decision_procedure_config,
                    representation=gp_rep,
                )
        return loops

    def _create_operations_paradigm(self) -> Any:
        """Factory for the configured operations paradigm."""
        paradigm_type = self.config.operations_paradigm
        paradigm_config = self.config.operations_paradigm_config

        if paradigm_type == "autonomous_onboard":
            from src.core.operations.autonomous_onboard import AutonomousOnboard
            return AutonomousOnboard(config=paradigm_config)
        elif paradigm_type == "autonomous_hybrid":
            from src.core.operations.autonomous_hybrid import AutonomousHybrid
            return AutonomousHybrid(config=paradigm_config)
        elif paradigm_type == "autonomous_ground":
            from src.core.operations.autonomous_ground import AutonomousGround
            return AutonomousGround(config=paradigm_config)
        elif paradigm_type == "conventional_ground":
            from src.core.operations.conventional_ground import ConventionalGround
            return ConventionalGround(config=paradigm_config)
        else:
            logger.warning(
                "Unknown operations_paradigm '%s', falling back to autonomous_hybrid.",
                paradigm_type,
            )
            from src.core.operations.autonomous_hybrid import AutonomousHybrid
            return AutonomousHybrid(config=paradigm_config)

    def _create_metrics_collector(self) -> Any:
        """Factory for the metrics collector."""
        scenario = self.config.environment.scenario
        if scenario in ("eventsat", "multieventsat"):
            # MultiEventsat exposes EventSat-compatible aggregate telemetry.
            from src.eventsat.metrics import EventSatMetricsCollector
            metrics_cfg = self.config.metrics.model_dump()
            # Pass environment parameters needed for energy/utility computation
            metrics_cfg["max_steps"] = self.config.max_steps
            metrics_cfg["step_duration_s"] = self.config.environment.timestep_seconds
            if self._environment is not None and hasattr(self._environment, "battery_capacity_wh"):
                metrics_cfg["battery_capacity_wh"] = self._environment.battery_capacity_wh
            return EventSatMetricsCollector(config=metrics_cfg)
        if scenario == "ssa":
            from src.ssa.metrics import SSAMetricsCollector
            metrics_cfg = self.config.metrics.model_dump()
            metrics_cfg["constellation_size"] = self.config.environment.constellation_size
            metrics_cfg["max_steps"] = self.config.max_steps
            metrics_cfg["step_duration_s"] = self.config.environment.timestep_seconds
            if self._environment is not None and hasattr(self._environment, "battery_capacity_wh"):
                metrics_cfg["battery_capacity_wh"] = self._environment.battery_capacity_wh
            return SSAMetricsCollector(config=metrics_cfg)
        logger.warning("No metrics collector for scenario '%s'.", scenario)
        return None

    def _core_representations(self) -> List[Any]:
        """Active reasoning cores: the onboard/primary representation plus any
        dual-slot AH ground-planner representations."""
        reps: List[Any] = []
        seen: set[int] = set()

        def add(rep: Any) -> None:
            if rep is None:
                return
            ident = id(rep)
            if ident not in seen:
                seen.add(ident)
                reps.append(rep)

        add(self._representation)
        for rep in self._representations.values():
            add(rep)
        for loop in self._ground_planner_loops.values():
            add(getattr(loop, "representation", None))
        return reps

    def _collect_core_llm_metrics(self) -> Dict[str, float]:
        """Gather per-episode ``llm_*`` metrics (api calls, latency, cache, tokens)
        from every active core. Empty for non-LLM cells."""
        out: Dict[str, float] = {}
        for rep in self._core_representations():
            if not hasattr(rep, "get_metrics"):
                continue
            for k, v in rep.get_metrics().items():
                if k.startswith("llm_") and isinstance(v, (int, float)):
                    out[k] = float(v)
        return out

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

        # Capture the actual per-episode orbit + pass schedule (if the scenario
        # exposes them) so results.json reproduces the exact simulated orbit for
        # analysis and ground-track figures.
        episode_orbit = None
        episode_ground_passes = None
        if self._environment is not None and hasattr(self._environment, "get_episode_orbit"):
            episode_orbit = self._environment.get_episode_orbit()
            if hasattr(self._environment, "get_ground_passes"):
                episode_ground_passes = self._environment.get_ground_passes()

        if self._memory is not None:
            self._memory.reset()

        for loop in self._decision_loops.values():
            loop.reset()

        # Zero per-episode LLM metric counters on every active core (onboard +
        # ground planner) so the episode's aggregated llm_* metrics are per-episode,
        # not cumulative. The response cache is preserved.
        for rep in self._core_representations():
            client = getattr(rep, "_client", None)
            if client is not None and hasattr(client, "reset_metrics"):
                client.reset_metrics()

        if self._operations_paradigm is not None:
            self._operations_paradigm.reset()

        self._world_model_trace = None
        trace_dir = self.config.representation_config.get("world_model_trace_dir")
        scenario = self.config.environment.scenario
        if trace_dir and scenario == "ssa":
            from src.ssa.trace import ConstellationTraceEpisode

            self._world_model_trace = ConstellationTraceEpisode(
                episode_id=episode_id,
                seed=self.config.seed + episode_id,
            )
        elif trace_dir and scenario == "eventsat":
            from src.eventsat.trace import WorldModelTraceEpisode

            self._world_model_trace = WorldModelTraceEpisode(
                episode_id=episode_id,
                seed=self.config.seed + episode_id,
            )

        # --- Decision trace (active when log_level == DEBUG) ---
        if self.config.log_level.upper() == "DEBUG":
            decisions_path = self.output_dir / f"decisions_ep{episode_id}.jsonl"
            self._decisions_file = open(
                decisions_path, "w", encoding="utf-8"
            )
        else:
            self._decisions_file = None

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

        # --- Close decision trace ---
        if self._decisions_file is not None:
            self._decisions_file.close()
            self._decisions_file = None

        if self._world_model_trace is not None:
            trace_dir = Path(self.config.representation_config["world_model_trace_dir"])
            self._world_model_trace.write_npz(
                trace_dir / f"episode_{episode_id:05d}.npz"
            )
            self._world_model_trace = None

        episode_duration = time.perf_counter() - episode_start

        # --- RL training update (learned mode only) ---
        if (
            self._rollout_buffer is not None
            and self._representation is not None
            and self._rollout_buffer.size > 0
        ):
            self._representation.update({
                "buffer": self._rollout_buffer,
                "episode": episode_id,
            })
            self._rollout_buffer.reset()

        # --- Finalise episode metrics ---
        episode_metrics = None
        if self._metrics_collector is not None:
            episode_metrics = self._metrics_collector.finalise_episode(episode_id)
            # Merge per-core LLM metrics (api calls, latency, cache, tokens) into the
            # episode's aggregated dict. finalise_episode returns the same object the
            # collector stores, so this also propagates to experiment_statistics.
            if episode_metrics is not None and hasattr(episode_metrics, "aggregated"):
                episode_metrics.aggregated.update(self._collect_core_llm_metrics())

        # Paradigm-level metrics (e.g. AH onboard_overrides) — captured before the
        # next episode's reset() clears them.
        paradigm_metrics = (
            self._operations_paradigm.get_metrics()
            if self._operations_paradigm is not None else {}
        )

        return {
            "episode_id": episode_id,
            "num_steps": len(step_data),
            "wall_clock_seconds": episode_duration,
            "episode_metrics": episode_metrics,
            "paradigm_metrics": paradigm_metrics,
            "orbital_elements": episode_orbit,
            "ground_passes": episode_ground_passes,
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

        # 3. Check if inference is allowed (ground paradigms skip between passes)
        inference_allowed = True
        if self._operations_paradigm is not None:
            inference_allowed = self._operations_paradigm.should_allow_inference(
                step, ground_pass_active
            )

        # 4. Decision loops (timed for latency metric)
        agent_actions = {}
        decision_metrics: Dict[str, Any] = {"inference_allowed": inference_allowed}
        from src.core.organization.base import AgentAction

        if inference_allowed:
            for agent_id, loop in self._decision_loops.items():
                obs = agent_obs.get(agent_id)
                t0 = time.perf_counter()
                action, self._memory = loop.process(obs, self._memory)
                decision_latency = time.perf_counter() - t0

                agent_actions[agent_id] = AgentAction(
                    agent_id=agent_id, action=action
                )
                # Collect decision loop metrics (latency, rationale, etc.)
                loop_metrics = (
                    loop.get_metrics() if hasattr(loop, "get_metrics") else {}
                )
                decision_metrics.update({
                    # Accumulate latency across all agents (important for
                    # hierarchical org where manager + local run sequentially).
                    "decision_latency_s": (
                        decision_metrics.get("decision_latency_s", 0.0)
                        + decision_latency
                    ),
                    "has_rationale": loop_metrics.get("has_rationale", False),
                    **loop_metrics,
                })
        else:
            # Between passes for ground paradigms: no inference, schedule
            # playback in process_action() handles the action.
            for agent_id in self._decision_loops:
                fallback_mode = getattr(self, "_last_action_mode", "charging")
                satellite_id = (
                    self._organization.satellite_for_agent(agent_id)
                    if self._organization is not None
                    else "eventsat_0"
                )
                agent_actions[agent_id] = AgentAction(
                    agent_id=agent_id,
                    action={satellite_id: {"mode": fallback_mode}},
                )
            decision_metrics.update({
                "decision_latency_s": 0.0,
                "has_rationale": False,
                "inference_skipped": True,
            })

        # 5. Collect actions (the onboard core's per-step action)
        if self._organization is not None:
            env_actions = self._organization.collect_actions(agent_actions)
            # Surface organisation-level coordination cost (e.g. all-to-all
            # message count / consensus rounds for DecentralizedMAS) so the
            # cost side of the organisation axis is measurable.
            org_metrics = self._organization.get_metrics()
            if org_metrics:
                decision_metrics.update(org_metrics)
        else:
            env_actions = {}

        # 5b. Dual-slot AH: at ground passes, refresh the uplinked plan by running
        # the ground planner on the stale ground view; the onboard action above is
        # then arbitrated against this plan in process_action.
        if self._ground_planner_loops and ground_pass_active and observation is not None:
            stale_obs = self._operations_paradigm.ground_planner_view(observation, step)
            gp_obs = (
                self._organization.distribute_observation(stale_obs)
                if (self._organization is not None and stale_obs is not None)
                else {}
            )
            gp_latency = 0.0
            for agent_id, gp_loop in self._ground_planner_loops.items():
                gp_t0 = time.perf_counter()
                gp_action, self._memory = gp_loop.process(
                    gp_obs.get(agent_id), self._memory
                )
                gp_latency += time.perf_counter() - gp_t0
                self._operations_paradigm.set_uplinked_plan(gp_action)
            # Surface the ground-planning wall-clock for M-07 (recorded below).
            decision_metrics["ground_decision_latency_s"] = gp_latency

        # 6. Process actions through operations paradigm (may buffer/gate)
        if self._operations_paradigm is not None:
            env_actions = self._operations_paradigm.process_action(
                env_actions, step, ground_pass_active
            )

        # 7. Environment step
        rewards: Dict[str, float] = {}
        info: Dict[str, Any] = {}
        new_observation = observation

        if self._environment is not None:
            step_result = self._environment.step(env_actions)
            new_observation = step_result.observation
            rewards = step_result.rewards
            info = step_result.info

        # 7b. Collect RL trajectory step (learned mode only)
        if self._rollout_buffer is not None and self._representation is not None:
            step_data_rl = None
            if hasattr(self._representation, "get_last_step_data"):
                step_data_rl = self._representation.get_last_step_data()
            if step_data_rl is not None and not self._rollout_buffer.is_full:
                scalar_reward = float(sum(rewards.values())) if rewards else 0.0
                done = (
                    self._environment.is_done()
                    if self._environment is not None else False
                )
                self._rollout_buffer.store(
                    obs=step_data_rl["obs_vec"],
                    action=step_data_rl["action_vec"],
                    reward=scalar_reward,
                    value=step_data_rl["value"],
                    log_prob=step_data_rl["log_prob"],
                    done=done,
                )

        # 8. Update ground knowledge on downlink (communication mode during pass)
        if (
            self._operations_paradigm is not None
            and ground_pass_active
            and info.get("resolved_mode") == "communication"
        ):
            self._operations_paradigm.update_ground_knowledge(
                new_observation, step
            )

        step_duration = time.perf_counter() - step_start

        # 9. Record metrics through the collector pipeline
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

        if self._world_model_trace is not None and observation is not None:
            self._world_model_trace.record(
                step=step,
                observation=observation,
                env_actions=env_actions,
                rewards=rewards,
                info=info,
            )

        # 10. Write decision trace (DEBUG only).
        # Includes raw env telemetry so research metrics can be recomputed
        # offline from this file alone (see scripts/recompute_metrics.py).
        if self._decisions_file is not None:
            trace_entry = {
                "step": step,
                "mode": info.get("resolved_mode", "unknown"),
                "requested_mode": info.get("requested_mode"),
                "forced": bool(info.get("forced", False)),
                # Pre-transition safety classification (M-05/M-13 ground truth).
                "safety_safe": float(info.get("safety_safe", 0.0)),
                "anomaly": info.get("anomaly") or "",
                "anomaly_forced_safe": float(info.get("anomaly_forced_safe", 0.0)),
                "rationale": decision_metrics.get("rationale", ""),
                "has_rationale": bool(decision_metrics.get("has_rationale", False)),
                "inference": inference_allowed,
                "latency_s": decision_metrics.get("decision_latency_s", 0.0),
                "battery_soc": info.get("battery_soc"),
                "in_sunlight": info.get("in_sunlight"),
                "ground_pass_active": info.get("ground_pass_active"),
                "jetson_raw_mb": info.get("jetson_raw_mb"),
                "jetson_compressed_mb": info.get("jetson_compressed_mb"),
                "obc_data_mb": info.get("obc_data_mb"),
                "data_downlinked_mb": info.get("data_downlinked_mb"),
                "step_downlinked_mb": info.get("step_downlinked_mb", 0.0),
                "total_raw_captured_mb": info.get("total_raw_captured_mb", 0.0),
                "downlink_raw_equivalent_mb": info.get("downlink_raw_equivalent_mb", 0.0),
                "observation_hours": info.get("observation_hours"),
                "total_detections": info.get("total_detections"),
                "undetected_observations": info.get("undetected_observations"),
                "max_achievable_downlink_mb": info.get("max_achievable_downlink_mb"),
                "reward": info.get("reward"),
                "prev_battery_soc": info.get("prev_battery_soc"),
                "data_stored_mb": info.get("data_stored_mb"),
                "in_transition": info.get("in_transition"),
                # Loop-specific diagnostics (zero on loops that don't emit them)
            }
            self._decisions_file.write(json.dumps(trace_entry) + "\n")

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
            # P7: Record Scale & Complexity metadata (Kim et al. 2025 taxonomy)
            complexity_map = {
                "sas": 0,
                "centralized_mas": 1,
                "decentralized_mas": 2,
                "independent_mas": 3,
                "hybrid_mas": 4,
            }
            stats.metadata = {
                "constellation_size": self.config.environment.constellation_size,
                "complexity_index": complexity_map.get(
                    self.config.agent_organization, 0
                ),
                "agent_organization": self.config.agent_organization,
                "decision_procedure": self.config.decision_procedure,
                "representation": self.config.representation,
                "behaviour": self.config.behaviour,
                "operations_paradigm": self.config.operations_paradigm,
                # Flag placeholder schedule-producers (ground-paradigm stand-ins)
                # so analysis can exclude them from headline comparisons until the
                # real RL/LLM schedulers land (see placeholders.py).
                "representation_is_placeholder": bool(
                    getattr(self._representation, "is_placeholder", False)
                ),
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

        # results.json is the compact, experiment-level artifact. Strip the raw
        # per-step payloads (multi-GB constellation_state snapshots + per-step
        # metric lists) before serialising — they live only in
        # decisions_ep*.jsonl (DEBUG). Keeps experiment_statistics and per-episode
        # AGGREGATED metrics. See _strip_per_step_data.
        serialisable = self._make_serialisable(
            self._strip_per_step_data(results)
        )

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

        serialisable = self._make_serialisable(
            self._strip_episode_for_disk(episode_result, with_telemetry=True)
        )
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, default=str)

    # Critical per-step fields kept in the compact telemetry block. Names match
    # what scripts/extract_telemetry.py and the board's Episode inspector read.
    _TELEMETRY_FIELDS = (
        ("soc", "battery_soc", 4),
        ("stored", "data_stored_mb", 2),
        ("downlinked", "data_downlinked_mb", 2),
        ("jetson_raw", "jetson_raw_mb", 2),
        ("obc", "obc_data_mb", 2),
    )

    @classmethod
    def _compact_telemetry(cls, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Downsampled, scalar-only per-step telemetry for graphs / inspector.

        Pulls the critical fields (battery, mode, data pools, ground-pass,
        anomaly) from each step's ``info`` dict, downsampled to at most
        ``TELEMETRY_MAX_POINTS`` evenly-spaced points. Tens of KB per episode,
        vs the multi-GB raw observation dump it replaces.
        """
        if not steps:
            return {}
        k = max(1, len(steps) // cls.TELEMETRY_MAX_POINTS)
        rows = steps[::k]
        infos = [r.get("info", {}) or {} for r in rows]
        tel: Dict[str, Any] = {
            "steps": [r.get("step") for r in rows],
            "mode": [i.get("resolved_mode", "?") for i in infos],
            "gpass": [int(bool(i.get("ground_pass_active"))) for i in infos],
            "sunlight": [int(bool(i.get("in_sunlight"))) for i in infos],
            "anomaly": [int(bool(i.get("anomaly_forced_safe"))) for i in infos],
        }
        for out_key, info_key, ndigits in cls._TELEMETRY_FIELDS:
            tel[out_key] = [round(i.get(info_key) or 0.0, ndigits) for i in infos]
        return tel

    @classmethod
    def _strip_episode_for_disk(
        cls, episode: Dict[str, Any], with_telemetry: bool = False
    ) -> Dict[str, Any]:
        """Drop raw per-step payloads from one episode result dict.

        Removes the multi-GB raw observation snapshots (``steps`` — each entry
        carries a full ``ConstellationState`` with every satellite's metadata)
        and the per-step metric list (``episode_metrics.step_metrics``), keeping
        only the per-episode AGGREGATED metrics. The full per-step trace lives
        only in ``decisions_ep*.jsonl`` (written when ``log_level == DEBUG``).

        When ``with_telemetry`` is set, a compact downsampled ``telemetry`` block
        of the critical scalar fields is attached before the raw steps are
        dropped — small enough for results.json, enough for graphs / the Episode
        inspector, and present regardless of log level.
        """
        out = {k: v for k, v in episode.items() if k != "steps"}
        if with_telemetry:
            tel = cls._compact_telemetry(episode.get("steps", []) or [])
            if tel:
                out["telemetry"] = tel
        em = out.get("episode_metrics")
        if (
            em is not None
            and dataclasses.is_dataclass(em)
            and not isinstance(em, type)
        ):
            out["episode_metrics"] = dataclasses.replace(em, step_metrics=[])
        return out

    @staticmethod
    def _strip_per_step_data(results: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of the full results with raw per-step data removed.

        results.json is the compact, experiment-level artifact: experiment_id,
        config, timestamp, num_episodes, experiment_statistics and per-episode
        AGGREGATED metrics. Raw per-step observation/state snapshots are excluded
        by design — they balloon the file to multi-GB and merely duplicate the
        per-step ``decisions_ep*.jsonl`` trace (see scripts/recompute_metrics.py,
        which reads that trace, not this file). Three leak sites are pruned:

        1. ``episodes[].steps`` — raw ``ConstellationState`` observation dumps.
        2. ``episodes[].episode_metrics.step_metrics`` — per-step scalar list.
        3. ``experiment_statistics.raw_episodes[].step_metrics`` — same list again.
        """
        out = dict(results)

        stats = out.get("experiment_statistics")
        if (
            stats is not None
            and dataclasses.is_dataclass(stats)
            and not isinstance(stats, type)
        ):
            out["experiment_statistics"] = dataclasses.replace(
                stats,
                raw_episodes=[
                    dataclasses.replace(em, step_metrics=[])
                    for em in stats.raw_episodes
                ],
            )

        out["episodes"] = [
            ExperimentRunner._strip_episode_for_disk(
                ep, with_telemetry=(i < ExperimentRunner.TELEMETRY_SAMPLE_EPISODES)
            )
            for i, ep in enumerate(out.get("episodes", []))
        ]
        return out

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

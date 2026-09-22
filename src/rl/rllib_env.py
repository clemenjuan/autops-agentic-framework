"""RLlib MultiAgentEnv bridge for AUTOPS experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    from ray.rllib.env.multi_agent_env import MultiAgentEnv

    RLLIB_AVAILABLE = True
except ImportError:
    DefaultCallbacks = object  # type: ignore[misc,assignment]
    MultiAgentEnv = object  # type: ignore[misc,assignment]
    RLLIB_AVAILABLE = False

from src.core.organization.base import (
    AgentAction,
    bind_communication_topology,
    validate_agent_satellite_mapping,
)
from src.core.config_loader import ExperimentConfig, validate_runtime_support
from src.rl.space_adapters import RLSpaceAdapter, make_space_adapter

_GROUND_ONLY_PARADIGMS = {"autonomous_ground", "conventional_ground"}

# Per-episode EventSat behaviour counters for training logs. Physical contact
# truth is used only here, never as a policy input.
_FAILED_MODE_KEYS = {
    "payload_observe": "failed_observe",
    "payload_compress": "failed_compress",
    "payload_detect": "failed_detect",
    "payload_send": "failed_send",
    "communication": "failed_comm",
}


def _empty_eventsat_diagnostics() -> Dict[str, float]:
    keys = (
        "downlinked_mb",
        "failed_action_penalty",
        "observations",
        "compressions",
        "settling_steps",
        "comm_steps_in_contact",
        "comm_steps_no_contact",
        *_FAILED_MODE_KEYS.values(),
        "failed_clamped",
        "final_raw_mb",
        "final_compressed_mb",
        "final_obc_mb",
    )
    return {key: 0.0 for key in keys}


def _accumulate_eventsat_diagnostics(
    diagnostics: Dict[str, float], info: Dict[str, Any]
) -> None:
    """Add one EventSat step; downlink and buffer levels in ``info`` are totals."""
    diagnostics["downlinked_mb"] = float(info["data_downlinked_mb"])
    penalty = float(info["failed_action_penalty"])
    diagnostics["failed_action_penalty"] += penalty
    mode = info.get("resolved_mode")
    diagnostics["observations"] += float(bool(info.get("observation_accepted")))
    diagnostics["compressions"] += float(bool(info.get("compression_completed")))
    diagnostics["settling_steps"] += float(bool(info.get("in_transition")))
    if mode == "communication":
        if info.get("communication_failure") == "no_contact":
            diagnostics["comm_steps_no_contact"] += 1.0
        elif info.get("pass_active"):
            diagnostics["comm_steps_in_contact"] += 1.0
    if info.get("failed_action"):
        # A failed charging step is an operational command clamped to charging.
        diagnostics[_FAILED_MODE_KEYS.get(mode, "failed_clamped")] += 1.0
    diagnostics["final_raw_mb"] = float(info.get("jetson_raw_mb", 0.0))
    diagnostics["final_compressed_mb"] = float(info.get("jetson_compressed_mb", 0.0))
    diagnostics["final_obc_mb"] = float(info.get("obc_data_mb", 0.0))


class AUTOPSEpisodeDiagnostics(DefaultCallbacks):
    """Attach EventSat totals to each completed RLlib episode (PPO, Schulman 2017)."""

    def on_episode_end(self, *, episode, base_env, env_index, **kwargs) -> None:
        if isinstance(episode, Exception):
            return
        env = base_env.get_sub_environments()[env_index]
        for key, value in env.get_episode_diagnostics().items():
            episode.hist_data[f"eventsat_{key}"] = [value]


class AUTOPSRLLibMultiAgentEnv(MultiAgentEnv):  # type: ignore[misc]
    """Expose an AUTOPS experiment as an RLlib multi-agent environment.

    Each agent observes/controls a specific satellite, given by
    ``organization.satellite_for_agent(agent_id)``. The bridge holds one space
    adapter per agent (parametrised by that satellite_id) so encode/decode/reward
    all target the right satellite. Single-satellite scenarios (eventsat) map
    every agent to the one canonical satellite, reproducing legacy behaviour.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: Dict[str, Any] | None = None) -> None:
        super().__init__()
        env_config = env_config or {}
        raw_config = env_config.get("experiment_config", env_config)
        self.config = (
            raw_config
            if isinstance(raw_config, ExperimentConfig)
            else ExperimentConfig(**raw_config)
        )
        validate_runtime_support(self.config)
        if self.config.operations_paradigm in _GROUND_ONLY_PARADIGMS:
            raise ValueError(
                "RLlib PPO training does not support ground-only paradigms "
                f"({self.config.operations_paradigm}) because the current RL "
                "action space emits per-step mode commands, not time-tagged "
                "ground schedules. Use an onboard/AH RL cell or implement a "
                "schedule-producing RLlib adapter instead of training a "
                "different MDP silently."
            )
        if (
            self.config.operations_paradigm == "autonomous_hybrid"
            and self.config.environment.scenario != "eventsat"
        ):
            raise ValueError(
                "RLlib autonomous_hybrid training is currently implemented only "
                "for EventSat, whose AH operations paradigm arbitrates an "
                "onboard mode against an EventSat ground schedule."
            )
        self._environment = self._create_environment()
        self._organization = self._create_organization()
        self._operations_paradigm = self._create_operations_paradigm()
        self._configure_environment_capabilities()
        self._memory = self._create_memory()
        self._ground_planner_loops: Dict[str, Any] = self._create_ground_planner_loops()

        self.possible_agents: List[str] = list(self._organization.get_agents())
        validate_agent_satellite_mapping(
            self._organization,
            self._environment,
            self.config.environment.constellation_size,
            self.config.environment.scenario,
        )
        bind_communication_topology(self._organization, self._environment)
        self._adapters: Dict[str, RLSpaceAdapter] = self._create_adapters(self.possible_agents)
        self._space_adapter: RLSpaceAdapter = (
            self._adapters[self.possible_agents[0]]
            if self.possible_agents
            else self._build_adapter(agent_id=None)
        )

        self.agents: List[str] = []
        self._agent_ids = set(self.possible_agents)
        self.observation_space = self._space_adapter.observation_space
        self.action_space = self._space_adapter.action_space
        self.observation_spaces = {
            agent_id: self._adapter_for(agent_id).observation_space
            for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: self._adapter_for(agent_id).action_space for agent_id in self.possible_agents
        }
        self._last_observation: Any = None
        self._episode_diagnostics: Dict[str, float] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        self._organization.initialize(
            constellation_size=self.config.environment.constellation_size,
        )
        if self._operations_paradigm is not None:
            self._operations_paradigm.reset()
        self.agents = list(self.possible_agents)
        self._last_observation = self._environment.reset(seed=seed)
        self._episode_diagnostics = (
            _empty_eventsat_diagnostics()
            if self.config.environment.scenario == "eventsat"
            else {}
        )
        bind_communication_topology(self._organization, self._environment)
        self._last_observation = self._environment.get_observation()
        observations = self._encode_current_observations(done=False)
        infos = {agent_id: {"agent_id": agent_id} for agent_id in self.agents}
        return observations, infos

    def step(
        self,
        action_dict: Dict[str, Any],
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        active_agents = list(self.agents)
        step_idx = self._current_step()
        policy_observation = self._operations_paradigm.filter_observation(
            self._last_observation,
            step_idx,
        )
        agent_observations = self._organization.distribute_observation(
            policy_observation,
        )
        agent_actions: Dict[str, AgentAction] = {}
        for agent_id in active_agents:
            if agent_id not in action_dict:
                continue
            adapter = self._adapter_for(agent_id)
            decoded_action = adapter.decode_action(action_dict[agent_id], agent_id=agent_id)
            agent_actions[agent_id] = AgentAction(
                agent_id=agent_id,
                action=adapter.ground_decoded_action(
                    decoded_action,
                    agent_observations.get(agent_id),
                ),
            )

        ground_pass_active = self._ground_pass_active(self._last_observation)
        env_actions = self._organization.collect_actions(agent_actions)
        if self._ground_planner_loops and ground_pass_active and self._last_observation is not None:
            stale_obs = self._operations_paradigm.ground_planner_view(
                self._last_observation,
                step_idx,
            )
            gp_obs = self._organization.distribute_observation(stale_obs)
            for agent_id, loop in self._ground_planner_loops.items():
                gp_action, self._memory = loop.process(gp_obs.get(agent_id), self._memory)
                self._operations_paradigm.set_uplinked_plan(gp_action)

        env_actions = self._operations_paradigm.process_action(
            env_actions,
            step_idx,
            ground_pass_active,
        )
        step_result = self._environment.step(env_actions)
        if self._episode_diagnostics:
            # Read before dropping terminal observations/infos.
            _accumulate_eventsat_diagnostics(self._episode_diagnostics, step_result.info)
        self._last_observation = step_result.observation
        done = bool(self._environment.is_done())

        if ground_pass_active and step_result.info.get("resolved_mode") == "communication":
            self._operations_paradigm.update_ground_knowledge(
                self._last_observation,
                step_idx,
            )

        if done:
            self.agents = []

        observations = self._encode_current_observations(done=done)
        rewards = {
            agent_id: self._resolve_agent_reward(agent_id, step_result.rewards)
            for agent_id in active_agents
        }
        terminateds = {agent_id: done for agent_id in active_agents}
        truncateds = {agent_id: False for agent_id in active_agents}
        terminateds["__all__"] = done
        truncateds["__all__"] = False
        infos = {
            agent_id: {"agent_id": agent_id, **dict(step_result.info)} for agent_id in observations
        }
        return observations, rewards, terminateds, truncateds, infos

    def render(self) -> None:
        return None

    def get_episode_diagnostics(self) -> Dict[str, float]:
        return dict(self._episode_diagnostics)

    def close(self) -> None:
        return None

    @property
    def space_adapter(self) -> RLSpaceAdapter:
        return self._space_adapter

    # ------------------------------------------------------------------
    # Reward resolution
    # ------------------------------------------------------------------

    def _resolve_agent_reward(self, agent_id: str, raw_rewards: Dict[str, float]) -> float:
        """Map an environment reward dict to a single per-agent reward.

        The organisation's action scope is authoritative. For multi-satellite
        agents, sum the rewards for controlled satellites so the total reward
        mass is invariant to the grouping. If the reward dict is not keyed by
        those satellites, fall back to the adapter scalar reducer.
        """
        sat_ids = list(self._organization.satellites_for_agent(agent_id))
        if not sat_ids:
            return 0.0
        scoped = [float(raw_rewards[sat_id]) for sat_id in sat_ids if sat_id in raw_rewards]
        if scoped:
            return float(sum(scoped))
        return self._adapter_for(agent_id).scalar_reward(raw_rewards)

    # ------------------------------------------------------------------
    # Operations-paradigm parity helpers
    # ------------------------------------------------------------------

    def _current_step(self) -> int:
        return int(getattr(self._environment, "current_step", 0))

    @staticmethod
    def _ground_pass_active(observation: Any) -> bool:
        if observation is None:
            return False
        return any(
            bool(sat.metadata.get("ground_pass_active", False))
            for sat in observation.constellation_state.satellites.values()
        )

    def _encode_current_observations(self, *, done: bool) -> Dict[str, Any]:
        if done or self._last_observation is None:
            return {}
        filtered = self._operations_paradigm.filter_observation(
            self._last_observation,
            self._current_step(),
        )
        agent_obs = self._organization.distribute_observation(filtered)
        return {
            agent_id: self._adapter_for(agent_id).encode_observation(agent_obs.get(agent_id))
            for agent_id in self.agents
        }

    def _configure_environment_capabilities(self) -> None:
        if hasattr(self._environment, "anomaly_requires_ground_pass"):
            self._environment.anomaly_requires_ground_pass = (
                not self._operations_paradigm.can_self_recover_anomaly()
            )
        if hasattr(self._environment, "onboard_compute_active"):
            self._environment.onboard_compute_active = self.config.onboard_uses_jetson

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    def _adapter_for(self, agent_id: str) -> RLSpaceAdapter:
        return self._adapters.get(agent_id, self._space_adapter)

    def _create_environment(self) -> Any:
        scenario = self.config.environment.scenario
        env_cfg = {
            **self.config.environment.scenario_config,
            "constellation_size": self.config.environment.constellation_size,
            "step_duration_s": self.config.environment.timestep_seconds,
            "max_steps": self.config.max_steps,
            "scenario": scenario,
            "seed": self.config.seed,
            "reward_discount_factor": float(
                self.config.behaviour_config.get("gamma", 1.0)
            ),
        }
        from src.core.scenario_registry import get_scenario_spec

        spec = get_scenario_spec(scenario)
        if spec is None:
            raise ValueError(f"No RLlib environment registered for scenario '{scenario}'")
        env_cfg["constellation_size"] = self.config.environment.constellation_size
        return spec.env_loader()(config=env_cfg)

    def _create_operations_paradigm(self) -> Any:
        paradigm_type = self.config.operations_paradigm
        paradigm_config = self.config.operations_paradigm_config
        if paradigm_type == "autonomous_onboard":
            from src.core.operations.autonomous_onboard import AutonomousOnboard

            return AutonomousOnboard(config=paradigm_config)
        if paradigm_type == "autonomous_hybrid":
            from src.core.operations.autonomous_hybrid import AutonomousHybrid

            return AutonomousHybrid(config=paradigm_config)
        raise ValueError(f"Unsupported RLlib operations_paradigm: {paradigm_type!r}")

    def _create_memory(self) -> Any:
        from src.core.memory.fixed_memory import FixedMemory

        return FixedMemory(config=self.config.memory_config)

    def _create_ground_planner_loops(self) -> Dict[str, Any]:
        if (
            self.config.operations_paradigm != "autonomous_hybrid"
            or self.config.resolved_ground_planner_type is None
        ):
            return {}

        from src.core.behaviour.controller import BehaviourController
        from src.core.decision_procedure.sda_loop import SDALoop

        import src.eventsat.agentic  # register agentic hybrid representation
        import src.eventsat.agentic_scheduler  # register agentic ground planners
        import src.eventsat.conventional  # register human schedule planner
        import src.eventsat.llm  # register LLM hybrid representation
        import src.eventsat.llm_scheduler  # register single-shot LLM ground planners
        import src.eventsat.placeholders  # register placeholder cells/schedulers
        import src.eventsat.rl  # register RL subsymbolic representation
        import src.eventsat.schedule_symbolic  # register schedule planner
        import src.eventsat.symbolic  # register representations

        behaviour_factory = BehaviourController(config=self.config.behaviour_config)
        agents = self._organization.get_agents()
        use_per_agent_representations = len(agents) > 1
        ground_repr_config = self._runtime_representation_config(
            self.config.ground_representation_config
        )
        shared_rep = None
        if not use_per_agent_representations:
            shared_rep = behaviour_factory.get_representation(
                repr_type=self.config.resolved_ground_planner_type,
                repr_config=self._representation_config_for_agent(
                    ground_repr_config,
                    agents[0],
                ),
            )
            self._reject_placeholder_ground_planner(shared_rep)
            if hasattr(shared_rep, "seed"):
                shared_rep.seed(self.config.seed)

        loops: Dict[str, Any] = {}
        for agent_id in agents:
            rep = shared_rep
            if use_per_agent_representations:
                rep = behaviour_factory.get_representation(
                    repr_type=self.config.resolved_ground_planner_type,
                    repr_config=self._representation_config_for_agent(
                        ground_repr_config,
                        agent_id,
                    ),
                )
                self._reject_placeholder_ground_planner(rep)
                if hasattr(rep, "seed"):
                    rep.seed(self.config.seed)
            loops[agent_id] = SDALoop(
                config=self.config.decision_procedure_config,
                representation=rep,
            )
        return loops

    def _reject_placeholder_ground_planner(self, representation: Any) -> None:
        if getattr(representation, "is_placeholder", False):
            raise ValueError(
                "RLlib autonomous_hybrid training requires a real ground planner. "
                f"Resolved '{self.config.resolved_ground_planner_type}' is a "
                "placeholder stand-in, so training would mix the learned onboard "
                "policy with an unintended symbolic scheduler."
            )

    def _runtime_representation_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        repr_config = dict(base_config)
        repr_config.setdefault("experiment_id", self.config.experiment_id)
        repr_config.setdefault("behaviour_config", self.config.behaviour_config)
        repr_config.setdefault("max_steps", self.config.max_steps)
        repr_config.setdefault("scenario", self.config.environment.scenario)
        return repr_config

    def _representation_config_for_agent(
        self,
        base_config: Dict[str, Any],
        agent_id: str,
    ) -> Dict[str, Any]:
        from src.rl.policy_mapping import PolicySharingConfig

        policy_sharing = PolicySharingConfig.from_config(
            self.config.behaviour_config.get("policy_sharing", {"mode": "shared_all"})
        )
        agent_config = dict(base_config)
        act_ids = list(self._organization.satellites_for_agent(agent_id))
        observe_ids = list(self._organization.observed_satellites_for_agent(agent_id))
        agent_config["act_ids"] = act_ids
        agent_config["observe_ids"] = observe_ids
        agent_config["policy_id"] = policy_sharing.policy_id_for(agent_id)
        if act_ids:
            agent_config["satellite_id"] = act_ids[0]
        elif observe_ids:
            agent_config["satellite_id"] = observe_ids[0]
        return agent_config

    def _create_organization(self) -> Any:
        from src.core.organization.centralized_mas import CentralizedMAS
        from src.core.organization.decentralized_mas import DecentralizedMAS
        from src.core.organization.hybrid_mas import HybridMAS
        from src.core.organization.independent_mas import IndependentMAS
        from src.core.organization.single_agent_system import SingleAgentSystem

        org_map = {
            "sas": SingleAgentSystem,
            "centralized_mas": CentralizedMAS,
            "decentralized_mas": DecentralizedMAS,
            "independent_mas": IndependentMAS,
            "hybrid_mas": HybridMAS,
        }
        org_cls = org_map.get(self.config.agent_organization)
        if org_cls is None:
            raise ValueError(f"Unknown agent_organization: '{self.config.agent_organization}'")
        if (
            self.config.environment.scenario == "multieventsat"
            and self.config.agent_organization != "independent_mas"
        ):
            raise ValueError(
                f"Organization '{self.config.agent_organization}' not present in "
                "scenario 'multieventsat'."
            )
        org_config = dict(self.config.agent_organization_config)
        prefixes = {
            "multieventsat": "sat",
            "ssa": "sat",
            "eventsat": "eventsat",
        }
        prefix = prefixes.get(self.config.environment.scenario)
        if prefix is not None:
            org_config.setdefault("satellite_prefix", prefix)
        org = org_cls(config=org_config)
        org.initialize(constellation_size=self.config.environment.constellation_size)
        return org

    def _create_adapters(self, agent_ids: List[str]) -> Dict[str, RLSpaceAdapter]:
        return {agent_id: self._build_adapter(agent_id=agent_id) for agent_id in agent_ids}

    def _build_adapter(self, agent_id: str | None = None) -> RLSpaceAdapter:
        adapter_cfg = dict(self.config.representation_config)
        adapter_cfg.setdefault("max_steps", self.config.max_steps)
        if agent_id is not None and self._organization is not None:
            act_ids = list(self._organization.satellites_for_agent(agent_id))
            observe_ids = list(self._organization.observed_satellites_for_agent(agent_id))
            adapter_cfg["act_ids"] = act_ids
            adapter_cfg["observe_ids"] = observe_ids
            if act_ids:
                adapter_cfg["satellite_id"] = act_ids[0]
            elif observe_ids:
                adapter_cfg["satellite_id"] = observe_ids[0]
        return make_space_adapter(
            scenario=self.config.environment.scenario,
            config=adapter_cfg,
            env=self._environment,
        )

"""RLlib MultiAgentEnv bridge for AUTOPS experiments."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv

    RLLIB_AVAILABLE = True
except ImportError:
    MultiAgentEnv = object  # type: ignore[misc,assignment]
    RLLIB_AVAILABLE = False

from src.agent_organization.base import AgentAction
from src.orchestration.config_loader import ExperimentConfig
from src.rl.space_adapters import RLSpaceAdapter, make_space_adapter


class AUTOPSRLLibMultiAgentEnv(MultiAgentEnv):  # type: ignore[misc]
    """Expose an AUTOPS experiment as an RLlib multi-agent environment.

    Even SAS is represented as a one-agent RLlib environment.  That keeps the
    training interface stable when future constellation scenarios instantiate
    several decision-making agents through ``agent_organization``.
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
        self._environment = self._create_environment()
        self._organization = self._create_organization()
        self._adapter = self._create_adapter()

        self.possible_agents: List[str] = list(self._organization.get_agents())
        self.agents: List[str] = []
        self._agent_ids = set(self.possible_agents)
        self.observation_space = self._adapter.observation_space
        self.action_space = self._adapter.action_space
        self.observation_spaces = {
            agent_id: self.observation_space for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: self.action_space for agent_id in self.possible_agents
        }
        self._last_observation: Any = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        self._organization.initialize(
            constellation_size=self.config.environment.constellation_size,
        )
        self.agents = list(self.possible_agents)
        self._last_observation = self._environment.reset(seed=seed)
        agent_obs = self._organization.distribute_observation(self._last_observation)
        observations = {
            agent_id: self._adapter.encode_observation(agent_obs.get(agent_id))
            for agent_id in self.agents
        }
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
        agent_actions: Dict[str, AgentAction] = {}
        for agent_id in active_agents:
            if agent_id not in action_dict:
                continue
            agent_actions[agent_id] = AgentAction(
                agent_id=agent_id,
                action=self._adapter.decode_action(action_dict[agent_id], agent_id=agent_id),
            )

        env_actions = self._organization.collect_actions(agent_actions)
        step_result = self._environment.step(env_actions)
        self._last_observation = step_result.observation
        done = bool(self._environment.is_done())
        reward = self._adapter.scalar_reward(step_result.rewards)

        if done:
            self.agents = []

        agent_obs = (
            {}
            if done
            else self._organization.distribute_observation(self._last_observation)
        )
        observations = {
            agent_id: self._adapter.encode_observation(agent_obs.get(agent_id))
            for agent_id in self.agents
        }
        rewards = {agent_id: reward for agent_id in active_agents}
        terminateds = {agent_id: done for agent_id in active_agents}
        truncateds = {agent_id: False for agent_id in active_agents}
        terminateds["__all__"] = done
        truncateds["__all__"] = False
        infos = {
            agent_id: {"agent_id": agent_id, **dict(step_result.info)}
            for agent_id in active_agents
        }
        return observations, rewards, terminateds, truncateds, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def space_adapter(self) -> RLSpaceAdapter:
        return self._adapter

    def _create_environment(self) -> Any:
        scenario = self.config.environment.scenario
        env_cfg = {
            "step_duration_s": self.config.environment.timestep_seconds,
            "max_steps": self.config.max_steps,
            **self.config.environment.scenario_config,
        }
        if scenario == "eventsat":
            from src.environment.scenarios.eventsat_env import EventSatEnvironment

            env_cfg["anomaly_requires_ground_pass"] = (
                self.config.operations_paradigm != "autonomous_hybrid"
            )
            return EventSatEnvironment(config=env_cfg)
        raise ValueError(f"No RLlib environment registered for scenario '{scenario}'")

    def _create_organization(self) -> Any:
        from src.agent_organization.centralized_mas import CentralizedMAS
        from src.agent_organization.decentralized_mas import DecentralizedMAS
        from src.agent_organization.hybrid_mas import HybridMAS
        from src.agent_organization.independent_mas import IndependentMAS
        from src.agent_organization.single_agent_system import SingleAgentSystem

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
        org = org_cls(config=self.config.agent_organization_config)
        org.initialize(constellation_size=self.config.environment.constellation_size)
        return org

    def _create_adapter(self) -> RLSpaceAdapter:
        adapter_cfg = dict(self.config.representation_config)
        adapter_cfg.setdefault("max_steps", self.config.max_steps)
        return make_space_adapter(
            scenario=self.config.environment.scenario,
            config=adapter_cfg,
            env=self._environment,
        )


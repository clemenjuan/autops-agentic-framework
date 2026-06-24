"""RLlib MultiAgentEnv bridge for AUTOPS experiments."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv

    RLLIB_AVAILABLE = True
except ImportError:
    MultiAgentEnv = object  # type: ignore[misc,assignment]
    RLLIB_AVAILABLE = False

from src.core.organization.base import AgentAction, validate_agent_satellite_mapping
from src.core.config_loader import ExperimentConfig
from src.rl.space_adapters import RLSpaceAdapter, make_space_adapter


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
        self._environment = self._create_environment()
        self._organization = self._create_organization()

        self.possible_agents: List[str] = list(self._organization.get_agents())
        validate_agent_satellite_mapping(
            self._organization,
            self._environment,
            self.config.environment.constellation_size,
            self.config.environment.scenario,
        )
        self._adapters: Dict[str, RLSpaceAdapter] = self._create_adapters(
            self.possible_agents
        )
        self._space_adapter: RLSpaceAdapter = (
            self._adapters[self.possible_agents[0]]
            if self.possible_agents
            else self._build_adapter(satellite_id=None)
        )

        self.agents: List[str] = []
        self._agent_ids = set(self.possible_agents)
        self.observation_space = self._space_adapter.observation_space
        self.action_space = self._space_adapter.action_space
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
            agent_id: self._adapter_for(agent_id).encode_observation(
                agent_obs.get(agent_id)
            )
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
                action=self._adapter_for(agent_id).decode_action(
                    action_dict[agent_id], agent_id=agent_id
                ),
            )

        env_actions = self._organization.collect_actions(agent_actions)
        step_result = self._environment.step(env_actions)
        self._last_observation = step_result.observation
        done = bool(self._environment.is_done())

        if done:
            self.agents = []

        agent_obs = (
            {}
            if done
            else self._organization.distribute_observation(self._last_observation)
        )
        observations = {
            agent_id: self._adapter_for(agent_id).encode_observation(
                agent_obs.get(agent_id)
            )
            for agent_id in self.agents
        }
        rewards = {
            agent_id: self._resolve_agent_reward(agent_id, step_result.rewards)
            for agent_id in active_agents
        }
        terminateds = {agent_id: done for agent_id in active_agents}
        truncateds = {agent_id: False for agent_id in active_agents}
        terminateds["__all__"] = done
        truncateds["__all__"] = False
        infos = {
            agent_id: {"agent_id": agent_id, **dict(step_result.info)}
            for agent_id in observations
        }
        return observations, rewards, terminateds, truncateds, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def space_adapter(self) -> RLSpaceAdapter:
        return self._space_adapter

    # ------------------------------------------------------------------
    # Reward resolution
    # ------------------------------------------------------------------

    def _resolve_agent_reward(
        self, agent_id: str, raw_rewards: Dict[str, float]
    ) -> float:
        """Map an environment reward dict to a single per-agent reward.

        Resolution is driven by the *structure* of ``raw_rewards``: if the
        agent's own satellite is a key, return that per-satellite value (already
        blended by the scenario reward function); otherwise fall back to
        ``scalar_reward`` (sum), reproducing legacy single-scalar behaviour.
        """
        sat_id = self._organization.satellite_for_agent(agent_id)
        if sat_id in raw_rewards:
            return float(raw_rewards[sat_id])
        return self._adapter_for(agent_id).scalar_reward(raw_rewards)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    def _adapter_for(self, agent_id: str) -> RLSpaceAdapter:
        return self._adapters.get(agent_id, self._space_adapter)

    def _create_environment(self) -> Any:
        scenario = self.config.environment.scenario
        env_cfg = {
            "step_duration_s": self.config.environment.timestep_seconds,
            "max_steps": self.config.max_steps,
            **self.config.environment.scenario_config,
        }
        if scenario == "eventsat":
            from src.eventsat.env import EventSatEnvironment

            env_cfg["anomaly_requires_ground_pass"] = (
                self.config.operations_paradigm != "autonomous_hybrid"
            )
            return EventSatEnvironment(config=env_cfg)
        if scenario == "basemultisat":
            from src.eventsat.basemultisat_env import BaseMultiSatEnvironment

            env_cfg["anomaly_requires_ground_pass"] = (
                self.config.operations_paradigm != "autonomous_hybrid"
            )
            env_cfg["constellation_size"] = self.config.environment.constellation_size
            return BaseMultiSatEnvironment(config=env_cfg)
        raise ValueError(f"No RLlib environment registered for scenario '{scenario}'")

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
        org = org_cls(config=self.config.agent_organization_config)
        org.initialize(constellation_size=self.config.environment.constellation_size)
        return org

    def _create_adapters(self, agent_ids: List[str]) -> Dict[str, RLSpaceAdapter]:
        return {
            agent_id: self._build_adapter(
                satellite_id=self._organization.satellite_for_agent(agent_id)
            )
            for agent_id in agent_ids
        }

    def _build_adapter(self, satellite_id: str | None) -> RLSpaceAdapter:
        adapter_cfg = dict(self.config.representation_config)
        adapter_cfg.setdefault("max_steps", self.config.max_steps)
        if satellite_id is not None:
            adapter_cfg["satellite_id"] = satellite_id
        return make_space_adapter(
            scenario=self.config.environment.scenario,
            config=adapter_cfg,
            env=self._environment,
        )

"""Tests for the BaseMultiSat constellation scenario and its RL integration."""
from __future__ import annotations

import numpy as np
import pytest

from src.environment.rewards import BaseMultiSatRewardFunction
from src.environment.scenarios.basemultisat_env import BaseMultiSatEnvironment


def _env_config(n: int = 3, max_steps: int = 5) -> dict:
    return {
        "constellation_size": n,
        "step_duration_s": 60,
        "max_steps": max_steps,
        "scenario_file": "configs/scenarios/basemultisat.yaml",
        "reward_config": {"local_weight": 0.7, "team_weight": 0.3, "team_reducer": "mean"},
    }


class TestBaseMultiSatRewardFunction:
    def test_pure_individual_when_team_weight_zero(self) -> None:
        rf = BaseMultiSatRewardFunction({"local_weight": 1.0, "team_weight": 0.0})
        out = rf.compute_rewards({"sat_0": 1.0, "sat_1": 3.0})
        assert out == {"sat_0": 1.0, "sat_1": 3.0}

    def test_local_weight_scales_individual(self) -> None:
        rf = BaseMultiSatRewardFunction({"local_weight": 0.5, "team_weight": 0.0})
        out = rf.compute_rewards({"sat_0": 2.0, "sat_1": 4.0})
        assert out == {"sat_0": 1.0, "sat_1": 2.0}

    def test_mixed_local_team_mean(self) -> None:
        rf = BaseMultiSatRewardFunction(
            {"local_weight": 0.5, "team_weight": 0.5, "team_reducer": "mean"}
        )
        out = rf.compute_rewards({"sat_0": 1.0, "sat_1": 3.0})  # team mean = 2.0
        assert out["sat_0"] == pytest.approx(0.5 * 1.0 + 0.5 * 2.0)
        assert out["sat_1"] == pytest.approx(0.5 * 3.0 + 0.5 * 2.0)

    def test_team_reducers(self) -> None:
        for reducer, expected_team in (("sum", 4.0), ("min", 1.0), ("mean", 2.0)):
            rf = BaseMultiSatRewardFunction(
                {"local_weight": 0.0, "team_weight": 1.0, "team_reducer": reducer}
            )
            out = rf.compute_rewards({"sat_0": 1.0, "sat_1": 3.0})
            assert out["sat_0"] == pytest.approx(expected_team)


class TestBaseMultiSatEnvironment:
    def test_observation_exposes_all_satellites(self) -> None:
        env = BaseMultiSatEnvironment(_env_config(n=3))
        obs = env.reset(seed=1)
        assert set(obs.constellation_state.satellites) == {"sat_0", "sat_1", "sat_2"}

    def test_step_returns_per_satellite_reward_dict(self) -> None:
        env = BaseMultiSatEnvironment(_env_config(n=3))
        env.reset(seed=1)
        result = env.step({sid: {"mode": "charging"} for sid in ("sat_0", "sat_1", "sat_2")})
        assert set(result.rewards) == {"sat_0", "sat_1", "sat_2"}
        assert all(isinstance(v, float) for v in result.rewards.values())

    def test_episode_terminates_at_max_steps(self) -> None:
        env = BaseMultiSatEnvironment(_env_config(n=2, max_steps=3))
        env.reset(seed=0)
        for _ in range(3):
            result = env.step({"sat_0": {"mode": "charging"}, "sat_1": {"mode": "charging"}})
        assert result.done is True

    def test_independent_launch_lottery_diverges(self) -> None:
        # Independent per-satellite launch lotteries give distinct ground-pass
        # schedules, so the satellites are genuinely decoupled even under an
        # identical action stream.
        env = BaseMultiSatEnvironment(_env_config(n=3, max_steps=200))
        env.reset(seed=7)
        diverged = False
        for _ in range(200):
            result = env.step({sid: {"mode": "communication"} for sid in ("sat_0", "sat_1", "sat_2")})
            sats = result.observation.constellation_state.satellites
            passes = [sats[sid].metadata.get("ground_pass_active") for sid in ("sat_0", "sat_1", "sat_2")]
            socs = [sats[sid].resources["battery_soc"] for sid in ("sat_0", "sat_1", "sat_2")]
            if len(set(passes)) > 1 or len(set(socs)) > 1:
                diverged = True
                break
        assert diverged, "satellites should diverge under independent launch lotteries"

    def test_reproducible_given_seed(self) -> None:
        def run() -> list:
            env = BaseMultiSatEnvironment(_env_config(n=2, max_steps=4))
            env.reset(seed=123)
            rewards = []
            for _ in range(4):
                r = env.step({"sat_0": {"mode": "charging"}, "sat_1": {"mode": "charging"}})
                rewards.append(tuple(sorted(r.rewards.items())))
            return rewards

        assert run() == run()


class TestBaseMultiSatRLLibIntegration:
    def _config(self, n: int = 3, max_steps: int = 5) -> dict:
        return {
            "experiment_id": "basemultisat_test",
            "seed": 0,
            "agent_organization": "independent_mas",
            "decision_loop": "sda",
            "representation": "subsymbolic",
            "emergence_mode": "learned",
            "operations_paradigm": "autonomous_hybrid",
            "representation_config": {"type": "subsymbolic_eventsat", "rl_mock": True},
            "emergence_config": {
                "mode": "learned",
                "mechanism": "ppo",
                "policy_sharing": {"mode": "shared_all"},
            },
            "environment": {
                "constellation_size": n,
                "timestep_seconds": 60,
                "max_steps": max_steps,
                "scenario": "basemultisat",
                "scenario_config": {
                    "scenario_file": "configs/scenarios/basemultisat.yaml",
                    "reward_config": {"local_weight": 0.7, "team_weight": 0.3, "team_reducer": "mean"},
                },
            },
            "num_episodes": 1,
            "max_steps": max_steps,
        }

    def test_bridge_exposes_one_agent_per_satellite(self) -> None:
        pytest.importorskip("gymnasium")
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": self._config(n=3)})
        assert env.possible_agents == ["sat_agent_0", "sat_agent_1", "sat_agent_2"]
        assert {a: env._organization.satellite_for_agent(a) for a in env.possible_agents} == {
            "sat_agent_0": "sat_0",
            "sat_agent_1": "sat_1",
            "sat_agent_2": "sat_2",
        }

    def test_centralized_mas_on_multisat_is_rejected_in_bridge(self) -> None:
        pytest.importorskip("gymnasium")
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        cfg = self._config(n=3)
        cfg["agent_organization"] = "centralized_mas"
        with pytest.raises(ValueError, match="not present in scenario"):
            AUTOPSRLLibMultiAgentEnv({"experiment_config": cfg})

    def test_reset_and_step_multiagent_contract(self) -> None:
        pytest.importorskip("gymnasium")
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": self._config(n=3)})
        obs, _ = env.reset(seed=0)
        assert set(obs) == {"sat_agent_0", "sat_agent_1", "sat_agent_2"}
        assert obs["sat_agent_0"].shape == (25,)
        action = {a: env.action_space.sample() for a in env.possible_agents}
        next_obs, rewards, terminateds, truncateds, infos = env.step(action)
        assert set(rewards) == {"sat_agent_0", "sat_agent_1", "sat_agent_2"}
        assert "__all__" in terminateds and "__all__" in truncateds

    def test_each_agent_gets_its_own_satellite_reward(self) -> None:
        pytest.importorskip("gymnasium")
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": self._config(n=3, max_steps=150)})
        env.reset(seed=3)
        distinct_seen = False
        for _ in range(150):
            action = {a: env.action_space.sample() for a in env.agents}
            _, rewards, terminateds, _, _ = env.step(action)
            if len(set(round(v, 9) for v in rewards.values())) > 1:
                distinct_seen = True
                break
            if terminateds["__all__"]:
                break
        assert distinct_seen, "per-agent rewards should differ across satellites"


class TestBaseMultiSatRayPolicySpecs:
    """RLlib training-wiring checks for the N-agent case.

    These exercise the parts that need the real Ray/RLlib stack (the env as a
    genuine ``MultiAgentEnv`` subclass and per-agent ``PolicySpec`` creation).
    They are skipped automatically when Ray is not installed, and run under
    ``uv run pytest`` where ``ray[rllib]`` is available.
    """

    def _config(self, n: int = 3) -> dict:
        return {
            "experiment_id": "basemultisat_ray_test",
            "seed": 0,
            "agent_organization": "independent_mas",
            "decision_loop": "sda",
            "representation": "subsymbolic",
            "emergence_mode": "learned",
            "operations_paradigm": "autonomous_hybrid",
            "representation_config": {"type": "subsymbolic_eventsat", "rl_mock": True},
            "emergence_config": {
                "mode": "learned",
                "mechanism": "ppo",
                "policy_sharing": {"mode": "shared_all"},
            },
            "environment": {
                "constellation_size": n,
                "timestep_seconds": 60,
                "max_steps": 5,
                "scenario": "basemultisat",
                "scenario_config": {
                    "scenario_file": "configs/scenarios/basemultisat.yaml",
                    "reward_config": {"local_weight": 0.7, "team_weight": 0.3, "team_reducer": "mean"},
                },
            },
            "num_episodes": 1,
            "max_steps": 5,
        }

    def test_env_is_real_rllib_multiagent_subclass(self) -> None:
        pytest.importorskip("ray")
        from ray.rllib.env.multi_agent_env import MultiAgentEnv

        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": self._config(n=3)})
        # With Ray installed the bridge subclasses the real MultiAgentEnv, not
        # the `object` fallback used when Ray is absent.
        assert isinstance(env, MultiAgentEnv)
        assert env.possible_agents == ["sat_agent_0", "sat_agent_1", "sat_agent_2"]

    def test_shared_all_builds_a_single_shared_policy_for_n_agents(self) -> None:
        pytest.importorskip("ray")
        from src.rl.policy_mapping import PolicySharingConfig, build_policy_specs
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": self._config(n=3)})
        sharing = PolicySharingConfig.from_config({"mode": "shared_all"})
        specs = build_policy_specs(
            env.possible_agents, env.observation_space, env.action_space, sharing
        )
        assert set(specs) == {"shared_policy"}
        assert specs["shared_policy"].observation_space == env.observation_space
        assert specs["shared_policy"].action_space == env.action_space

    def test_independent_per_agent_builds_one_policy_per_satellite(self) -> None:
        pytest.importorskip("ray")
        from src.rl.policy_mapping import PolicySharingConfig, build_policy_specs
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": self._config(n=3)})
        sharing = PolicySharingConfig.from_config({"mode": "independent_per_agent"})
        specs = build_policy_specs(
            env.possible_agents, env.observation_space, env.action_space, sharing
        )
        assert set(specs) == {
            "policy_sat_agent_0",
            "policy_sat_agent_1",
            "policy_sat_agent_2",
        }

    def test_policy_mapping_fn_covers_every_agent(self) -> None:
        pytest.importorskip("ray")
        from src.rl.policy_mapping import PolicySharingConfig, build_policy_specs
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": self._config(n=3)})
        sharing = PolicySharingConfig.from_config({"mode": "shared_all"})
        specs = build_policy_specs(
            env.possible_agents, env.observation_space, env.action_space, sharing
        )
        map_fn = sharing.mapping_fn()
        # Every agent must map to a policy id that exists in the spec set.
        for agent_id in env.possible_agents:
            assert map_fn(agent_id) in specs


class TestBaseMultiSatExperimentRunner:
    """End-to-end evaluation of the multi-satellite scenario via ExperimentRunner.

    Uses ``rl_mock`` (RandomPolicy) so it runs without Ray/torch: it exercises
    the full ``autops run`` evaluation path — per-agent satellite-aware
    representations, the multi-agent step loop, and the (reused) EventSat
    metrics collector consuming basemultisat's aggregated info.
    """

    def _config(self, n: int = 3, episodes: int = 2, steps: int = 15) -> dict:
        return {
            "experiment_id": "basemultisat_eval_test",
            "seed": 1,
            "agent_organization": "independent_mas",
            "decision_loop": "sda",
            "representation": "subsymbolic",
            "emergence_mode": "learned",
            "operations_paradigm": "autonomous_hybrid",
            "representation_config": {"type": "subsymbolic_eventsat", "rl_mock": True},
            "emergence_config": {
                "mode": "learned",
                "mechanism": "ppo",
                "policy_sharing": {"mode": "shared_all"},
            },
            "environment": {
                "constellation_size": n,
                "timestep_seconds": 60,
                "max_steps": steps,
                "scenario": "basemultisat",
                "scenario_config": {
                    "scenario_file": "configs/scenarios/basemultisat.yaml",
                    "reward_config": {"local_weight": 0.7, "team_weight": 0.3},
                },
            },
            "num_episodes": episodes,
            "max_steps": steps,
            "output_dir": "data/results/basemultisat_eval_test",
        }

    def _run(self, **kwargs):
        from src.orchestration.config_loader import ExperimentConfig
        from src.orchestration.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(**self._config(**kwargs))
        runner = ExperimentRunner(config=cfg)
        return runner, runner.run()

    def test_runner_completes_episodes(self) -> None:
        _, res = self._run(episodes=2, steps=12)
        assert res["num_episodes"] == 2
        assert len(res["episodes"]) == 2
        assert all(ep["num_steps"] == 12 for ep in res["episodes"])

    def test_per_agent_representations_bound_to_own_satellite(self) -> None:
        runner, _ = self._run(n=3, episodes=1, steps=5)
        assert set(runner._decision_loops) == {"sat_agent_0", "sat_agent_1", "sat_agent_2"}
        mapping = {a: r._satellite_id for a, r in runner._representations.items()}
        assert mapping == {
            "sat_agent_0": "sat_0",
            "sat_agent_1": "sat_1",
            "sat_agent_2": "sat_2",
        }

    def test_metrics_collected_for_constellation(self) -> None:
        _, res = self._run(n=3, episodes=1, steps=12)
        episode_metrics = res["episodes"][0].get("episode_metrics")
        assert episode_metrics is not None

    def test_centralized_mas_on_multisat_is_rejected_in_runner(self) -> None:
        from src.orchestration.config_loader import ExperimentConfig
        from src.orchestration.experiment_runner import ExperimentRunner

        cfg = self._config(n=3)
        cfg["agent_organization"] = "centralized_mas"
        runner = ExperimentRunner(config=ExperimentConfig(**cfg))
        with pytest.raises(ValueError, match="not present in scenario"):
            runner.run()

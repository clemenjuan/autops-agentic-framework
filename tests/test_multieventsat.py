"""Tests for the MultiEventsat constellation scenario and its RL integration."""
from __future__ import annotations

import numpy as np
import pytest

from src.eventsat.rewards import MultiEventsatRewardFunction
from src.eventsat.multieventsat_env import MultiEventsatEnv


def _env_config(n: int = 3, max_steps: int = 5) -> dict:
    return {
        "constellation_size": n,
        "step_duration_s": 60,
        "max_steps": max_steps,
        "scenario_file": "configs/scenarios/multieventsat.yaml",
        "reward_config": {"local_weight": 0.7, "team_weight": 0.3, "team_reducer": "mean"},
    }


class TestMultiEventsatRewardFunction:
    def test_pure_individual_when_team_weight_zero(self) -> None:
        rf = MultiEventsatRewardFunction({"local_weight": 1.0, "team_weight": 0.0})
        out = rf.compute_rewards({"sat_0": 1.0, "sat_1": 3.0})
        assert out == {"sat_0": 1.0, "sat_1": 3.0}

    def test_local_weight_scales_individual(self) -> None:
        rf = MultiEventsatRewardFunction({"local_weight": 0.5, "team_weight": 0.0})
        out = rf.compute_rewards({"sat_0": 2.0, "sat_1": 4.0})
        assert out == {"sat_0": 1.0, "sat_1": 2.0}

    def test_mixed_local_team_mean(self) -> None:
        rf = MultiEventsatRewardFunction(
            {"local_weight": 0.5, "team_weight": 0.5, "team_reducer": "mean"}
        )
        out = rf.compute_rewards({"sat_0": 1.0, "sat_1": 3.0})  # team mean = 2.0
        assert out["sat_0"] == pytest.approx(0.5 * 1.0 + 0.5 * 2.0)
        assert out["sat_1"] == pytest.approx(0.5 * 3.0 + 0.5 * 2.0)

    def test_team_reducers(self) -> None:
        for reducer, expected_team in (("sum", 4.0), ("min", 1.0), ("mean", 2.0)):
            rf = MultiEventsatRewardFunction(
                {"local_weight": 0.0, "team_weight": 1.0, "team_reducer": reducer}
            )
            out = rf.compute_rewards({"sat_0": 1.0, "sat_1": 3.0})
            assert out["sat_0"] == pytest.approx(expected_team)


class TestMultiEventsatEnv:
    def test_observation_exposes_all_satellites(self) -> None:
        env = MultiEventsatEnv(_env_config(n=3))
        obs = env.reset(seed=1)
        assert set(obs.constellation_state.satellites) == {"sat_0", "sat_1", "sat_2"}

    def test_aggregation_stamps_task_and_event_source_satellite(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = MultiEventsatEnv(_env_config(n=2))
        env.reset(seed=1)
        for sat_id, sub in env._subenvs.items():
            sub_obs = sub.get_observation()
            sub_obs.tasks = [
                {"task": f"{sat_id}-unaddressed"},
                {"satellite_id": "eventsat_0", "task": f"{sat_id}-addressed"},
            ]
            sub_obs.events = [
                {"event": f"{sat_id}-unaddressed"},
                {"satellite_id": "eventsat_0", "event": f"{sat_id}-addressed"},
            ]
            monkeypatch.setattr(sub, "get_observation", lambda obs=sub_obs: obs)

        obs = env.get_observation()

        assert {task["satellite_id"] for task in obs.tasks} == {"sat_0", "sat_1"}
        assert {event["satellite_id"] for event in obs.events} == {"sat_0", "sat_1"}

    def test_step_returns_per_satellite_reward_dict(self) -> None:
        env = MultiEventsatEnv(_env_config(n=3))
        env.reset(seed=1)
        result = env.step({sid: {"mode": "charging"} for sid in ("sat_0", "sat_1", "sat_2")})
        assert set(result.rewards) == {"sat_0", "sat_1", "sat_2"}
        assert all(isinstance(v, float) for v in result.rewards.values())

    def test_step_rejects_unknown_satellite_action_keys(self) -> None:
        env = MultiEventsatEnv(_env_config(n=2))
        env.reset(seed=1)

        with pytest.raises(ValueError, match="Unknown satellite action keys.*eventsat_0"):
            env.step({"eventsat_0": {"mode": "charging"}})

    def test_episode_terminates_at_max_steps(self) -> None:
        env = MultiEventsatEnv(_env_config(n=2, max_steps=3))
        env.reset(seed=0)
        for _ in range(3):
            result = env.step({"sat_0": {"mode": "charging"}, "sat_1": {"mode": "charging"}})
        assert result.done is True

    def test_independent_launch_lottery_diverges(self) -> None:
        # Independent per-satellite launch lotteries give distinct ground-pass
        # schedules, so the satellites are genuinely decoupled even under an
        # identical action stream.
        env = MultiEventsatEnv(_env_config(n=3, max_steps=200))
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
            env = MultiEventsatEnv(_env_config(n=2, max_steps=4))
            env.reset(seed=123)
            rewards = []
            for _ in range(4):
                r = env.step({"sat_0": {"mode": "charging"}, "sat_1": {"mode": "charging"}})
                rewards.append(tuple(sorted(r.rewards.items())))
            return rewards

        assert run() == run()


    def test_metrics_normalize_constellation_downlink_targets(self) -> None:
        from src.eventsat.metrics import EventSatMetricsCollector

        env = MultiEventsatEnv(_env_config(n=3, max_steps=1))
        per_satellite = {
            f"sat_{idx}": {
                "battery_soc": 0.89,
                "prev_battery_soc": 0.90,
                "data_downlinked_mb": 10.0,
                "step_downlinked_mb": 10.0,
                "observation_hours": 0.0,
                "max_achievable_downlink_mb": 20.0,
                "gross_energy_consumed_wh": 0.1,
                "solar_generation_wh": 0.0,
                "net_battery_depletion_wh": 0.1,
            }
            for idx in range(3)
        }
        aggregate_info = env._aggregate_info(per_satellite)

        base_cfg = {
            "max_steps": 1,
            "step_duration_s": 86400.0,
            "battery_capacity_wh": 100.0,
            "utility_weights": {"observation": 0.0, "downlink": 1.0, "anomaly_penalty": 0.0},
            "utility_targets": {
                "observation_hours": 1.0,
                "downlinked_mb": 10.0,
                "mission_duration_days": 1.0,
            },
        }
        constellation = EventSatMetricsCollector(
            config={**base_cfg, "constellation_size": 3},
            constellation_size=3,
        )

        constellation_episode = constellation.aggregate_episode_metrics([
            constellation.collect_step_metrics(
                0, 0.0, None, {}, {},
                {"per_satellite": per_satellite, **aggregate_info},
                {"inference_allowed": True}
            )
        ])

        assert aggregate_info["battery_soc"] == pytest.approx(0.89)
        assert aggregate_info["battery_soc_delta_sum"] == pytest.approx(0.03)
        assert constellation_episode.aggregated["utility"] == pytest.approx(1.0)
        assert constellation_episode.aggregated["physical_utility_ceiling"] == pytest.approx(2.0)
        assert constellation_episode.aggregated["total_energy_consumed_wh"] == pytest.approx(0.3)

    def test_heterogeneous_n3_metrics_preserve_satellite_identity(self) -> None:
        from src.eventsat.metrics import EventSatMetricsCollector

        env = MultiEventsatEnv(_env_config(n=3, max_steps=2))
        collector = EventSatMetricsCollector(
            config={
                "max_steps": 2,
                "step_duration_s": 60.0,
                "constellation_size": 3,
                "utility_weights": {
                    "observation": 0.0, "downlink": 0.0, "anomaly_penalty": 0.0
                },
            },
            constellation_size=3,
        )
        steps = []
        for timestep in range(2):
            per_satellite = {
                # One rejected operational request every step.
                "sat_0": {
                    "requested_mode": "payload_observe",
                    "resolved_mode": "charging",
                    "forced": True,
                    "safety_safe": 0.0,
                    "step_downlinked_mb": 0.0,
                },
                # One protective safety override every step.
                "sat_1": {
                    "requested_mode": "payload_observe",
                    "resolved_mode": "safe",
                    "forced": True,
                    "safety_safe": 1.0,
                    "step_downlinked_mb": 0.0,
                },
                # A clean satellite downlinks only on the first step.
                "sat_2": {
                    "requested_mode": "communication",
                    "resolved_mode": "communication",
                    "forced": False,
                    "safety_safe": 0.0,
                    "step_downlinked_mb": 1.0 if timestep == 0 else 0.0,
                },
            }
            info = {"per_satellite": per_satellite, **env._aggregate_info(per_satellite)}
            steps.append(collector.collect_step_metrics(
                timestep, 0.0, None, {}, {}, info, {"inference_allowed": True}
            ))

        episode = collector.aggregate_episode_metrics(steps)
        aggregated = episode.aggregated

        assert aggregated["operator_load"] == pytest.approx(1.0 / 3.0)
        assert aggregated["constraint_violation_rate"] == pytest.approx(1.0 / 3.0)
        assert aggregated["safety_overrides"] == pytest.approx(2.0)
        assert aggregated["constraint_violations"] == pytest.approx(2.0)
        assert aggregated["mean_aoi_s"] == pytest.approx(70.0)
        assert aggregated["peak_aoi_s"] == pytest.approx(120.0)
        assert aggregated["command_count"] == pytest.approx(3.0)
        assert aggregated["commanding_effort"] == pytest.approx(2160.0)

    def test_recovery_is_tracked_independently_per_satellite(self) -> None:
        from src.eventsat.metrics import EventSatMetricsCollector

        env = MultiEventsatEnv(_env_config(n=3, max_steps=3))
        collector = EventSatMetricsCollector(
            {"max_steps": 3, "step_duration_s": 60.0, "constellation_size": 3},
            constellation_size=3,
        )
        steps = []
        for timestep in range(3):
            per_satellite = {}
            for sat_id, recovery_step in (("sat_0", 1), ("sat_1", 2)):
                active = timestep < recovery_step
                per_satellite[sat_id] = {
                    "requested_mode": "safe" if active else "charging",
                    "resolved_mode": "safe" if active else "charging",
                    "safety_safe": float(active),
                    "anomaly": "thermal_warning" if timestep == 0 else None,
                    "anomaly_forced_safe": float(active),
                }
            per_satellite["sat_2"] = {
                "requested_mode": "charging",
                "resolved_mode": "charging",
                "safety_safe": 0.0,
            }
            info = {"per_satellite": per_satellite, **env._aggregate_info(per_satellite)}
            steps.append(collector.collect_step_metrics(
                timestep, 0.0, None, {}, {}, info, {"inference_allowed": True}
            ))

        aggregated = collector.aggregate_episode_metrics(steps).aggregated
        assert aggregated["anomaly_events"] == pytest.approx(2.0)
        assert aggregated["robustness_mean_recovery_steps"] == pytest.approx(1.5)
        assert aggregated["unrecovered_anomaly_events"] == pytest.approx(0.0)

    def test_gross_energy_and_net_depletion_are_distinct_for_n3(self) -> None:
        from src.eventsat.metrics import EventSatMetricsCollector

        env = MultiEventsatEnv(_env_config(n=3, max_steps=1))
        env.reset(seed=7)
        for idx, sub in enumerate(env._subenvs.values()):
            sub.battery_capacity_wh = 100.0
            sub.battery_soc = 0.8
            sub.consumption = {"charging": {"sun_w": 6.0, "eclipse_w": 6.0}}
            sub.solar_generation_w = 6.0
            in_sunlight = idx == 0
            sub._orbital_ctx.is_in_sunlight = lambda step, value=in_sunlight: value

        result = env.step({sat_id: {"mode": "charging"} for sat_id in env._sat_ids})
        assert result.info["gross_energy_consumed_wh"] == pytest.approx(0.300)
        assert result.info["solar_generation_wh"] == pytest.approx(0.100)
        assert result.info["net_battery_depletion_wh"] == pytest.approx(0.200)

        collector = EventSatMetricsCollector(
            {"max_steps": 1, "step_duration_s": 60.0, "constellation_size": 3},
            constellation_size=3,
        )
        step = collector.collect_step_metrics(
            0, 0.0, result.observation, {}, result.rewards, result.info, {}
        )
        aggregated = collector.aggregate_episode_metrics([step]).aggregated
        assert aggregated["total_energy_consumed_wh"] == pytest.approx(0.300)
        assert aggregated["total_solar_generation_wh"] == pytest.approx(0.100)
        assert aggregated["net_battery_depletion_wh"] == pytest.approx(0.200)


class TestMultiEventsatRLLibIntegration:
    def _config(self, n: int = 3, max_steps: int = 5) -> dict:
        return {
            "experiment_id": "multieventsat_test",
            "seed": 0,
            "agent_organization": "independent_mas",
            "decision_procedure": "sda",
            "representation": "subsymbolic",
            "behaviour": "emergent",
            "operations_paradigm": "autonomous_onboard",
            "representation_config": {"type": "subsymbolic_eventsat", "rl_mock": True},
            "behaviour_config": {
                "mode": "emergent",
                "mechanism": "ppo",
                "policy_sharing": {"mode": "shared_all"},
            },
            "environment": {
                "constellation_size": n,
                "timestep_seconds": 60,
                "max_steps": max_steps,
                "scenario": "multieventsat",
                "scenario_config": {
                    "scenario_file": "configs/scenarios/multieventsat.yaml",
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
        with pytest.raises(ValueError, match="centralized_mas.*not implemented"):
            AUTOPSRLLibMultiAgentEnv({"experiment_config": cfg})

    def test_reset_and_step_multiagent_contract(self) -> None:
        pytest.importorskip("gymnasium")
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": self._config(n=3)})
        obs, _ = env.reset(seed=0)
        assert set(obs) == {"sat_agent_0", "sat_agent_1", "sat_agent_2"}
        assert obs["sat_agent_0"].shape == (33,)
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


class TestMultiEventsatRayPolicySpecs:
    """RLlib training-wiring checks for the N-agent case.

    These exercise the parts that need the real Ray/RLlib stack (the env as a
    genuine ``MultiAgentEnv`` subclass and per-agent ``PolicySpec`` creation).
    They are skipped automatically when Ray is not installed, and run under
    ``uv run pytest`` where ``ray[rllib]`` is available.
    """

    def _config(self, n: int = 3) -> dict:
        return {
            "experiment_id": "multieventsat_ray_test",
            "seed": 0,
            "agent_organization": "independent_mas",
            "decision_procedure": "sda",
            "representation": "subsymbolic",
            "behaviour": "emergent",
            "operations_paradigm": "autonomous_onboard",
            "representation_config": {"type": "subsymbolic_eventsat", "rl_mock": True},
            "behaviour_config": {
                "mode": "emergent",
                "mechanism": "ppo",
                "policy_sharing": {"mode": "shared_all"},
            },
            "environment": {
                "constellation_size": n,
                "timestep_seconds": 60,
                "max_steps": 5,
                "scenario": "multieventsat",
                "scenario_config": {
                    "scenario_file": "configs/scenarios/multieventsat.yaml",
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


class TestMultiEventsatExperimentRunner:
    """End-to-end evaluation of the multi-satellite scenario via ExperimentRunner.

    Uses ``rl_mock`` (RandomPolicy) so it runs without Ray/torch: it exercises
    the full ``autops run`` evaluation path — per-agent satellite-aware
    representations, the multi-agent step loop, and the (reused) EventSat
    metrics collector consuming multieventsat's aggregated info.
    """

    def _config(self, n: int = 3, episodes: int = 2, steps: int = 15) -> dict:
        return {
            "experiment_id": "multieventsat_eval_test",
            "seed": 1,
            "agent_organization": "independent_mas",
            "decision_procedure": "sda",
            "representation": "subsymbolic",
            "behaviour": "emergent",
            "operations_paradigm": "autonomous_onboard",
            "representation_config": {"type": "subsymbolic_eventsat", "rl_mock": True},
            "behaviour_config": {
                "mode": "emergent",
                "mechanism": "ppo",
                "policy_sharing": {"mode": "shared_all"},
            },
            "environment": {
                "constellation_size": n,
                "timestep_seconds": 60,
                "max_steps": steps,
                "scenario": "multieventsat",
                "scenario_config": {
                    "scenario_file": "configs/scenarios/multieventsat.yaml",
                    "reward_config": {"local_weight": 0.7, "team_weight": 0.3},
                },
            },
            "num_episodes": episodes,
            "max_steps": steps,
            "output_dir": "data/results/multieventsat_eval_test",
        }

    def _run(self, **kwargs):
        from src.core.config_loader import ExperimentConfig
        from src.core.experiment_runner import ExperimentRunner

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
        from src.core.config_loader import ExperimentConfig
        from src.core.experiment_runner import ExperimentRunner

        cfg = self._config(n=3)
        cfg["agent_organization"] = "centralized_mas"
        runner = ExperimentRunner(config=ExperimentConfig(**cfg))
        with pytest.raises(ValueError, match="centralized_mas.*not implemented"):
            runner.run()

    def test_autonomous_hybrid_on_multieventsat_fails_fast_in_runner(self) -> None:
        from src.core.config_loader import ExperimentConfig
        from src.core.experiment_runner import ExperimentRunner

        cfg = self._config(n=3)
        cfg["operations_paradigm"] = "autonomous_hybrid"
        runner = ExperimentRunner(config=ExperimentConfig(**cfg))
        with pytest.raises(ValueError, match="autonomous_hybrid.*only.*eventsat"):
            runner.run()

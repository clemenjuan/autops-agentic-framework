"""Tests for the RLlib PPO backend bridge."""
from __future__ import annotations

import pytest


def _minimal_config(max_steps: int = 3) -> dict:
    return {
        "experiment_id": "rllib_backend_test",
        "seed": 0,
        "agent_organization": "sas",
        "decision_loop": "sda",
        "representation": "subsymbolic",
        "emergence_mode": "learned",
        "operations_paradigm": "autonomous_hybrid",
        "representation_config": {
            "type": "subsymbolic_eventsat",
            "rl_mock": True,
            "deterministic": False,
        },
        "emergence_config": {
            "mode": "learned",
            "mechanism": "ppo",
            "policy_sharing": {"mode": "shared_all"},
        },
        "environment": {
            "constellation_size": 1,
            "timestep_seconds": 60,
            "max_steps": max_steps,
            "scenario": "eventsat",
            "scenario_config": {
                "scenario_params": {
                    "orbit": {"orbital_period_s": 5676, "eclipse_fraction": 0.36},
                    "power": {
                        "solar_panels": {"generation_peak_w": 24.0},
                        "battery": {"capacity_wh": 84.0, "initial_soc": 0.8, "min_soc": 0.2},
                        "consumption": {},
                    },
                    "storage": {},
                    "communications": {"sband": {"downlink_rate_kbps": 128}},
                    "modes": {},
                    "payload": {},
                }
            },
        },
        "num_episodes": 1,
        "max_steps": max_steps,
        "output_dir": "data/results/rllib_backend_test",
    }


class TestPolicySharing:
    def test_shared_all_maps_every_agent_to_one_policy(self) -> None:
        from src.rl.policy_mapping import PolicySharingConfig

        sharing = PolicySharingConfig.from_config({"mode": "shared_all"})
        assert sharing.policy_id_for("central_agent") == "shared_policy"
        assert sharing.policy_id_for("sat_agent_0") == "shared_policy"

    def test_shared_by_role_maps_manager_and_satellite(self) -> None:
        from src.rl.policy_mapping import PolicySharingConfig

        sharing = PolicySharingConfig.from_config({"mode": "shared_by_role"})
        assert sharing.policy_id_for("mission_manager") == "manager_policy"
        assert sharing.policy_id_for("sat_agent_0") == "satellite_policy"

    def test_independent_per_agent_uses_agent_id(self) -> None:
        from src.rl.policy_mapping import PolicySharingConfig

        sharing = PolicySharingConfig.from_config({"mode": "independent_per_agent"})
        assert sharing.policy_id_for("sat_agent_2") == "policy_sat_agent_2"


class TestRLLibEnv:
    def test_sas_env_exposes_one_agent_multiagent_api(self) -> None:
        pytest.importorskip("gymnasium")
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": _minimal_config()})
        obs, infos = env.reset(seed=0)

        assert env.possible_agents == ["central_agent"]
        assert list(obs) == ["central_agent"]
        assert obs["central_agent"].shape == (25,)
        assert infos["central_agent"]["agent_id"] == "central_agent"

    def test_sas_env_step_returns_rllib_multiagent_contract(self) -> None:
        pytest.importorskip("gymnasium")
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": _minimal_config()})
        obs, _ = env.reset(seed=0)
        action = {"central_agent": env.action_space.sample()}
        next_obs, rewards, terminateds, truncateds, infos = env.step(action)

        assert "central_agent" in rewards
        assert "__all__" in terminateds
        assert "__all__" in truncateds
        assert "central_agent" in infos
        if not terminateds["__all__"]:
            assert "central_agent" in next_obs

    def test_terminal_step_infos_match_returned_observations(self) -> None:
        pytest.importorskip("gymnasium")
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": _minimal_config(max_steps=1)})
        env.reset(seed=0)
        action = {"central_agent": env.action_space.sample()}
        next_obs, rewards, terminateds, truncateds, infos = env.step(action)

        assert terminateds["__all__"] is True
        assert rewards.keys() == {"central_agent"}
        assert next_obs == {}
        assert infos == {}
        assert set(infos).issubset(next_obs)


class TestRLLibTrainerImport:
    def test_trainer_can_be_constructed_without_importing_ray(self, tmp_path) -> None:
        from src.emergence.rllib_training_pipeline import RLLibPPOTrainer

        trainer = RLLibPPOTrainer(
            _minimal_config(max_steps=2),
            timesteps=1,
            checkpoint_dir=tmp_path,
        )
        assert trainer.config.experiment_id == "rllib_backend_test"

    def test_default_model_architecture_uses_autops_actor_critic(self, tmp_path) -> None:
        pytest.importorskip("ray")
        from ray.rllib.algorithms.ppo import PPOConfig

        from src.emergence.rllib_training_pipeline import RLLibPPOTrainer

        trainer = RLLibPPOTrainer(_minimal_config(max_steps=2), timesteps=1, checkpoint_dir=tmp_path)
        rllib_config = trainer._configure_model(PPOConfig())

        assert rllib_config.model["custom_model"] == "autops_actor_critic_v1"
        assert rllib_config.model["custom_model_config"]["hidden_size"] == 256
        assert rllib_config.model["custom_model_config"]["action_dims"] == [7, 2, 2]

    def test_unknown_model_architecture_raises(self, tmp_path) -> None:
        pytest.importorskip("ray")
        from ray.rllib.algorithms.ppo import PPOConfig

        from src.emergence.rllib_training_pipeline import RLLibPPOTrainer

        config = _minimal_config(max_steps=2)
        config["emergence_config"]["model_architecture"] = "other_model"
        trainer = RLLibPPOTrainer(config, timesteps=1, checkpoint_dir=tmp_path)

        with pytest.raises(ValueError, match="model_architecture"):
            trainer._configure_model(PPOConfig())

    def test_episode_reward_mean_reads_env_runner_metric(self, tmp_path) -> None:
        from src.emergence.rllib_training_pipeline import RLLibPPOTrainer

        trainer = RLLibPPOTrainer(_minimal_config(max_steps=2), timesteps=1, checkpoint_dir=tmp_path)

        assert trainer._episode_reward_mean({"episode_reward_mean": 1.25}) == 1.25
        assert (
            trainer._episode_reward_mean(
                {"env_runners": {"episode_reward_mean": -0.5}}
            )
            == -0.5
        )
        assert (
            trainer._episode_reward_mean(
                {"env_runners": {"episode_return_mean": -0.75}}
            )
            == -0.75
        )
        assert trainer._episode_reward_mean({}) is None


class TestAUTOPSActorCriticModel:
    def test_forward_outputs_multidiscrete_logits_and_value(self) -> None:
        pytest.importorskip("ray")
        torch = pytest.importorskip("torch")
        spaces = pytest.importorskip("gymnasium.spaces")

        from src.rl.models.autops_actor_critic import AUTOPSActorCriticModel

        model = AUTOPSActorCriticModel(
            obs_space=spaces.Box(low=-1.0, high=2.0, shape=(25,)),
            action_space=spaces.MultiDiscrete([7, 2, 2]),
            num_outputs=11,
            model_config={"custom_model_config": {}},
            name="test_autops_actor_critic",
        )
        logits, state = model.forward(
            {"obs": torch.zeros((4, 25), dtype=torch.float32)},
            [],
            None,
        )

        assert state == []
        assert tuple(logits.shape) == (4, 11)
        assert tuple(model.value_function().shape) == (4,)
        assert len(model.actor_heads) == 3

"""Tests for the RLlib PPO backend bridge."""

from __future__ import annotations

import numpy as np
import pytest


def _minimal_config(max_steps: int = 3) -> dict:
    return {
        "experiment_id": "rllib_backend_test",
        "seed": 0,
        "agent_organization": "sas",
        "decision_procedure": "sda",
        "representation": "subsymbolic",
        "behaviour": "emergent",
        "operations_paradigm": "autonomous_onboard",
        "representation_config": {
            "type": "subsymbolic_eventsat",
            "rl_mock": True,
            "deterministic": False,
        },
        "behaviour_config": {
            "mode": "emergent",
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

    def test_shared_policy_rejects_incompatible_spaces(self) -> None:
        pytest.importorskip("ray")
        spaces = pytest.importorskip("gymnasium.spaces")
        from src.rl.policy_mapping import PolicySharingConfig, build_policy_specs

        sharing = PolicySharingConfig.from_config({"mode": "shared_all"})
        with pytest.raises(ValueError, match="cannot share"):
            build_policy_specs(
                ["cluster_agent_0", "cluster_agent_1"],
                {
                    "cluster_agent_0": spaces.Box(0.0, 1.0, shape=(87,)),
                    "cluster_agent_1": spaces.Box(0.0, 1.0, shape=(58,)),
                },
                {
                    "cluster_agent_0": spaces.MultiDiscrete([8, 2, 2] * 3),
                    "cluster_agent_1": spaces.MultiDiscrete([8, 2, 2] * 2),
                },
                sharing,
            )


class TestRLLibEnv:
    def test_sas_env_exposes_one_agent_multiagent_api(self) -> None:
        pytest.importorskip("gymnasium")
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": _minimal_config()})
        obs, infos = env.reset(seed=0)

        assert env.possible_agents == ["central_agent"]
        assert list(obs) == ["central_agent"]
        assert obs["central_agent"].shape == (33,)
        assert list(env.action_space.nvec) == [7]
        assert list(env.action_spaces["central_agent"].nvec) == [7]
        assert env._space_adapter.decode_action([0]) == {"eventsat_0": {"mode": "charging"}}
        assert infos["central_agent"]["agent_id"] == "central_agent"

    def test_onboard_capabilities_match_runner_semantics(self) -> None:
        pytest.importorskip("gymnasium")
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": _minimal_config()})

        assert env._environment.anomaly_requires_ground_pass is False
        assert env._environment.onboard_compute_active is True

    def test_ssa_dmas_reset_binds_topology_before_local_encoding(self) -> None:
        pytest.importorskip("gymnasium")
        from src.core.config_loader import apply_overrides, load_config
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        config = apply_overrides(
            load_config("configs/experiments/ssa_dmas_ao_rl_n3.yaml"),
            episodes=1,
            steps=2,
        )
        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": config.model_dump()})
        observations, _ = env.reset(seed=0)

        assert set(observations) == {
            "sat_agent_0",
            "sat_agent_1",
            "sat_agent_2",
        }
        assert all(vector.shape == (30,) for vector in observations.values())
        expected_links = {
            (src, dst)
            for src in ("sat_0", "sat_1", "sat_2")
            for dst in ("sat_0", "sat_1", "sat_2")
            if src != dst
        }
        assert env._environment._authorized_communication_links == expected_links
        local_views = env._organization.distribute_observation(env._last_observation)
        assert all(
            len(view.local_state["full_observation"].constellation_state.satellites) == 1
            for view in local_views.values()
        )
        assert all(
            view.local_state["full_observation"].constellation_state.global_info == {}
            for view in local_views.values()
        )

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

    def test_event_sat_training_grounding_matches_evaluation(self) -> None:
        """A raw PPO action must execute identically in train and evaluation."""
        pytest.importorskip("gymnasium")
        from src.core.decision_procedure.context import DecisionContext
        from src.eventsat.rl import SubsymbolicEventSat
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        train_env = AUTOPSRLLibMultiAgentEnv({"experiment_config": _minimal_config()})
        train_obs, _ = train_env.reset(seed=0)
        train_state = train_env._last_observation
        satellite = train_state.constellation_state.satellites["eventsat_0"]
        assert not satellite.metadata["contact_window_active"]

        representation = SubsymbolicEventSat(
            {
                "rl_mock": True,
                "deterministic": True,
                "act_ids": ["eventsat_0"],
                "observe_ids": ["eventsat_0"],
            }
        )
        representation._policy.get_action = lambda obs, **kwargs: (np.asarray([1]), 0.0, 0.0)
        evaluation_action = representation.select_action(
            DecisionContext(
                state=representation.encode_observation(train_state),
                loop_type="sda",
                memory=None,
            )
        )
        assert evaluation_action == {"eventsat_0": {"mode": "communication"}}

        _, rewards, _, _, infos = train_env.step({"central_agent": np.asarray([1])})
        assert train_obs["central_agent"].shape == (33,)
        assert infos["central_agent"]["requested_mode"] == "communication"
        assert infos["central_agent"]["resolved_mode"] == "communication"
        assert infos["central_agent"]["communication_failure"] == "no_contact"
        assert infos["central_agent"]["step_downlinked_mb"] == 0.0
        assert infos["central_agent"]["constraint_violation"] is False
        expected = -train_env._environment.reward_fn.failed_action_penalty
        expected *= train_env._environment.reward_fn.reward_scale
        assert rewards["central_agent"] == pytest.approx(expected)

    def test_pipeline_shaping_uses_ppo_discount_factor(self) -> None:
        pytest.importorskip("gymnasium")
        from src.core.config_loader import ExperimentConfig
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        # Independent of diagnostic configs: only the gamma wiring is checked.
        raw = _minimal_config()
        raw["behaviour_config"]["gamma"] = 0.97
        raw["environment"]["scenario_config"]["reward_config"] = {
            "pipeline_shaping": {"enabled": True}
        }
        config = ExperimentConfig(**raw)
        env = AUTOPSRLLibMultiAgentEnv({"experiment_config": config.model_dump()})

        assert env._environment.reward_fn.pipeline_shaping_enabled is True
        assert env._environment.reward_fn.discount_factor == pytest.approx(0.97)
        env.close()

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

    def test_ground_only_paradigm_fails_fast(self) -> None:
        pytest.importorskip("gymnasium")
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        config = _minimal_config()
        config["operations_paradigm"] = "autonomous_ground"
        config["representation_config"]["type"] = "subsymbolic_scheduler_eventsat"

        with pytest.raises(ValueError, match="ground-only paradigms"):
            AUTOPSRLLibMultiAgentEnv({"experiment_config": config})

    def test_hybrid_placeholder_ground_planner_fails_fast(self) -> None:
        pytest.importorskip("gymnasium")
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        config = _minimal_config()
        config["operations_paradigm"] = "autonomous_hybrid"
        # Let the paradigm resolve its matching ground scheduler. The explicit
        # AO per-step override would otherwise override both AH slots by design.
        config["representation_config"].pop("type")

        with pytest.raises(ValueError, match="requires a real ground planner"):
            AUTOPSRLLibMultiAgentEnv({"experiment_config": config})

    def test_cmas_multisat_fails_fast(self) -> None:
        pytest.importorskip("gymnasium")
        from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

        config = _minimal_config()
        config["agent_organization"] = "centralized_mas"
        config["environment"]["scenario"] = "ssa"
        config["environment"]["constellation_size"] = 3

        with pytest.raises(ValueError, match="centralized_mas.*not implemented"):
            AUTOPSRLLibMultiAgentEnv({"experiment_config": config})


class TestRLLibTrainerImport:
    def test_trainer_can_be_constructed_without_importing_ray(self, tmp_path) -> None:
        from src.core.behaviour.rllib_training_pipeline import RLLibPPOTrainer

        trainer = RLLibPPOTrainer(
            _minimal_config(max_steps=2),
            timesteps=1,
            checkpoint_dir=tmp_path,
        )
        assert trainer.config.experiment_id == "rllib_backend_test"

    def test_default_model_architecture_uses_autops_actor_critic(self, tmp_path) -> None:
        pytest.importorskip("ray")
        from ray.rllib.algorithms.ppo import PPOConfig

        from src.core.behaviour.rllib_training_pipeline import RLLibPPOTrainer

        trainer = RLLibPPOTrainer(
            _minimal_config(max_steps=2), timesteps=1, checkpoint_dir=tmp_path
        )
        rllib_config = trainer._configure_model(PPOConfig())

        assert rllib_config.model["custom_model"] == "autops_actor_critic_v1"
        assert rllib_config.model["custom_model_config"]["hidden_size"] == 256
        assert "action_dims" not in rllib_config.model["custom_model_config"]

    def test_unknown_model_architecture_raises(self, tmp_path) -> None:
        pytest.importorskip("ray")
        from ray.rllib.algorithms.ppo import PPOConfig

        from src.core.behaviour.rllib_training_pipeline import RLLibPPOTrainer

        config = _minimal_config(max_steps=2)
        config["behaviour_config"]["model_architecture"] = "other_model"
        trainer = RLLibPPOTrainer(config, timesteps=1, checkpoint_dir=tmp_path)

        with pytest.raises(ValueError, match="model_architecture"):
            trainer._configure_model(PPOConfig())

    def test_ssa_manifest_records_local_observation_schema(self, tmp_path) -> None:
        import json

        from src.core.behaviour.rllib_training_pipeline import RLLibPPOTrainer
        from src.core.config_loader import load_config
        from src.rl.policy_mapping import PolicySharingConfig
        from src.ssa.rl_features import SSA_OBS_SCHEMA_ID

        trainer = RLLibPPOTrainer(
            load_config("configs/experiments/ssa_sas_ao_rl_n3.yaml"),
            checkpoint_dir=tmp_path,
        )
        trainer._policy_observation_shapes = {"shared_policy": [90]}
        trainer._policy_action_nvec = {"shared_policy": [8, 8, 8]}
        trainer._write_manifest(
            str(tmp_path / "checkpoint_000001"),
            PolicySharingConfig(),
            ["shared_policy"],
        )

        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["observation_schema_id"] == SSA_OBS_SCHEMA_ID
        assert manifest["policy_observation_shapes"] == {"shared_policy": [90]}
        assert manifest["policy_action_nvec"] == {"shared_policy": [8, 8, 8]}

    def test_eventsat_manifest_records_log_observation_schema(self, tmp_path) -> None:
        import json

        from src.core.behaviour.rllib_training_pipeline import RLLibPPOTrainer
        from src.core.config_loader import load_config
        from src.eventsat.rl_obs_encoder import EVENTSAT_OBS_SCHEMA_ID
        from src.rl.policy_mapping import PolicySharingConfig

        trainer = RLLibPPOTrainer(
            load_config("configs/experiments/eventsat_sas_ao_rl.yaml"),
            checkpoint_dir=tmp_path,
        )
        trainer._policy_observation_shapes = {"shared_policy": [33]}
        trainer._policy_action_nvec = {"shared_policy": [7]}
        trainer._write_manifest(
            str(tmp_path / "checkpoint_000001"),
            PolicySharingConfig(),
            ["shared_policy"],
        )

        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["observation_schema_id"] == EVENTSAT_OBS_SCHEMA_ID

    def test_episode_reward_mean_reads_env_runner_metric(self, tmp_path) -> None:
        from src.core.behaviour.rllib_training_pipeline import RLLibPPOTrainer

        trainer = RLLibPPOTrainer(
            _minimal_config(max_steps=2), timesteps=1, checkpoint_dir=tmp_path
        )

        assert trainer._episode_reward_mean({"episode_reward_mean": 1.25}) == 1.25
        assert trainer._episode_reward_mean({"env_runners": {"episode_reward_mean": -0.5}}) == -0.5
        assert (
            trainer._episode_reward_mean({"env_runners": {"episode_return_mean": -0.75}}) == -0.75
        )
        assert trainer._episode_reward_mean({}) is None


class TestAUTOPSActorCriticModel:
    def test_forward_outputs_mode_logits_and_value(self) -> None:
        pytest.importorskip("ray")
        torch = pytest.importorskip("torch")
        spaces = pytest.importorskip("gymnasium.spaces")

        from src.rl.models.autops_actor_critic import AUTOPSActorCriticModel

        model = AUTOPSActorCriticModel(
            obs_space=spaces.Box(low=-1.0, high=2.0, shape=(25,)),
            action_space=spaces.MultiDiscrete([7]),
            num_outputs=7,
            model_config={"custom_model_config": {}},
            name="test_autops_actor_critic",
        )
        logits, state = model.forward(
            {"obs": torch.zeros((4, 25), dtype=torch.float32)},
            [],
            None,
        )

        assert state == []
        assert tuple(logits.shape) == (4, 7)
        assert tuple(model.value_function().shape) == (4,)
        assert len(model.actor_heads) == 1

    def test_forward_builds_every_declared_categorical_head(self) -> None:
        pytest.importorskip("ray")
        torch = pytest.importorskip("torch")
        spaces = pytest.importorskip("gymnasium.spaces")

        from src.rl.models.autops_actor_critic import AUTOPSActorCriticModel

        model = AUTOPSActorCriticModel(
            obs_space=spaces.Box(low=-1.0, high=2.0, shape=(25,)),
            action_space=spaces.MultiDiscrete([7, 3]),
            num_outputs=10,
            model_config={"custom_model_config": {}},
            name="test_extensible_autops_actor_critic",
        )
        logits, _ = model.forward(
            {"obs": torch.zeros((2, 25), dtype=torch.float32)},
            [],
            None,
        )

        assert tuple(logits.shape) == (2, 10)
        assert [head.out_features for head in model.actor_heads] == [7, 3]

"""RLlib PPO training pipeline for AUTOPS subsymbolic representations.

Uses RLlib as the canonical PPO backend while preserving the AUTOPS EventSat
observation/action design and ActorCritic architecture.

Grounding:
- Oliver et al. EUCASS 2025 (8KDZ5Z53): PPO hyperparameters, training protocol,
  [256,256] tanh architecture, EventSat obs/action space.
- Hamilton et al. 2025 (GWQ3LK6H): observation-space design and 10-seed
  evaluation protocol.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.core.config_loader import ExperimentConfig
from src.rl.policy_mapping import PolicySharingConfig, build_policy_specs
from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ARCHITECTURE = "autops_actor_critic_v1"


class RLLibPPOTrainer:
    """Train ``behaviour_config.mechanism = ppo`` with RLlib."""

    def __init__(
        self,
        config: ExperimentConfig | Dict[str, Any],
        *,
        timesteps: int | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.config = config if isinstance(config, ExperimentConfig) else ExperimentConfig(**config)
        self.training_config = dict(self.config.behaviour_config)
        if timesteps is not None:
            self.training_config["timesteps"] = int(timesteps)
        self.checkpoint_dir = Path(
            checkpoint_dir or f"data/trained_models/{self.config.experiment_id}"
        )
        self._last_result: Dict[str, Any] = {}

    def train(self) -> str:
        """Run RLlib PPO training and return the saved checkpoint path."""
        try:
            import ray
            from ray.rllib.algorithms.ppo import PPOConfig
            from ray.tune.registry import register_env
        except ImportError as exc:
            raise ImportError(
                "RLlib PPO training requires ray[rllib]. Install with: uv sync --extra rl"
            ) from exc

        started_ray = False
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False,
            )
            started_ray = True

        env_name = f"autops_{self.config.experiment_id}_rllib"
        env_config = {"experiment_config": self.config.model_dump()}
        register_env(env_name, lambda cfg: AUTOPSRLLibMultiAgentEnv(cfg))

        probe_env = AUTOPSRLLibMultiAgentEnv(env_config)
        sharing = PolicySharingConfig.from_config(
            self.training_config.get("policy_sharing", {"mode": "shared_all"})
        )
        policies = build_policy_specs(
            probe_env.possible_agents,
            probe_env.observation_space,
            probe_env.action_space,
            sharing,
        )

        config = PPOConfig()
        config = self._disable_new_api_stack_if_available(config)
        config = config.environment(env=env_name, env_config=env_config)
        config = config.framework("torch")
        config = self._configure_model(config)
        config = self._configure_rollouts(config)
        config = self._configure_resources(config)
        config = self._configure_training(config)
        config = config.multi_agent(
            policies=policies,
            policy_mapping_fn=sharing.mapping_fn(),
            policies_to_train=list(policies.keys()),
        )
        config = config.debugging(seed=self.config.seed)

        build = getattr(config, "build_algo", None) or getattr(config, "build")
        algo = build()
        try:
            target_timesteps = int(self.training_config.get("timesteps", 50_000))
            min_iterations = int(self.training_config.get("training_iterations", 1))
            max_iterations = int(self.training_config.get("max_iterations", 10_000))
            iterations = 0
            sampled_steps = 0
            while (
                (sampled_steps < target_timesteps or iterations < min_iterations)
                and iterations < max_iterations
            ):
                self._last_result = algo.train()
                iterations += 1
                sampled_steps = self._sampled_steps(self._last_result)
                progress = (
                    min(100.0, 100.0 * sampled_steps / target_timesteps)
                    if target_timesteps > 0
                    else 100.0
                )
                episode_reward = self._episode_reward_mean(self._last_result)
                reward_text = "n/a" if episode_reward is None else f"{episode_reward:.3f}"
                logger.info(
                    "RLlib PPO iteration %d: sampled_steps=%d/%d (%.1f%%) episode_reward=%s",
                    iterations,
                    sampled_steps,
                    target_timesteps,
                    progress,
                    reward_text,
                )

            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = algo.save(str(self.checkpoint_dir))
            checkpoint_path = self._checkpoint_to_path(checkpoint)
            self._write_manifest(checkpoint_path, sharing, policies.keys())
            return checkpoint_path
        finally:
            probe_env.close()
            algo.stop()
            if started_ray:
                ray.shutdown()

    def get_last_result(self) -> Dict[str, Any]:
        return dict(self._last_result)

    def _configure_rollouts(self, config: Any) -> Any:
        rollout_fragment = int(self.training_config.get("rollout_fragment", 128))
        num_env_runners = int(self.training_config.get("num_env_runners", 0))
        if hasattr(config, "env_runners"):
            try:
                return config.env_runners(
                    num_env_runners=num_env_runners,
                    rollout_fragment_length=rollout_fragment,
                    batch_mode="truncate_episodes",
                )
            except TypeError:
                pass
        if hasattr(config, "rollouts"):
            return config.rollouts(
                num_rollout_workers=num_env_runners,
                rollout_fragment_length=rollout_fragment,
                batch_mode="truncate_episodes",
            )
        return config

    def _configure_resources(self, config: Any) -> Any:
        if not hasattr(config, "resources"):
            return config
        return config.resources(num_gpus=float(self.training_config.get("num_gpus", 0)))

    def _configure_model(self, config: Any) -> Any:
        """Select the named AUTOPS neural architecture for RLlib."""
        architecture = self._model_architecture()
        rllib_model = dict(getattr(config, "model", {}) or {})

        if architecture != DEFAULT_MODEL_ARCHITECTURE:
            raise ValueError(
                f"Unknown PPO model_architecture='{architecture}'. "
                f"Supported: {DEFAULT_MODEL_ARCHITECTURE}"
            )

        from src.rl.models import register_autops_models

        register_autops_models()
        rllib_model["custom_model"] = architecture
        rllib_model["custom_model_config"] = {
            "hidden_size": 256,
            "action_dims": [7, 2, 2],
        }
        rllib_model["_disable_action_flattening"] = False

        config.model = rllib_model
        return config

    def _configure_training(self, config: Any) -> Any:
        params = inspect.signature(config.training).parameters
        minibatch = int(self.training_config.get("minibatch_size", 256))
        train_batch = int(
            self.training_config.get(
                "train_batch_size",
                self.training_config.get("batch_size", max(minibatch, int(self.training_config.get("rollout_fragment", 128)))),
            )
        )
        kwargs: Dict[str, Any] = {}
        self._maybe_set(kwargs, params, "lr", "lr")
        self._maybe_set(kwargs, params, "gamma", "gamma")
        self._maybe_set(kwargs, params, "clip_param", "clip_ratio")
        self._maybe_set(kwargs, params, "entropy_coeff", "entropy_coef")
        self._maybe_set(kwargs, params, "vf_loss_coeff", "value_coef")
        if "lambda_" in params and "gae_lambda" in self.training_config:
            kwargs["lambda_"] = self.training_config["gae_lambda"]
        elif "lambda" in params and "gae_lambda" in self.training_config:
            kwargs["lambda"] = self.training_config["gae_lambda"]
        if "train_batch_size" in params:
            kwargs["train_batch_size"] = train_batch
        if "minibatch_size" in params:
            kwargs["minibatch_size"] = min(minibatch, train_batch)
        elif "sgd_minibatch_size" in params:
            kwargs["sgd_minibatch_size"] = min(minibatch, train_batch)
        if "num_epochs" in params:
            kwargs["num_epochs"] = int(self.training_config.get("ppo_epochs", 30))
        elif "num_sgd_iter" in params:
            kwargs["num_sgd_iter"] = int(self.training_config.get("ppo_epochs", 30))
        if "grad_clip" in params and "max_grad_norm" in self.training_config:
            kwargs["grad_clip"] = self.training_config["max_grad_norm"]
        config = config.training(**kwargs)

        # Ray 2.x exposes several PPO knobs as config attributes rather than
        # explicit ``training()`` parameters.
        attr_map = {
            "train_batch_size": train_batch,
            "minibatch_size": min(minibatch, train_batch),
            "sgd_minibatch_size": min(minibatch, train_batch),
            "num_epochs": int(self.training_config.get("ppo_epochs", 30)),
            "num_sgd_iter": int(self.training_config.get("ppo_epochs", 30)),
        }
        if "lr" in self.training_config:
            attr_map["lr"] = self.training_config["lr"]
        if "gamma" in self.training_config:
            attr_map["gamma"] = self.training_config["gamma"]
        if "gae_lambda" in self.training_config:
            attr_map["lambda_"] = self.training_config["gae_lambda"]
        if "clip_ratio" in self.training_config:
            attr_map["clip_param"] = self.training_config["clip_ratio"]
        if "entropy_coef" in self.training_config:
            attr_map["entropy_coeff"] = self.training_config["entropy_coef"]
        if "value_coef" in self.training_config:
            attr_map["vf_loss_coeff"] = self.training_config["value_coef"]
        if "max_grad_norm" in self.training_config:
            attr_map["grad_clip"] = self.training_config["max_grad_norm"]
        if "lr_schedule" in self.training_config:
            attr_map["lr_schedule"] = self.training_config["lr_schedule"]
        for attr, value in attr_map.items():
            if hasattr(config, attr):
                setattr(config, attr, value)
        return config

    def _maybe_set(
        self,
        kwargs: Dict[str, Any],
        params: Dict[str, Any],
        rllib_key: str,
        autops_key: str,
    ) -> None:
        if rllib_key in params and autops_key in self.training_config:
            kwargs[rllib_key] = self.training_config[autops_key]

    def _disable_new_api_stack_if_available(self, config: Any) -> Any:
        if not hasattr(config, "api_stack"):
            return config
        try:
            return config.api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
        except TypeError:
            return config

    def _sampled_steps(self, result: Dict[str, Any]) -> int:
        for key in (
            "num_env_steps_sampled_lifetime",
            "num_agent_steps_sampled_lifetime",
            "timesteps_total",
        ):
            if key in result:
                return int(result[key])
        env_runner = result.get("env_runners")
        if isinstance(env_runner, dict):
            for key in ("num_env_steps_sampled_lifetime", "num_agent_steps_sampled_lifetime"):
                if key in env_runner:
                    return int(env_runner[key])
        return 0

    def _episode_reward_mean(self, result: Dict[str, Any]) -> float | None:
        value = result.get("episode_reward_mean")
        if value is None:
            env_runner = result.get("env_runners")
            if isinstance(env_runner, dict):
                value = env_runner.get("episode_reward_mean")
                if value is None:
                    value = env_runner.get("episode_return_mean")
        return None if value is None else float(value)

    def _checkpoint_to_path(self, checkpoint: Any) -> str:
        if isinstance(checkpoint, str):
            return checkpoint
        if hasattr(checkpoint, "checkpoint") and hasattr(checkpoint.checkpoint, "path"):
            return str(checkpoint.checkpoint.path)
        if hasattr(checkpoint, "path"):
            return str(checkpoint.path)
        return str(checkpoint)

    def _write_manifest(
        self,
        checkpoint_path: str,
        sharing: PolicySharingConfig,
        policy_ids: Any,
    ) -> None:
        manifest = {
            "experiment_id": self.config.experiment_id,
            "mechanism": "ppo",
            "implementation": "rllib",
            "checkpoint_path": checkpoint_path,
            "policy_sharing": sharing.mode,
            "policy_ids": list(policy_ids),
            "model_architecture": self._model_architecture(),
            "last_result": {
                key: value
                for key, value in self._last_result.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
        }
        with open(self.checkpoint_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _model_architecture(self) -> str:
        return str(
            self.training_config.get(
                "model_architecture",
                self.config.representation_config.get(
                    "model_architecture",
                    DEFAULT_MODEL_ARCHITECTURE,
                ),
            )
        )

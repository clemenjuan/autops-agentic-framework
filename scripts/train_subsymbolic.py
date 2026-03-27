"""
Training Script for Subsymbolic RL Representation.

Runs multi-episode PPO training using the EventSat Gymnasium wrapper,
saving checkpoints to data/trained_models/subsymbolic/.

Usage:
    uv run python scripts/train_subsymbolic.py
    uv run python scripts/train_subsymbolic.py --config configs/experiments/eventsat_cen_sda_subm_le_ah.yaml
    uv run python scripts/train_subsymbolic.py --episodes 1000 --seed 0 --checkpoint-freq 100

Design:
  - Uses EventSatGymnasium for the standard RL interface
  - Collects rollouts of length rollout_fragment (128 steps per EUCASS 2025)
  - Updates PPO every rollout_fragment steps
  - Saves checkpoint every --checkpoint-freq episodes
  - Logs metrics to stdout and data/trained_models/subsymbolic/training_log.json

Hyperparameters default to Oliver et al. EUCASS 2025 (8KDZ5Z53):
  [256,256] tanh, lr=1e-4→1e-5, gamma=0.97, clip=0.3, 30 epochs, batch=4096

Papers:
- Oliver et al. EUCASS 2025 (8KDZ5Z53): hyperparameters, training protocol
- Hamilton et al. 2025 (GWQ3LK6H): 10 seeds required for significance
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path when running directly
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train subsymbolic RL representation via PPO."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/eventsat_cen_sda_subm_le_ah.yaml",
        help="Path to experiment YAML config.",
    )
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override num_episodes from config.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max_steps from config.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed from config.")
    parser.add_argument(
        "--checkpoint-freq", type=int, default=100,
        help="Save checkpoint every N episodes.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/trained_models/subsymbolic",
        help="Directory for checkpoints and training log.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return parser.parse_args()


def setup_logging(log_level: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "training.log", mode="a"),
        ],
    )


def build_env_config(exp_config: dict) -> dict:
    """Extract env config dict from ExperimentConfig model_dump."""
    env_section = exp_config.get("environment", {})
    scenario_cfg = env_section.get("scenario_config", {})
    return {
        "max_steps": exp_config.get("max_steps", 10080),
        "step_duration_s": env_section.get("timestep_seconds", 60.0),
        **scenario_cfg,
    }


def main() -> None:
    args = parse_args()

    # --- Load config ---
    from src.orchestration.config_loader import load_config
    config = load_config(args.config)

    if args.episodes is not None:
        config = config.model_copy(update={"num_episodes": args.episodes})
    if args.steps is not None:
        config = config.model_copy(update={"max_steps": args.steps})
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})

    output_dir = Path(args.output_dir) / config.experiment_id
    setup_logging(args.log_level, output_dir)

    logger.info(
        "Starting training: %s — %d episodes, seed=%d",
        config.experiment_id, config.num_episodes, config.seed,
    )

    # --- Reproducibility ---
    import random
    random.seed(config.seed)
    np.random.seed(config.seed)
    try:
        import torch
        torch.manual_seed(config.seed)
    except ImportError:
        logger.warning("torch not available — cannot train ActorCritic")
        sys.exit(1)

    from torch.optim import Adam

    # --- Build Gymnasium wrapper ---
    from src.environment.gymnasium_wrapper import EventSatGymnasium, GYMNASIUM_AVAILABLE
    if not GYMNASIUM_AVAILABLE:
        logger.error("gymnasium not installed. Run: uv sync --extra rl")
        sys.exit(1)

    env_config = build_env_config(config.model_dump())
    env = EventSatGymnasium(env_config=env_config)

    # --- Build policy + trainer ---
    from src.representation.neural_policy import ActorCritic
    from src.emergence.rollout_buffer import RolloutBuffer
    from src.emergence.training_pipeline import PPOTrainer

    policy = ActorCritic()
    rollout_fragment = config.emergence_config.get("rollout_fragment", 128)
    buffer = RolloutBuffer(buffer_size=rollout_fragment)
    trainer = PPOTrainer(policy=policy, config=config.emergence_config)

    # Load checkpoint if specified
    checkpoint_path = config.representation_config.get("checkpoint_path", "")
    if checkpoint_path and os.path.exists(checkpoint_path):
        trainer.load(checkpoint_path)
        logger.info("Resumed from checkpoint at step %d", trainer.training_step)

    # --- Training loop ---
    training_log: list[dict] = []
    policy.train()

    for episode in range(config.num_episodes):
        ep_start = time.perf_counter()
        obs, _ = env.reset(seed=config.seed + episode)
        buffer.reset()

        ep_reward = 0.0
        ep_steps = 0
        update_infos: list[dict] = []

        for step in range(config.max_steps):
            # Collect step
            obs_tensor = torch.FloatTensor(obs)
            action, log_prob, value = policy.get_action(obs_tensor, deterministic=False)
            log_prob_val = float(log_prob.item()) if hasattr(log_prob, "item") else float(log_prob)
            value_val = float(value.item()) if hasattr(value, "item") else float(value)

            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

            buffer.store(
                obs=obs,
                action=action,
                reward=reward,
                value=value_val,
                log_prob=log_prob_val,
                done=done,
            )

            obs = obs_next

            # PPO update when buffer full
            if buffer.is_full:
                # Bootstrap last value
                with torch.no_grad():
                    if not done:
                        _, last_v = policy.forward(torch.FloatTensor(obs).unsqueeze(0))
                        last_value = float(last_v.item())
                    else:
                        last_value = 0.0
                buffer.observations[buffer.size - 1]  # ensure index is valid
                update_info = trainer.update(buffer)
                update_infos.append(update_info)
                buffer.reset()

            if done:
                break

        # Final update on partial buffer
        if buffer.size > 0:
            update_info = trainer.update(buffer)
            update_infos.append(update_info)
            buffer.reset()

        ep_duration = time.perf_counter() - ep_start
        mean_policy_loss = float(np.mean([u["policy_loss"] for u in update_infos])) if update_infos else 0.0
        mean_entropy = float(np.mean([u["entropy"] for u in update_infos])) if update_infos else 0.0

        log_entry = {
            "episode": episode,
            "reward": ep_reward,
            "steps": ep_steps,
            "training_step": trainer.training_step,
            "policy_loss": mean_policy_loss,
            "entropy": mean_entropy,
            "wall_clock_s": ep_duration,
        }
        training_log.append(log_entry)

        logger.info(
            "Episode %4d/%d | reward=%.3f | steps=%d | policy_loss=%.4f | "
            "entropy=%.4f | total_steps=%d",
            episode + 1, config.num_episodes,
            ep_reward, ep_steps,
            mean_policy_loss, mean_entropy,
            trainer.training_step,
        )

        # Save checkpoint
        if (episode + 1) % args.checkpoint_freq == 0:
            ckpt_path = output_dir / f"checkpoint_ep{episode+1:05d}.pt"
            trainer.save(ckpt_path)

    # Final checkpoint
    final_path = output_dir / "checkpoint_final.pt"
    trainer.save(final_path)
    logger.info("Training complete. Final checkpoint saved to %s", final_path)

    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info("Training log saved to %s", log_path)


if __name__ == "__main__":
    main()

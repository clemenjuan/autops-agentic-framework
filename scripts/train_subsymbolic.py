"""Train the subsymbolic EventSat policy with the canonical RLlib PPO backend.

Prefer the equivalent CLI:

    uv run autops train configs/experiments/eventsat_sas_sda_subm_le_ah.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train subsymbolic RL via RLlib PPO.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/eventsat_sas_sda_subm_le_ah.yaml",
        help="Path to experiment YAML config.",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Override training timesteps.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for RLlib checkpoints.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    from src.emergence.rllib_training_pipeline import RLLibPPOTrainer
    from src.orchestration.config_loader import load_config

    config = load_config(args.config)
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})

    checkpoint_dir = args.output_dir or f"data/trained_models/{config.experiment_id}"
    trainer = RLLibPPOTrainer(
        config,
        timesteps=args.timesteps,
        checkpoint_dir=checkpoint_dir,
    )
    checkpoint_path = trainer.train()
    logger.info("Training complete. RLlib checkpoint saved to %s", checkpoint_path)


if __name__ == "__main__":
    main()

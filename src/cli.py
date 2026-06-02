"""
AUTOPS CLI — unified entry point for running experiments and analysis.

Usage::

    uv run autops run configs/experiments/eventsat_sas_sda_symb_hd_ah.yaml
    uv run autops run configs/experiments/eventsat_sas_sda_symb_hd_ah.yaml --episodes 3 --analyze
    uv run autops batch configs/experiments/generated/
    uv run autops generate --template configs/experiments/template.yaml
    uv run autops analyze data/results/eventsat_sas_sda_symb_hd_ah/

    # Training learned-emergence variants
    uv run autops train configs/experiments/eventsat_sas_sda_subm_le_ah.yaml      # PPO
    uv run autops train configs/experiments/eventsat_sas_sda_hybr_lep_ah.yaml     # prompt-optimized
    uv run autops train configs/experiments/eventsat_sas_sda_agnt_lep_ah.yaml     # agentic prompt-opt
    uv run autops train configs/experiments/eventsat_sas_sda_agnt_lec_ah.yaml     # writable CoALA
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single experiment."""
    from src.orchestration.config_loader import load_config, apply_overrides
    from src.orchestration.experiment_runner import ExperimentRunner

    cfg = load_config(args.config)
    cfg = apply_overrides(
        cfg,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        output_dir=args.output_dir,
        log_level=getattr(args, "log_level", None),
    )
    runner = ExperimentRunner(config=cfg)
    results = runner.run()

    print(f"\nCompleted: {results['experiment_id']} "
          f"({results['num_episodes']} episodes)")
    print(f"Results saved to: {cfg.output_dir}")

    if args.analyze:
        print("\n--- Analysis ---\n")
        from src.orchestration.auto_analyze import run_analysis
        run_analysis(cfg.output_dir)


def cmd_batch(args: argparse.Namespace) -> None:
    """Run multiple experiments."""
    from src.orchestration.config_loader import load_config, apply_overrides
    from src.orchestration.experiment_runner import ExperimentRunner

    configs = _discover_configs(args.configs)
    if not configs:
        print("No YAML config files found.")
        sys.exit(1)

    total = len(configs)
    print(f"Running {total} experiment(s)...\n{'=' * 60}")

    successes = 0
    failures: List[str] = []

    for i, config_path in enumerate(configs, 1):
        print(f"\n[{i}/{total}] {config_path.name}\n{'-' * 60}")
        try:
            cfg = load_config(config_path)
            cfg = apply_overrides(
                cfg,
                episodes=args.episodes,
                steps=args.steps,
                seed=args.seed,
                log_level=getattr(args, "log_level", None),
            )
            runner = ExperimentRunner(config=cfg)
            results = runner.run()
            print(f"  Completed: {results['experiment_id']} "
                  f"({results['num_episodes']} episodes)")
            successes += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failures.append(str(config_path))

    print(f"\n{'=' * 60}")
    print(f"Results: {successes}/{total} succeeded, {len(failures)} failed")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate config combinations from a template."""
    # Import inline to avoid pulling in yaml at CLI parse time
    from scripts.generate_experiment_configs import generate_configs

    generated = generate_configs(
        template_path=args.template,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"Generated {len(generated)} configs in {args.output_dir}")
    for p in generated:
        print(f"  - {p.name}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train a learned-emergence representation for the given config.

    Dispatches based on ``representation`` × ``emergence_config.mechanism``:

    - subsymbolic + ppo         → RLLibPPOTrainer (writes RLlib checkpoint)
    - hybrid    + prompt_optimized → PromptOptimizer (writes prompt.txt)
    - hybrid    + writable_coala   → no pre-training; memory accretes online
    """
    from src.orchestration.config_loader import load_config, apply_overrides

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, seed=args.seed, output_dir=args.output_dir)

    representation = cfg.representation
    mechanism = cfg.emergence_config.get("mechanism", "")
    experiment_id = cfg.experiment_id

    print(f"Training: {experiment_id}")
    print(f"  representation : {representation}")
    print(f"  mechanism      : {mechanism or '(none)'}")

    if representation == "subsymbolic" and mechanism in ("ppo", ""):
        _train_ppo(cfg, args)

    elif representation == "hybrid" and mechanism == "prompt_optimized":
        _train_prompt_optimized(cfg, args)

    elif representation == "hybrid" and mechanism == "writable_coala":
        print(
            "\n[writable_coala] No pre-training needed — semantic and episodic "
            "memory accrete online during run-time.\n"
            "Run the experiment directly with:\n"
            f"  uv run autops run {args.config}"
        )

    else:
        print(
            f"ERROR: No training defined for representation='{representation}', "
            f"mechanism='{mechanism}'.\n"
            "Valid combinations: subsymbolic+ppo, hybrid+prompt_optimized, "
            "hybrid+writable_coala."
        )
        sys.exit(1)


def _train_ppo(cfg: "Any", args: argparse.Namespace) -> None:
    """Invoke RLlib PPO trainer for subsymbolic representation."""
    try:
        from src.emergence.rllib_training_pipeline import RLLibPPOTrainer
    except ImportError as e:
        print(
            f"ERROR: PPO training requires RLlib. Install with:\n"
            f"  uv sync --extra rl\n"
            f"Details: {e}"
        )
        sys.exit(1)

    training_cfg = cfg.emergence_config.get("training_config", {})
    timesteps = args.timesteps or training_cfg.get("timesteps", 50_000)
    checkpoint_dir = f"data/trained_models/{cfg.experiment_id}"

    print(f"  timesteps      : {timesteps}")
    print(f"  checkpoint_dir : {checkpoint_dir}")

    trainer = RLLibPPOTrainer(
        cfg,
        timesteps=timesteps,
        checkpoint_dir=checkpoint_dir,
    )
    checkpoint_path = trainer.train()
    print(f"\nPPO training complete. RLlib checkpoint saved to: {checkpoint_path}")


def _train_prompt_optimized(cfg: "Any", args: argparse.Namespace) -> None:
    """Invoke PromptOptimizer for prompt_optimized LLM/agentic representations."""
    from src.emergence.prompt_optimizer import PromptOptimizer

    # Derive the baseline source dir (hand-designed sibling)
    source_dir = args.source_dir or ""

    # Build config dict for optimizer (include experiment_id, repr type, llm settings)
    opt_config = {
        "experiment_id": cfg.experiment_id,
        "representation_config": cfg.representation_config,
        "emergence_config": cfg.emergence_config,
        "output_dir": "data/trained_prompts",
        **{k: v for k, v in (cfg.model_dump() if hasattr(cfg, "model_dump") else vars(cfg)).items()
           if k in ("llm_host", "llm_model", "llm_mock")},
    }
    # Pull LLM settings from representation_config if present
    repr_cfg = cfg.representation_config or {}
    for key in ("llm_host", "llm_model", "llm_mock"):
        if key in repr_cfg:
            opt_config[key] = repr_cfg[key]

    print(f"  source_dir     : {source_dir or '(auto-derived from experiment_id)'}")
    print(f"  output_dir     : data/trained_prompts/{cfg.experiment_id}/")

    optimizer = PromptOptimizer(config=opt_config)
    prompt = optimizer.optimize(
        source_results_dir=source_dir,
        seed=cfg.seed,
    )
    print(
        f"\nPrompt optimization complete.\n"
        f"Prompt saved to: data/trained_prompts/{cfg.experiment_id}/prompt.txt\n"
        f"Prompt length: {len(prompt)} characters"
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze experiment results."""
    from src.orchestration.auto_analyze import run_analysis

    run_analysis(args.results_dir, output_dir=args.output_dir)


def _discover_configs(paths: List[str]) -> List[Path]:
    """Discover YAML config files from paths (files, directories, or glob patterns)."""
    import glob as _glob
    configs: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix in (".yaml", ".yml"):
            configs.append(path)
        elif path.is_dir():
            configs.extend(sorted(path.glob("*.yaml")))
            configs.extend(sorted(path.glob("*.yml")))
        else:
            # Try glob expansion (handles patterns on shells that don't auto-expand,
            # e.g. PowerShell: eventsat_hier_*.yaml)
            matches = sorted(_glob.glob(p))
            yaml_matches = [Path(m) for m in matches if Path(m).suffix in (".yaml", ".yml")]
            if yaml_matches:
                configs.extend(yaml_matches)
            else:
                print(f"WARNING: Skipping '{p}' (not a file or directory)")
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autops",
        description="AUTOPS — Autonomous Satellite Operations Experiment Framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run a single experiment")
    p_run.add_argument("config", help="Path to experiment YAML config")
    p_run.add_argument("--episodes", type=int, help="Override num_episodes")
    p_run.add_argument("--steps", type=int, help="Override max_steps")
    p_run.add_argument("--seed", type=int, help="Override random seed")
    p_run.add_argument("--output-dir", help="Override output directory")
    p_run.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Override log_level (DEBUG enables per-step trace)")
    p_run.add_argument("--analyze", action="store_true",
                       help="Run analysis after experiment completes")
    p_run.set_defaults(func=cmd_run)

    # --- batch ---
    p_batch = subparsers.add_parser("batch", help="Run multiple experiments")
    p_batch.add_argument("configs", nargs="+",
                         help="YAML config files or directories")
    p_batch.add_argument("--episodes", type=int, help="Override num_episodes")
    p_batch.add_argument("--steps", type=int, help="Override max_steps")
    p_batch.add_argument("--seed", type=int, help="Override random seed")
    p_batch.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Override log_level (DEBUG enables per-step trace)")
    p_batch.set_defaults(func=cmd_batch)

    # --- generate ---
    p_gen = subparsers.add_parser("generate",
                                  help="Generate config combinations from template")
    p_gen.add_argument("--template", default="configs/experiments/template.yaml",
                       help="Template YAML file")
    p_gen.add_argument("--output-dir", default="configs/experiments/generated/",
                       help="Output directory for generated configs")
    p_gen.add_argument("--seed", type=int, default=42, help="Base random seed")
    p_gen.set_defaults(func=cmd_generate)

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze",
                                      help="Analyze experiment results")
    p_analyze.add_argument("results_dir",
                           help="Path to results directory or results.json")
    p_analyze.add_argument("--output-dir",
                           help="Directory for figure output (default: data/figures/)")
    p_analyze.set_defaults(func=cmd_analyze)

    # --- train ---
    p_train = subparsers.add_parser(
        "train",
        help="Train a learned-emergence representation (PPO / prompt-optimized / writable-CoALA)",
    )
    p_train.add_argument("config", help="Path to a *_le_* or *_lep_* or *_lec_* experiment YAML")
    p_train.add_argument("--timesteps", type=int,
                         help="Override PPO training timesteps (subsymbolic only)")
    p_train.add_argument("--source-dir",
                         help="Source results dir for prompt optimization "
                              "(default: auto-derived from experiment_id)")
    p_train.add_argument("--seed", type=int, help="Override random seed")
    p_train.add_argument("--output-dir", help="Override output directory")
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

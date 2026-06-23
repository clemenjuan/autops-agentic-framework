"""
Run Batch Experiments.

Execute one or more experiment configurations sequentially or in parallel.

Usage::

    # Single experiment
    python scripts/run_batch.py configs/experiments/exp_001.yaml

    # All configs in a directory
    python scripts/run_batch.py configs/experiments/generated/

    # Multiple configs
    python scripts/run_batch.py exp_001.yaml exp_002.yaml exp_003.yaml
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List

from src.core.experiment_runner import ExperimentRunner


def discover_configs(paths: List[str]) -> List[Path]:
    """Discover YAML config files from paths (files or directories).

    Args:
        paths: List of file or directory paths.

    Returns:
        Sorted list of YAML config file paths.
    """
    configs: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix in (".yaml", ".yml"):
            configs.append(path)
        elif path.is_dir():
            configs.extend(sorted(path.glob("*.yaml")))
            configs.extend(sorted(path.glob("*.yml")))
        else:
            print(f"WARNING: Skipping '{p}' (not a file or directory)")
    return configs


def run_batch(config_paths: List[Path]) -> None:
    """Run a batch of experiments.

    Args:
        config_paths: List of YAML config file paths to execute.
    """
    total = len(config_paths)
    print(f"Running {total} experiment(s)...")
    print("=" * 60)

    successes = 0
    failures: List[str] = []

    for i, config_path in enumerate(config_paths, 1):
        print(f"\n[{i}/{total}] {config_path.name}")
        print("-" * 60)

        try:
            runner = ExperimentRunner(config_path=config_path)
            results = runner.run()
            print(f"  ✓ Completed: {results['experiment_id']} "
                  f"({results['num_episodes']} episodes)")
            successes += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            failures.append(str(config_path))

    print("\n" + "=" * 60)
    print(f"Results: {successes}/{total} succeeded, {len(failures)} failed")
    if failures:
        print("Failed experiments:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch experiments.")
    parser.add_argument(
        "configs",
        nargs="+",
        help="YAML config files or directories containing them.",
    )
    args = parser.parse_args()

    config_paths = discover_configs(args.configs)
    if not config_paths:
        print("No YAML config files found.")
        sys.exit(1)

    run_batch(config_paths)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export AUTOPS SSA constellation telemetry in the ssa_world_model_v1 schema.

Adds a satellite axis and SSA collective coverage fields to the single-sat
EventSat world-model export. Consumed by space-world-models
(``swm_eventsat.schema.load_ssa_world_model_dataset``).

Example:
    uv run python scripts/export_ssa_world_model_traces.py \
        configs/experiments/ssa_imas_ao_symb_n3.yaml \
        --episodes 4 --steps 256 --out data/world_model/ssa_v1
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.core.config_loader import apply_overrides, load_config
from src.core.experiment_runner import ExperimentRunner
from src.ssa.trace import write_ssa_trace_metadata


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _episode_files(root: Path) -> List[Path]:
    return sorted(root.glob("*/trace/episode_*.npz")) + sorted(root.glob("trace/episode_*.npz"))


def _stack(files: List[Path], out_npz: Path) -> Dict[str, Any]:
    if not files:
        raise RuntimeError("no episode trace files were produced")
    blobs = [np.load(p, allow_pickle=False) for p in files]

    lengths = {int(b["obs"].shape[0]) for b in blobs}
    if len(lengths) != 1:
        raise RuntimeError(f"episode lengths differ; refusing to pad silently: {sorted(lengths)}")
    sat_counts = {int(b["obs"].shape[1]) for b in blobs}
    if len(sat_counts) != 1:
        raise RuntimeError(f"constellation sizes differ; refusing to pad: {sorted(sat_counts)}")

    per_step_keys = (
        "obs", "action", "state", "reward", "mode", "resolved_mode", "forced_mode",
        "delivered_coverage", "onboard_coverage", "archive_records",
    )
    arrays: Dict[str, np.ndarray] = {k: np.stack([b[k] for b in blobs], axis=0) for k in per_step_keys}
    arrays["episode_seed"] = np.asarray([int(b["episode_seed"]) for b in blobs], dtype=np.int64)
    arrays["episode_id"] = np.asarray([int(b["episode_id"]) for b in blobs], dtype=np.int64)
    arrays["sat_ids"] = blobs[0]["sat_ids"]

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    return {
        "episodes": int(arrays["obs"].shape[0]),
        "steps": int(arrays["obs"].shape[1]),
        "satellites": int(arrays["obs"].shape[2]),
        "obs_dim": int(arrays["obs"].shape[3]),
        "action_dim": int(arrays["action"].shape[3]),
        "state_dim": int(arrays["state"].shape[3]),
        "dataset_steps": int(arrays["obs"].shape[0] * arrays["obs"].shape[1] * arrays["obs"].shape[2]),
    }


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    run_records: List[Dict[str, Any]] = []

    for config_path in args.configs:
        cfg = load_config(config_path)
        cfg = apply_overrides(
            cfg,
            episodes=args.episodes,
            steps=args.steps,
            seed=args.seed,
            output_dir=str(out_root / "runs" / Path(config_path).stem / "autops_results"),
            log_level=args.log_level,
        )
        if cfg.environment.scenario != "ssa":
            raise SystemExit(f"{config_path} is not an SSA scenario (got {cfg.environment.scenario})")
        repr_cfg = dict(cfg.representation_config)
        repr_cfg["world_model_trace_dir"] = str(out_root / "runs" / cfg.experiment_id / "trace")
        cfg.representation_config = repr_cfg
        runner = ExperimentRunner(config=cfg)
        results = runner.run()
        run_records.append(
            {
                "config": str(config_path),
                "experiment_id": cfg.experiment_id,
                "episodes": int(results["num_episodes"]),
                "trace_dir": repr_cfg["world_model_trace_dir"],
                "results_dir": cfg.output_dir,
            }
        )

    episode_files = _episode_files(out_root / "runs")
    dataset_path = out_root / "ssa_world_model_v1.npz"
    summary = _stack(episode_files, dataset_path)
    write_ssa_trace_metadata(
        out_root / "ssa_world_model_v1.metadata.json",
        {
            "dataset": str(dataset_path),
            "autops_commit": _git_commit(),
            "source_runs": run_records,
            "summary": summary,
        },
    )
    print(
        f"wrote {dataset_path} episodes={summary['episodes']} "
        f"steps={summary['steps']} satellites={summary['satellites']} "
        f"dataset_steps={summary['dataset_steps']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="SSA experiment YAML files to roll out")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="data/world_model/ssa_v1")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

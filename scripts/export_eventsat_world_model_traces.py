#!/usr/bin/env python3
"""Export AUTOPS EventSat telemetry in the world-model dataset schema.

Example:
    uv run python scripts/export_eventsat_world_model_traces.py \
        configs/experiments/eventsat_sas_ao_symb.yaml \
        --episodes 4 --steps 256 --out data/world_model/eventsat_autops_v1
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.core.config_loader import apply_overrides, load_config
from src.eventsat.trace import write_trace_metadata
from src.core.experiment_runner import ExperimentRunner


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
    blobs = [np.load(p) for p in files]
    lengths = {int(b["obs"].shape[0]) for b in blobs}
    if len(lengths) != 1:
        raise RuntimeError(f"episode lengths differ; refusing to pad silently: {sorted(lengths)}")
    keys = ("obs", "action", "state", "reward", "mode", "resolved_mode", "forced_mode")
    arrays = {k: np.stack([b[k] for b in blobs], axis=0) for k in keys}
    arrays["episode_seed"] = np.asarray([int(b["episode_seed"]) for b in blobs], dtype=np.int64)
    arrays["episode_id"] = np.asarray([int(b["episode_id"]) for b in blobs], dtype=np.int64)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    return {
        "episodes": int(arrays["obs"].shape[0]),
        "steps": int(arrays["obs"].shape[1]),
        "obs_dim": int(arrays["obs"].shape[2]),
        "action_dim": int(arrays["action"].shape[2]),
        "state_dim": int(arrays["state"].shape[2]),
        "dataset_steps": int(arrays["obs"].shape[0] * arrays["obs"].shape[1]),
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
    dataset_path = out_root / "eventsat_world_model_v1.npz"
    summary = _stack(episode_files, dataset_path)
    write_trace_metadata(
        out_root / "eventsat_world_model_v1.metadata.json",
        {
            "dataset": str(dataset_path),
            "autops_commit": _git_commit(),
            "source_runs": run_records,
            "summary": summary,
        },
    )
    print(
        f"wrote {dataset_path} episodes={summary['episodes']} "
        f"steps={summary['steps']} dataset_steps={summary['dataset_steps']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="EventSat experiment YAML files to roll out")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="data/world_model/eventsat_autops_v1")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

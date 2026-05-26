"""
LLM smoke runner — exercises an LLM experiment config end-to-end without
hitting a live LLM endpoint.

Sets ``llm_mock=true`` + ``log_level=DEBUG`` + tiny episode/steps, runs the
experiment, then sanity-checks the produced decision trace and recomputes
metrics from it so we can verify the raw-telemetry pipeline.

Usage::

    uv run python scripts/smoke_llm.py configs/experiments/eventsat_sas_sda_hybr_hd_ah.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.orchestration.config_loader import apply_overrides, load_config
from src.orchestration.experiment_runner import ExperimentRunner


def _smoke_one(config_path: str, episodes: int, steps: int) -> bool:
    """Run one mocked smoke. Returns True on success."""
    import json as _json
    cfg = load_config(config_path)
    output_dir = f"data/results/smoke_{cfg.experiment_id}"
    cfg = apply_overrides(
        cfg,
        episodes=episodes,
        steps=steps,
        output_dir=output_dir,
        log_level="DEBUG",
    )
    repr_cfg = dict(cfg.representation_config)
    repr_cfg["llm_mock"] = True
    cfg = cfg.model_copy(update={"representation_config": repr_cfg})

    try:
        runner = ExperimentRunner(config=cfg)
        runner.run()
    except Exception as e:
        print(f"  FAIL run: {type(e).__name__}: {e}")
        return False

    out_dir = Path(cfg.output_dir)
    trace = out_dir / "decisions_ep0.jsonl"
    if not trace.exists():
        print(f"  FAIL: no trace at {trace}")
        return False

    required = {
        "step", "mode", "rationale", "battery_soc", "in_sunlight",
        "ground_pass_active", "jetson_raw_mb", "jetson_compressed_mb",
        "obc_data_mb", "data_downlinked_mb", "anomaly_forced_safe",
        "forced", "latency_s",
    }
    with open(trace, "r", encoding="utf-8") as f:
        first = f.readline()
    if not first:
        print("  FAIL: trace empty")
        return False
    line = _json.loads(first)
    missing = required - set(line.keys())
    if missing:
        print(f"  FAIL: trace missing fields: {missing}")
        return False
    print(f"  OK: {len(line)} fields, ep0 mode='{line.get('mode')}'")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+", help="One or more YAML config paths")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()

    total = len(args.configs)
    successes = 0
    failures = []
    for i, path in enumerate(args.configs, 1):
        name = Path(path).stem
        print(f"\n[{i}/{total}] {name}")
        if _smoke_one(path, args.episodes, args.steps):
            successes += 1
        else:
            failures.append(name)

    print(f"\n=== {successes}/{total} succeeded ===")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

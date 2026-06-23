"""
Recompute research metrics from raw per-step decision traces.

Reads ``decisions_ep<N>.jsonl`` files written by ``ExperimentRunner`` when
``log_level: DEBUG``, plus the saved ``config.json``, and re-runs the metric
aggregator (:class:`EventSatMetricsCollector`).  Lets you change metric
*definitions* in :mod:`src.eventsat.metrics` and regenerate
numbers without re-rolling expensive LLM episodes.

Usage::

    uv run python scripts/recompute_metrics.py data/results/eventsat_sas_ag_llm-s/
    uv run python scripts/recompute_metrics.py data/results/eventsat_sas_ah_*/
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on path when invoked as a script
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.eventsat.metrics import EventSatMetricsCollector
from src.core.metrics_collector import StepMetrics


def _trace_to_step_metrics(trace_lines: List[Dict[str, Any]]) -> List[StepMetrics]:
    """Convert raw trace lines into ``StepMetrics`` the aggregator expects."""
    out: List[StepMetrics] = []
    for line in trace_lines:
        # Derive what the live collector would have stored. Same field names.
        anomaly_raw = line.get("anomaly", "")
        anomaly_flag = float(bool(anomaly_raw))
        forced = float(line.get("forced", False))
        anomaly_active = float(line.get("anomaly_forced_safe") or 0.0)
        # Pre-transition safety classification: exact when the trace carries it,
        # else fall back to the executed mode (older traces predating safety_safe).
        _ss = line.get("safety_safe")
        _safety_safe = float(_ss) if _ss is not None else float(line.get("mode", "") == "safe")
        battery_soc = float(line.get("battery_soc") or 0.0)
        prev_soc = line.get("prev_battery_soc")
        # If prev_battery_soc wasn't recorded (older traces), fall back to 0
        # consumed — the aggregator just sums these.
        if prev_soc is None:
            soc_delta = 0.0
        else:
            soc_delta = max(0.0, float(prev_soc) - battery_soc)
        # Battery capacity is multiplied back by the aggregator's
        # ``_battery_capacity_wh`` — but here we precompute the same way.
        # Reuse the same constant by reading it later from collector config.

        metrics = {
            "battery_soc": battery_soc,
            "data_stored_mb": float(line.get("data_stored_mb") or 0.0),
            "data_downlinked_mb": float(line.get("data_downlinked_mb") or 0.0),
            "observation_hours": float(line.get("observation_hours") or 0.0),
            "total_detections": float(line.get("total_detections") or 0.0),
            "max_achievable_downlink_mb": float(line.get("max_achievable_downlink_mb") or 0.0),
            "in_sunlight": float(line.get("in_sunlight") or 0.0),
            "ground_pass_active": float(line.get("ground_pass_active") or 0.0),
            "forced": forced,
            "anomaly": anomaly_flag,
            # M-05 = protective safe mode (anomaly or critical battery); M-13 =
            # precondition clamp to charging. Keyed off the env's pre-transition
            # safety_safe when present (exact); fall back to the executed mode for
            # older traces. Matches EventSatMetricsCollector.collect_step_metrics.
            "safety_override": _safety_safe,
            "anomaly_active": anomaly_active,
            "in_safe_mode": _safety_safe,
            "constraint_violation": float(bool(forced) and _safety_safe == 0.0),
            "step_downlinked_mb": float(line.get("step_downlinked_mb") or 0.0),
            "total_raw_captured_mb": float(line.get("total_raw_captured_mb") or 0.0),
            "downlink_raw_equivalent_mb": float(line.get("downlink_raw_equivalent_mb") or 0.0),
            "jetson_raw_mb": float(line.get("jetson_raw_mb") or 0.0),
            "jetson_compressed_mb": float(line.get("jetson_compressed_mb") or 0.0),
            "obc_data_mb": float(line.get("obc_data_mb") or 0.0),
            "in_transition": float(line.get("in_transition") or 0.0),
            # Will be scaled in-place after we know battery capacity
            "_soc_delta": soc_delta,
            "decision_latency_s": float(line.get("latency_s") or 0.0),
            "has_rationale": float(bool(line.get("has_rationale", False))),
            "inference_allowed": float(bool(line.get("inference", line.get("inference_allowed", True)))),
            "orient_latency_s": float(line.get("orient_latency_s") or 0.0),
            "orient_iterations": float(line.get("orient_iterations") or 0.0),
            "orient_urgency": float(line.get("orient_urgency") or 0.0),
            "reasoning_depth": float(line.get("reasoning_depth") or 0.0),
            "react_iterations": float(line.get("react_iterations") or 0.0),
            "grounding_violations": float(line.get("grounding_violations") or 0.0),
            "converged": float(line.get("converged") or 0.0),
        }
        out.append(StepMetrics(
            timestep=int(line.get("step", 0)),
            wall_clock_seconds=float(line.get("latency_s") or 0.0),
            reward=float(line.get("reward") or 0.0),
            metrics=metrics,
            metadata={
                "requested_mode": line.get("requested_mode", ""),
                "resolved_mode": line.get("mode", ""),
            },
        ))
    return out


def _scale_energy(step_metrics: List[StepMetrics], battery_capacity_wh: float) -> None:
    """Fill in ``energy_consumed_wh`` using the SoC delta + capacity.

    Mirrors :meth:`EventSatMetricsCollector.collect_step_metrics`.
    """
    for s in step_metrics:
        s.metrics["energy_consumed_wh"] = s.metrics.pop("_soc_delta", 0.0) * battery_capacity_wh


def _load_trace(path: Path) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            lines.append(json.loads(raw))
    return lines


def recompute_for_dir(exp_dir: Path) -> Dict[str, Any] | None:
    """Recompute metrics for a single experiment results directory."""
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        print(f"[skip] {exp_dir}: no config.json")
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    trace_files = sorted(exp_dir.glob("decisions_ep*.jsonl"))
    if not trace_files:
        print(f"[skip] {exp_dir}: no decisions_ep*.jsonl traces "
              f"(run with --log-level DEBUG)")
        return None

    # Build collector with the same config the run used
    metrics_cfg = dict(config.get("metrics") or {})
    metrics_cfg["max_steps"] = config.get("max_steps", 10080)
    env_cfg = config.get("environment") or {}
    metrics_cfg["step_duration_s"] = env_cfg.get("timestep_seconds", 60.0)
    # Battery capacity isn't in the dumped config; fall back to the default
    # used by EventSatEnvironment (84.0 Wh) unless results.json has it.
    results_path = exp_dir / "results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            old_results = json.load(f)
        # Try to recover battery capacity from any stored env state, else default
        metrics_cfg["battery_capacity_wh"] = (
            old_results.get("battery_capacity_wh", 84.0)
        )
    else:
        metrics_cfg["battery_capacity_wh"] = 84.0

    collector = EventSatMetricsCollector(config=metrics_cfg)

    episode_metrics_list = []
    for ep_idx, tf in enumerate(trace_files):
        trace = _load_trace(tf)
        if not trace:
            continue
        step_metrics = _trace_to_step_metrics(trace)
        _scale_energy(step_metrics, collector._battery_capacity_wh)
        em = collector.aggregate_episode_metrics(step_metrics)
        em.episode_id = ep_idx
        episode_metrics_list.append(em)

    if not episode_metrics_list:
        print(f"[skip] {exp_dir}: all traces empty")
        return None

    stats = collector.compute_statistics(episode_metrics_list)

    # Serialise: dataclass → dict via asdict (skip step_metrics to keep size sane)
    def _ep_to_dict(em):
        d = dataclasses.asdict(em)
        d.pop("step_metrics", None)
        return d

    out = {
        "experiment_id": config.get("experiment_id"),
        "recomputed_at": datetime.now(timezone.utc).isoformat(),
        "num_episodes": len(episode_metrics_list),
        "experiment_statistics": {
            "num_episodes": stats.num_episodes,
            "mean": stats.mean,
            "std": stats.std,
            "min_val": stats.min_val,
            "max_val": stats.max_val,
        },
        "episodes": [_ep_to_dict(em) for em in episode_metrics_list],
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = exp_dir / f"metrics_recomputed_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"[ok]   {exp_dir.name}: {out.get('num_episodes')} episodes "
          f"-> {out_path.name}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute research metrics from raw decision traces."
    )
    parser.add_argument(
        "paths", nargs="+",
        help="Experiment result directories (globs OK).",
    )
    args = parser.parse_args()

    targets: List[Path] = []
    for raw in args.paths:
        # Expand shell glob if the shell didn't (e.g. PowerShell)
        matched = glob.glob(raw)
        if matched:
            for m in matched:
                p = Path(m)
                if p.is_dir():
                    targets.append(p)
        else:
            p = Path(raw)
            if p.is_dir():
                targets.append(p)

    if not targets:
        print("No result directories found.")
        sys.exit(1)

    ok = 0
    for d in sorted(set(targets)):
        if recompute_for_dir(d) is not None:
            ok += 1
    print(f"\nRecomputed {ok}/{len(targets)} experiments.")


if __name__ == "__main__":
    main()

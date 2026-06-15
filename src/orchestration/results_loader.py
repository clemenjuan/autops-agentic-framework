"""Results Loader — JSON experiment results to pandas DataFrames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a results.json file and return the raw dict."""
    with open(path) as f:
        return json.load(f)


def results_to_step_df(results: Dict[str, Any]) -> pd.DataFrame:
    """Flatten per-step data from all episodes into a single DataFrame.

    Columns include environment telemetry and, when available, the new
    research metrics (decision_latency_s, energy_consumed_wh, etc.).
    """
    rows: List[Dict[str, Any]] = []
    exp_id = results.get("experiment_id", "unknown")

    for ep in results.get("episodes", []):
        ep_id = ep["episode_id"]
        for s in ep["steps"]:
            row = {
                "experiment_id": exp_id,
                "episode_id": ep_id,
                "step": s["step"],
                "wall_clock_seconds": s.get("wall_clock_seconds", 0.0),
                "reward": sum(s.get("rewards", {}).values()),
            }
            row.update(s.get("info", {}))
            rows.append(row)

    return pd.DataFrame(rows)


def results_to_episode_df(results: Dict[str, Any]) -> pd.DataFrame:
    """Build per-episode summary DataFrame from results dict.

    If the results contain ``episode_metrics`` (from the new metrics
    pipeline), the research metrics are extracted from there.
    Otherwise falls back to re-computing from raw step data.
    """
    rows: List[Dict[str, Any]] = []
    exp_id = results.get("experiment_id", "unknown")

    for ep in results.get("episodes", []):
        ep_id = ep["episode_id"]
        steps = ep["steps"]
        n = len(steps)
        total_reward = sum(
            sum(s.get("rewards", {}).values()) for s in steps
        )
        forced_count = sum(1 for s in steps if s.get("info", {}).get("forced"))
        anomaly_count = sum(
            1 for s in steps
            if s.get("info", {}).get("anomaly") is not None
            and s.get("info", {}).get("anomaly") is not False
            and s.get("info", {}).get("anomaly") != 0
            and s.get("info", {}).get("anomaly") != 0.0
        )
        last_info = steps[-1].get("info", {}) if steps else {}

        row = {
            "experiment_id": exp_id,
            "episode_id": ep_id,
            "num_steps": n,
            "wall_clock_seconds": ep.get("wall_clock_seconds", 0.0),
            "total_reward": total_reward,
            "mean_reward": total_reward / max(n, 1),
            "observation_hours": last_info.get("observation_hours", 0.0),
            "downlinked_mb": last_info.get("data_downlinked_mb", 0.0),
            "final_battery_soc": last_info.get("battery_soc", 0.0),
            "final_data_stored_mb": last_info.get("data_stored_mb", 0.0),
            "forced_mode_changes": forced_count,
            "anomaly_events": anomaly_count,
        }

        # Extract research metrics from the new pipeline if available
        em = ep.get("episode_metrics")
        if em is not None and isinstance(em, dict):
            agg = em.get("aggregated", {})
            # Merge all aggregated metrics dynamically so loop-specific metrics
            # (OODA: mean_orient_latency_s, ReAct: reasoning_depth, etc.) are
            # included without needing explicit registration here.
            row.update(agg)

        rows.append(row)

    return pd.DataFrame(rows)


def results_to_statistics_df(results: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Extract experiment-level statistics as a single-row DataFrame.

    Returns None if ``experiment_statistics`` is not present in the results.
    """
    stats = results.get("experiment_statistics")
    if stats is None or not isinstance(stats, dict):
        return None

    row: Dict[str, Any] = {
        "experiment_id": stats.get("experiment_id", results.get("experiment_id", "unknown")),
        "num_episodes": stats.get("num_episodes", 0),
    }

    # Flatten mean/std/min/max dicts
    for prefix, d in [("mean", stats.get("mean", {})),
                      ("std", stats.get("std", {})),
                      ("min", stats.get("min_val", {})),
                      ("max", stats.get("max_val", {}))]:
        if isinstance(d, dict):
            for k, v in d.items():
                row[f"{prefix}_{k}"] = v

    # Metadata (scale & complexity)
    meta = stats.get("metadata", {})
    if isinstance(meta, dict):
        for k, v in meta.items():
            row[k] = v

    return pd.DataFrame([row])


def load_experiment(
    path: Union[str, Path],
) -> tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Convenience: load results and return (raw_dict, step_df, episode_df)."""
    results = load_results(path)
    return results, results_to_step_df(results), results_to_episode_df(results)


def load_multiple_experiments(
    paths: List[Union[str, Path]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load multiple experiment results and concatenate into unified DataFrames.

    Also injects morphological matrix dimensions from the config section.

    Returns:
        (step_df, episode_df) with all experiments combined.
    """
    step_dfs: List[pd.DataFrame] = []
    episode_dfs: List[pd.DataFrame] = []

    for p in paths:
        results = load_results(p)
        s_df = results_to_step_df(results)
        e_df = results_to_episode_df(results)

        # Tag with morphological matrix dimensions from config
        cfg = results.get("config", {})
        for dim in [
            "agent_organization",
            "decision_procedure",
            "representation",
            "behaviour",
            "operations_paradigm",
        ]:
            s_df[dim] = cfg.get(dim, "unknown")
            e_df[dim] = cfg.get(dim, "unknown")

        step_dfs.append(s_df)
        episode_dfs.append(e_df)

    return (
        pd.concat(step_dfs, ignore_index=True),
        pd.concat(episode_dfs, ignore_index=True),
    )

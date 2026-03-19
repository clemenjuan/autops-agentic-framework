"""
Statistical Analysis Utilities.

Helpers for post-experiment statistical analysis and Pareto frontier
computation. Includes confidence intervals, non-parametric hypothesis
tests, and scaling analysis for cross-experiment comparison.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_pareto_frontier(
    points: List[Dict[str, float]],
    objectives: List[str],
    maximise: Optional[List[bool]] = None,
) -> List[int]:
    """Identify indices of Pareto-optimal points.

    A point is Pareto-optimal if no other point is at least as good on
    all objectives and strictly better on at least one.

    Args:
        points: List of dictionaries mapping objective names to values.
        objectives: Ordered list of objective names to consider.
        maximise: Per-objective flag indicating whether higher is better.
            Defaults to ``True`` for all objectives.

    Returns:
        Sorted list of indices into ``points`` that are Pareto-optimal.
    """
    if not points or not objectives:
        return []

    if maximise is None:
        maximise = [True] * len(objectives)

    n = len(points)
    # Build matrix (n x k)
    matrix = np.array(
        [[p.get(obj, 0.0) for obj in objectives] for p in points],
        dtype=np.float64,
    )

    # Flip sign for minimisation objectives so we can always maximise
    for j, do_max in enumerate(maximise):
        if not do_max:
            matrix[:, j] *= -1

    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # j dominates i if j >= i on all objectives and j > i on at least one
            if np.all(matrix[j] >= matrix[i]) and np.any(matrix[j] > matrix[i]):
                is_dominated[i] = True
                break

    return sorted([i for i in range(n) if not is_dominated[i]])


def compare_experiments(
    experiment_stats: List[Dict[str, Any]],
    metric_name: str,
) -> Dict[str, Any]:
    """Compare a single metric across multiple experiments.

    Args:
        experiment_stats: List of experiment summary dictionaries, each
            containing at least ``experiment_id`` and a ``mean`` dict.
        metric_name: Name of the metric to compare.

    Returns:
        Dictionary with comparison results including ranking.
    """
    entries = []
    for stats in experiment_stats:
        exp_id = stats.get("experiment_id", "unknown")
        mean_val = stats.get("mean", {}).get(metric_name)
        std_val = stats.get("std", {}).get(metric_name)
        if mean_val is not None:
            entries.append(
                {
                    "experiment_id": exp_id,
                    "mean": mean_val,
                    "std": std_val,
                }
            )

    # Sort descending by mean
    entries.sort(key=lambda e: e["mean"], reverse=True)

    return {
        "metric": metric_name,
        "ranking": entries,
        "num_experiments": len(entries),
    }


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute mean and confidence interval using t-distribution.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    n = len(data)
    mean = float(np.mean(data))
    if n < 2:
        return mean, mean, mean
    se = float(stats.sem(data))
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


def episode_summary_with_ci(
    episode_df: pd.DataFrame,
    metrics: List[str],
    group_col: str = "experiment_id",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Per-experiment mean +/- CI for each metric.

    Returns DataFrame with columns: group, metric, mean, ci_lower, ci_upper, n.
    """
    rows: List[Dict[str, Any]] = []
    for name, grp in episode_df.groupby(group_col):
        for m in metrics:
            vals = grp[m].dropna().values
            mean, lo, hi = confidence_interval(vals, confidence)
            rows.append({
                group_col: name,
                "metric": m,
                "mean": mean,
                "ci_lower": lo,
                "ci_upper": hi,
                "n": len(vals),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Non-parametric hypothesis tests
# ---------------------------------------------------------------------------


def mann_whitney_test(
    episode_df: pd.DataFrame,
    metric: str,
    group_col: str = "experiment_id",
) -> pd.DataFrame:
    """Pairwise Mann-Whitney U tests between all experiment groups.

    Returns DataFrame with columns: group_a, group_b, U_statistic, p_value.
    """
    groups = list(episode_df[group_col].unique())
    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            vals_a = episode_df.loc[episode_df[group_col] == a, metric].dropna()
            vals_b = episode_df.loc[episode_df[group_col] == b, metric].dropna()
            if len(vals_a) < 2 or len(vals_b) < 2:
                continue
            u_stat, p_val = stats.mannwhitneyu(
                vals_a, vals_b, alternative="two-sided",
            )
            rows.append({
                "group_a": a,
                "group_b": b,
                "U_statistic": u_stat,
                "p_value": p_val,
            })
    return pd.DataFrame(rows)


def kruskal_wallis_test(
    episode_df: pd.DataFrame,
    metric: str,
    group_col: str = "experiment_id",
) -> Dict[str, float]:
    """Kruskal-Wallis H-test across all experiment groups.

    Returns dict with H_statistic and p_value.
    """
    groups = [
        grp[metric].dropna().values
        for _, grp in episode_df.groupby(group_col)
    ]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return {"H_statistic": float("nan"), "p_value": float("nan")}
    h_stat, p_val = stats.kruskal(*groups)
    return {"H_statistic": h_stat, "p_value": p_val}


# ---------------------------------------------------------------------------
# Scaling analysis
# ---------------------------------------------------------------------------


def scaling_analysis(
    episode_df: pd.DataFrame,
    metric: str,
    size_col: str = "constellation_size",
    group_col: str = "experiment_id",
) -> pd.DataFrame:
    """Compute mean metric as a function of constellation size per architecture.

    Requires `size_col` in the DataFrame (injected from config during loading).

    Returns DataFrame with columns: group, constellation_size, mean, std, n.
    """
    rows: List[Dict[str, Any]] = []
    for (name, size), grp in episode_df.groupby([group_col, size_col]):
        vals = grp[metric].dropna().values
        rows.append({
            group_col: name,
            size_col: size,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
        })
    return pd.DataFrame(rows)

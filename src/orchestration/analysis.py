"""
Statistical Analysis Utilities.

Helpers for post-experiment statistical analysis and Pareto frontier
computation. These will be extended as the metrics framework matures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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

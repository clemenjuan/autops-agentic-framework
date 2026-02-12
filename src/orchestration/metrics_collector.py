"""
Metrics Collector — Abstract Framework.

Collects, aggregates, and computes statistics over experiment metrics.
Specific metric implementations will be developed following literature
review and theoretical justification.

Core metrics (all require deeper study for precise operationalisation):
- **Utility**: Total value achieved from completed tasks/objectives.
- **Latency**: Decision-making computational time.
- **Robustness**: Performance stability under perturbations.
- **Resource Efficiency**: Utility per unit resource consumed.
- **Operator Load**: Required human intervention frequency.
- **Scalability**: Performance degradation as constellation size increases.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepMetrics:
    """Metrics collected for a single simulation timestep.

    Attributes:
        timestep: Simulation step index.
        wall_clock_seconds: Wall-clock time for this step.
        metrics: Dictionary of metric name → value.
    """

    timestep: int = 0
    wall_clock_seconds: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for one complete episode.

    Attributes:
        episode_id: Episode index.
        num_steps: Total steps taken in this episode.
        total_wall_clock_seconds: Total wall-clock duration.
        aggregated: Aggregated metric values (e.g. means, totals).
        step_metrics: Optional list of per-step metrics (for detailed analysis).
    """

    episode_id: int = 0
    num_steps: int = 0
    total_wall_clock_seconds: float = 0.0
    aggregated: Dict[str, float] = field(default_factory=dict)
    step_metrics: List[StepMetrics] = field(default_factory=list)


@dataclass
class ExperimentStatistics:
    """Statistical summary across all episodes of an experiment.

    Attributes:
        experiment_id: Experiment identifier.
        num_episodes: Number of episodes run.
        mean: Mean of each metric across episodes.
        std: Standard deviation of each metric across episodes.
        min_val: Minimum value of each metric across episodes.
        max_val: Maximum value of each metric across episodes.
        raw_episodes: Optional list of per-episode metrics.
    """

    experiment_id: str = ""
    num_episodes: int = 0
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)
    min_val: Dict[str, float] = field(default_factory=dict)
    max_val: Dict[str, float] = field(default_factory=dict)
    raw_episodes: List[EpisodeMetrics] = field(default_factory=list)


class MetricsCollector(ABC):
    """Abstract metrics collection framework.

    Subclasses implement scenario-specific metric computation.
    The base class provides timing utilities and the collection pipeline.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialise the metrics collector.

        Args:
            config: Metrics configuration section from experiment YAML.
        """
        self.config = config or {}
        self._episode_step_metrics: List[StepMetrics] = []
        self._all_episode_metrics: List[EpisodeMetrics] = []
        self._step_start_time: float = 0.0

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------

    def start_step_timer(self) -> None:
        """Record the start time for the current step."""
        self._step_start_time = time.perf_counter()

    def get_step_elapsed(self) -> float:
        """Return elapsed seconds since :meth:`start_step_timer`."""
        return time.perf_counter() - self._step_start_time

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def collect_step_metrics(
        self,
        env_state: Any,
        actions: Any,
        rewards: Dict[str, float],
        info: Dict[str, Any],
    ) -> StepMetrics:
        """Collect metrics for a single timestep.

        Args:
            env_state: Current environment state / observation.
            actions: Actions taken this step.
            rewards: Reward components from the environment.
            info: Additional info dict from the environment step.

        Returns:
            Populated :class:`StepMetrics` for this timestep.
        """
        ...

    @abstractmethod
    def aggregate_episode_metrics(
        self,
        step_metrics: List[StepMetrics],
    ) -> EpisodeMetrics:
        """Aggregate step-level metrics into an episode summary.

        Args:
            step_metrics: List of per-step metrics for the completed episode.

        Returns:
            Aggregated :class:`EpisodeMetrics`.
        """
        ...

    @abstractmethod
    def compute_statistics(
        self,
        episode_metrics: List[EpisodeMetrics],
    ) -> ExperimentStatistics:
        """Compute statistical measures across all episodes.

        Args:
            episode_metrics: List of per-episode aggregated metrics.

        Returns:
            :class:`ExperimentStatistics` with means, stds, etc.
        """
        ...

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def record_step(
        self,
        env_state: Any,
        actions: Any,
        rewards: Dict[str, float],
        info: Dict[str, Any],
    ) -> StepMetrics:
        """Collect and store metrics for one step.

        Wrapper around :meth:`collect_step_metrics` that also appends
        the result to the internal episode buffer.

        Returns:
            The collected :class:`StepMetrics`.
        """
        sm = self.collect_step_metrics(env_state, actions, rewards, info)
        self._episode_step_metrics.append(sm)
        return sm

    def finalise_episode(self, episode_id: int) -> EpisodeMetrics:
        """Finalise the current episode and store the aggregated metrics.

        Args:
            episode_id: Episode index.

        Returns:
            Aggregated :class:`EpisodeMetrics`.
        """
        em = self.aggregate_episode_metrics(self._episode_step_metrics)
        em.episode_id = episode_id
        self._all_episode_metrics.append(em)
        self._episode_step_metrics = []
        return em

    def finalise_experiment(self, experiment_id: str) -> ExperimentStatistics:
        """Compute final statistics after all episodes are complete.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            :class:`ExperimentStatistics`.
        """
        stats = self.compute_statistics(self._all_episode_metrics)
        stats.experiment_id = experiment_id
        return stats

    def reset(self) -> None:
        """Clear all collected data."""
        self._episode_step_metrics = []
        self._all_episode_metrics = []

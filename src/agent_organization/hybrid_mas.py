"""
Hybrid MAS Organization.

Implements the Hybrid MAS topology from Kim et al. (2025) [FVFQ73RF]
"Towards a Science of Scaling Agent Systems":
  S = (A, E, C, Ω) where
  A = mixed agent types (cluster heads over local satellites),
  C = heterogeneous (coordination *within* a cluster, none *across* clusters),
  Ω = hybrid (within-cluster deconfliction + across-cluster independence).

The constellation is partitioned into clusters. Each cluster has a head agent
that sees only its own cluster's satellites and their visible tasks and produces
a deconflicted assignment for that cluster (a mini-SAS). Across clusters there is
no coordination, so two satellites in *different* clusters can still collide on
the same RSO.

This makes HybridMAS the tunable midpoint of the organisation axis:

- ``num_clusters = 1``  → one cluster sees everything → behaves like SAS
  (fully coordinated, zero duplicates).
- ``num_clusters = n``  → every satellite is its own singleton cluster → behaves
  like IndependentMAS (no coordination, maximal duplicates).
- in between → partial coordination: duplicates only across cluster boundaries,
  so mission outcome and coordination cost both land between SAS and IMAS.

Coordination cost mirrors DecentralizedMAS but localised to clusters:
``coordination_messages = Σ_i c_i·(c_i-1)`` over clusters of size ``c_i`` — which
is ``n·(n-1)`` for one cluster (≡ DMAS) and ``0`` for all singletons (≡ IMAS).

First increment uses contiguous near-equal clusters (``num_clusters``, default 2)
or an explicit ``clusters`` partition. Visibility-/capability-based clustering is
future work.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agent_organization.base import AgentAction, AgentObservation, AgentOrganization


class HybridMAS(AgentOrganization):
    """Clustered heterogeneous organisation: coordinate within, independent across.

    C = heterogeneous, Ω = hybrid. See Kim et al. (2025) [FVFQ73RF] §4.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._clusters: List[List[int]] = []
        self._agent_ids: List[str] = []
        self._messages_exchanged: int = 0

    def initialize(self, constellation_size: int, **kwargs: Any) -> None:
        self._clusters = self._build_clusters(constellation_size)
        self._agent_ids = [f"cluster_agent_{i}" for i in range(len(self._clusters))]
        self._messages_exchanged = 0

    def _build_clusters(self, n: int) -> List[List[int]]:
        explicit = self.config.get("clusters")
        if explicit:
            return [[int(i) for i in cluster] for cluster in explicit]
        num_clusters = int(self.config.get("num_clusters", 2))
        num_clusters = max(1, min(num_clusters, n)) if n > 0 else 1
        # Contiguous near-equal split of range(n) into num_clusters groups.
        base, extra = divmod(n, num_clusters)
        clusters: List[List[int]] = []
        start = 0
        for c in range(num_clusters):
            size = base + (1 if c < extra else 0)
            clusters.append(list(range(start, start + size)))
            start += size
        return [cluster for cluster in clusters if cluster]

    def distribute_observation(
        self,
        env_observation: Any,
    ) -> Dict[str, AgentObservation]:
        """Each cluster head sees only its own cluster's satellites and tasks."""
        sat_ids = list(env_observation.constellation_state.satellites.keys())
        result: Dict[str, AgentObservation] = {}
        for cluster_idx, member_indices in enumerate(self._clusters):
            member_ids = [
                sat_ids[i] for i in member_indices if i < len(sat_ids)
            ]
            result[self._agent_ids[cluster_idx]] = AgentObservation(
                agent_id=self._agent_ids[cluster_idx],
                local_state={
                    "full_observation": self._cluster_view(env_observation, member_ids)
                },
            )
        return result

    def collect_actions(
        self,
        agent_actions: Dict[str, AgentAction],
    ) -> Dict[str, Any]:
        """Merge per-cluster assignments; no deconfliction *across* clusters.

        Within a cluster the head already deconflicted; across clusters the
        merge keeps any collisions (Ω = hybrid). Records the localised
        coordination cost ``Σ c_i·(c_i-1)``.
        """
        merged: Dict[str, Any] = {}
        for agent_action in agent_actions.values():
            if agent_action is not None and isinstance(agent_action.action, dict):
                merged.update(agent_action.action)
        self._messages_exchanged = sum(
            len(cluster) * (len(cluster) - 1) for cluster in self._clusters
        )
        return merged

    def get_agents(self) -> List[str]:
        return list(self._agent_ids)

    def get_metrics(self) -> Dict[str, float]:
        return {
            "coordination_messages": float(self._messages_exchanged),
            "num_clusters": float(len(self._clusters)),
        }

    @staticmethod
    def _cluster_view(env_observation: Any, member_ids: List[str]) -> Any:
        """Build a slice of the observation containing only the cluster's sats."""
        from src.environment.satellite_env import (
            ConstellationState,
            EnvironmentObservation,
        )

        cstate = env_observation.constellation_state
        member_set = set(member_ids)
        local_satellites = {
            sid: cstate.satellites[sid]
            for sid in member_ids
            if sid in cstate.satellites
        }
        local_tasks = [
            task
            for task in (getattr(env_observation, "tasks", []) or [])
            if task.get("satellite_id") in member_set
        ]
        return EnvironmentObservation(
            constellation_state=ConstellationState(
                timestep=cstate.timestep,
                epoch_seconds=cstate.epoch_seconds,
                satellites=local_satellites,
                global_info=dict(getattr(cstate, "global_info", {}) or {}),
            ),
            tasks=local_tasks,
            events=list(getattr(env_observation, "events", []) or []),
        )

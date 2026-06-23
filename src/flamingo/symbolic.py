"""
Rule-based symbolic planner for the Flamingo-lite SSA scenario.

The planner greedily assigns each satellite its highest-priority visible RSO
while avoiding duplicate target assignments within the same decision cycle.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.behaviour.controller import register
from src.core.representation import Representation


@register("rule_based_flamingo")
class RuleBasedFlamingo(Representation):
    """Greedy symbolic scheduler for Flamingo-lite."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._last_rationale: Optional[str] = None

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        if not hasattr(observation, "constellation_state"):
            return {"satellites": [], "tasks": []}

        satellites = list(observation.constellation_state.satellites.keys())
        tasks = []
        for task in getattr(observation, "tasks", []) or []:
            tasks.append({
                "satellite_id": task.get("satellite_id"),
                "target_id": task.get("target_id"),
                "priority": float(task.get("priority", 0.0)),
            })
        return {"satellites": satellites, "tasks": tasks}

    def select_action(self, context: Any) -> Dict[str, Any]:
        state = context.state or {}
        tasks_by_sat: Dict[str, List[Dict[str, Any]]] = {
            sat_id: [] for sat_id in state.get("satellites", [])
        }
        for task in state.get("tasks", []):
            sat_id = task.get("satellite_id")
            if sat_id in tasks_by_sat:
                tasks_by_sat[sat_id].append(task)

        assigned_targets: set[str] = set()
        actions: Dict[str, Dict[str, str]] = {}
        assignments = []

        for sat_id, tasks in tasks_by_sat.items():
            candidates = sorted(
                tasks,
                key=lambda item: (-float(item.get("priority", 0.0)), item.get("target_id", "")),
            )
            chosen = None
            for task in candidates:
                target_id = task.get("target_id")
                if target_id and target_id not in assigned_targets:
                    chosen = target_id
                    break
            if chosen is None:
                actions[sat_id] = {"target_id": "idle"}
                continue
            assigned_targets.add(chosen)
            actions[sat_id] = {"target_id": chosen}
            assignments.append(f"{sat_id}->{chosen}")

        if assignments:
            self._last_rationale = "Greedy priority assignment: " + ", ".join(assignments)
        else:
            self._last_rationale = "No visible unassigned RSO targets; idling."
        return actions

    def get_rationale(self) -> Optional[str]:
        return self._last_rationale


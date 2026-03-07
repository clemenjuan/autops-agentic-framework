"""
Rule-Based Symbolic Representation for EventSat.

Hand-designed priority rules for single-satellite mode selection.
Registered as "rule_based_eventsat" in the emergence controller.
"""
from __future__ import annotations
from typing import Any, Dict
from src.representation.base import Representation
from src.emergence.controller import register


@register("rule_based_eventsat")
class RuleBasedEventSat(Representation):
    """Priority-based rule system for EventSat mode selection."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        """Extract decision-relevant features from the environment observation."""
        if hasattr(observation, "constellation_state"):
            sat = observation.constellation_state.satellites.get("eventsat_0")
            if sat is None:
                return {}
            res = sat.resources or {}
            meta = sat.metadata or {}
            return {
                "battery_soc": res.get("battery_soc", 0.5),
                "current_mode": sat.status,
                "in_sunlight": meta.get("in_sunlight", False),
                "ground_pass_active": meta.get("ground_pass_active", False),
                "data_stored_mb": res.get("data_stored_mb", 0.0),
                "storage_capacity_mb": meta.get("storage_capacity_mb", 512.0),
                "uncompressed_observations": meta.get("uncompressed_observations", 0),
                "total_observation_s": meta.get("total_observation_s", 0.0),
                "health_status": meta.get("health_status", "nominal"),
            }
        return {}

    def select_action(self, state: Dict[str, Any], memory: Any = None) -> Dict[str, Any]:
        """Apply priority rules to select the next mode."""
        if not state:
            return {"eventsat_0": {"mode": "charging"}}

        soc = state.get("battery_soc", 0.5)
        pass_active = state.get("ground_pass_active", False)
        data_mb = state.get("data_stored_mb", 0.0)
        cap_mb = state.get("storage_capacity_mb", 512.0)
        uncomp = state.get("uncompressed_observations", 0)
        health = state.get("health_status", "nominal")

        # Rule 1: Anomaly -> safe
        if health != "nominal":
            return {"eventsat_0": {"mode": "safe"}}

        # Rule 2: Critically low battery -> charge
        if soc < 0.35:
            return {"eventsat_0": {"mode": "charging"}}

        # Rule 3: Ground pass active with data -> communicate
        if pass_active and data_mb > 0:
            return {"eventsat_0": {"mode": "communication"}}

        # Rule 4: Storage pressure + uncompressed data -> compress
        if data_mb > cap_mb * 0.7 and uncomp > 0:
            return {"eventsat_0": {"mode": "payload_compress"}}

        # Rule 5: Compression backlog
        if uncomp >= 3:
            return {"eventsat_0": {"mode": "payload_compress"}}

        # Rule 6: Good battery + storage space -> observe
        if soc > 0.6 and data_mb < cap_mb * 0.8:
            return {"eventsat_0": {"mode": "payload_observe"}}

        # Rule 7: Some uncompressed + decent battery -> compress
        if uncomp > 0 and soc > 0.45:
            return {"eventsat_0": {"mode": "payload_compress"}}

        # Default: charge
        return {"eventsat_0": {"mode": "charging"}}

"""Rule-based symbolic representation for SSA constellations."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from src.core.behaviour.controller import register
from src.core.representation import Representation


@register("rule_based_ssa")
class RuleBasedSSA(Representation):
    """Resource-aware SSA mode selector with constellation deconfliction.

    The resource thresholds mirror the source ``autops-rl`` rule baseline:
    low battery 0.3, high battery 0.8, high storage 0.7, low storage 0.3.
    The environment resolves the actual RSO contacts; this policy only selects
    modes from resources, pipeline state, and whether each satellite currently
    sees any RSO ids in its anti-nadir FOV.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.satellite_id = self.config.get("satellite_id")
        self.battery_threshold_high = float(self.config.get("battery_threshold_high", 0.8))
        self.battery_threshold_low = float(self.config.get("battery_threshold_low", 0.3))
        self.storage_threshold_high = float(self.config.get("storage_threshold_high", 0.7))
        self.storage_threshold_low = float(self.config.get("storage_threshold_low", 0.3))
        self.observe_soc = float(self.config.get("observe_soc", 0.6))
        self.compress_soc = float(self.config.get("compress_soc", 0.45))
        self._last_rationale: Optional[str] = None
        self._last_metrics: Dict[str, float] = {}

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        if not hasattr(observation, "constellation_state"):
            return {"satellites": {}, "tasks": []}

        cstate = observation.constellation_state
        satellites: Dict[str, Dict[str, Any]] = {}
        scoped_known: Set[str] = set()
        scoped_delivered: Set[str] = set()
        for sat in cstate.satellites.values():
            meta = sat.metadata or {}
            scoped_known.update(str(oid) for oid in (meta.get("ssa_known_objects", []) or []))
            scoped_delivered.update(str(oid) for oid in (meta.get("ssa_delivered_objects", []) or []))

        for sat_id, sat in cstate.satellites.items():
            res = sat.resources or {}
            meta = sat.metadata or {}
            cap_mb = float(meta.get("storage_capacity_mb", 4096.0) or 4096.0)
            data_mb = float(res.get("data_stored_mb", 0.0) or 0.0)
            visible = [str(oid) for oid in (meta.get("visible_rso_ids", []) or [])]
            known = set(str(oid) for oid in (meta.get("ssa_known_objects", []) or []))
            delivered = set(str(oid) for oid in (meta.get("ssa_delivered_objects", []) or []))
            satellites[sat_id] = {
                "satellite_id": sat_id,
                "battery_soc": float(res.get("battery_soc", 0.5) or 0.0),
                "current_mode": sat.status,
                "health_status": meta.get("health_status", "nominal"),
                "ground_pass_active": bool(meta.get("ground_pass_active", False)),
                "data_stored_mb": data_mb,
                "storage_capacity_mb": cap_mb,
                "storage_used_fraction": data_mb / cap_mb if cap_mb > 0 else 0.0,
                "obc_data_mb": float(res.get("obc_data_mb", meta.get("obc_data_mb", 0.0)) or 0.0),
                "jetson_raw_mb": float(meta.get("jetson_raw_mb", 0.0) or 0.0),
                "jetson_compressed_mb": float(meta.get("jetson_compressed_mb", 0.0) or 0.0),
                "uncompressed_observations": int(meta.get("uncompressed_observations", 0) or 0),
                "undetected_observations": int(meta.get("undetected_observations", 0) or 0),
                "daily_downlink_budget_mb": float(meta.get("daily_downlink_budget_mb", 27.0) or 27.0),
                "visible_rso_ids": visible,
                "visible_new_rso_ids": [
                    oid for oid in visible
                    if oid not in scoped_known and oid not in scoped_delivered
                ],
                "known_objects": sorted(known),
                "delivered_objects": sorted(delivered),
            }

        return {
            "satellites": satellites,
            "tasks": list(getattr(observation, "tasks", []) or []),
            "global": dict(getattr(cstate, "global_info", {}) or {}),
        }

    def select_action(self, context: Any) -> Dict[str, Any]:
        state = context.state or {}
        satellites: Dict[str, Dict[str, Any]] = state.get("satellites", {}) or {}
        if not satellites:
            self._last_rationale = "No SSA state available; charging."
            return {}

        sat_ids = sorted(satellites)
        if self.satellite_id in satellites:
            sat_ids = [str(self.satellite_id)]

        covered: Set[str] = set()
        actions: Dict[str, Dict[str, str]] = {}
        observed = 0
        for sat_id in sat_ids:
            mode, claimed, rationale = self._mode_for_satellite(
                sat_id,
                satellites[sat_id],
                covered,
                coordinated=len(sat_ids) > 1,
            )
            if mode == "payload_observe":
                observed += 1
                covered.update(claimed)
            actions[sat_id] = {"mode": mode}
            if len(sat_ids) == 1:
                self._last_rationale = rationale

        if len(sat_ids) > 1:
            self._last_rationale = (
                f"SSA coordinated rule plan for {len(sat_ids)} satellites; "
                f"{observed} observe assignments, {len(covered)} claimed RSOs."
            )
        self._last_metrics = {
            "ssa_symbolic_observe_assignments": float(observed),
            "ssa_symbolic_claimed_objects": float(len(covered)),
        }
        return actions

    def _mode_for_satellite(
        self,
        sat_id: str,
        sat: Dict[str, Any],
        covered: Set[str],
        *,
        coordinated: bool,
    ) -> tuple[str, List[str], str]:
        soc = float(sat.get("battery_soc", 0.5))
        storage_used = float(sat.get("storage_used_fraction", 0.0))
        health = sat.get("health_status", "nominal")
        pass_active = bool(sat.get("ground_pass_active", False))
        obc_mb = float(sat.get("obc_data_mb", 0.0))
        jetson_compressed_mb = float(sat.get("jetson_compressed_mb", 0.0))
        uncomp = int(sat.get("uncompressed_observations", 0))
        undetected = int(sat.get("undetected_observations", 0))
        daily_budget_mb = float(sat.get("daily_downlink_budget_mb", 27.0))
        visible_new = list(sat.get("visible_new_rso_ids", []) or [])
        candidate_targets = [oid for oid in visible_new if oid not in covered]
        known_objects = set(sat.get("known_objects", []) or [])
        delivered_objects = set(sat.get("delivered_objects", []) or [])

        if health != "nominal":
            return "safe", [], f"{sat_id}: anomaly {health}; safe."
        if soc < self.battery_threshold_low:
            return "charging", [], f"{sat_id}: battery {soc:.2f}<0.30; charging."
        if pass_active and obc_mb > 0.0:
            return "communication", [], f"{sat_id}: ground pass with {obc_mb:.1f} MB; communicating."
        if known_objects - delivered_objects:
            return "communication", [], f"{sat_id}: SSA records await ground delivery; communicating."
        if storage_used > self.storage_threshold_high and obc_mb > 0.0:
            return "communication", [], f"{sat_id}: storage {storage_used:.0%}>70%; communicating."
        if storage_used > self.storage_threshold_high and uncomp > 0:
            return "payload_compress", [], f"{sat_id}: storage pressure and raw backlog; compressing."
        if uncomp > 0:
            return "payload_compress", [], f"{sat_id}: raw observation backlog; compressing."

        pipeline_mb = obc_mb + jetson_compressed_mb
        if pipeline_mb > daily_budget_mb:
            return "charging", [], f"{sat_id}: pipeline above daily budget; holding."
        if undetected > 0:
            return "payload_detect", [], f"{sat_id}: detection backlog; detecting."
        if jetson_compressed_mb > 0.0:
            return "payload_send", [], f"{sat_id}: compressed data ready; sending to OBC."

        has_observation_opportunity = bool(candidate_targets)
        storage_allows_observe = storage_used < self.storage_threshold_high
        if (
            has_observation_opportunity
            and soc > self.battery_threshold_high
            and storage_used < self.storage_threshold_low
        ):
            return "payload_observe", candidate_targets, (
                f"{sat_id}: high battery, low storage, {len(candidate_targets)} visible RSOs; observing."
            )
        if has_observation_opportunity and soc > self.observe_soc and storage_allows_observe:
            return "payload_observe", candidate_targets, (
                f"{sat_id}: visible RSOs and resources available; observing."
            )
        if sat.get("known_objects") and coordinated and not candidate_targets:
            return "isl_share", [], f"{sat_id}: no unique target; sharing known SSA records by ISL."
        if uncomp > 0 and soc > self.compress_soc:
            return "payload_compress", [], f"{sat_id}: backlog with adequate battery; compressing."
        return "charging", [], f"{sat_id}: no priority rule matched; charging."

    def reason(self, state: Dict[str, Any], memory: Any) -> List[Dict[str, Any]]:
        satellites = (state or {}).get("satellites", {}) if isinstance(state, dict) else {}
        return [
            {
                "check": "ssa_visibility",
                "satellites": len(satellites),
                "implication": "observe_visible_unique_targets_else_resource_mode",
            }
        ]

    def get_rationale(self) -> Optional[str]:
        return self._last_rationale

    def get_metrics(self) -> Dict[str, float]:
        return dict(self._last_metrics)

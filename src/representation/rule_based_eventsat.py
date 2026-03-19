"""
Rule-Based Symbolic Representation for EventSat.

Hand-designed priority rules for single-satellite mode selection.
Registered as "rule_based_eventsat" in the emergence controller.

Priority rule chain (evaluated top-to-bottom, first match wins):
  R1  Anomaly active           → safe
  R2  Battery critically low   → charging (SoC < 0.35)
  R3  Ground pass + OBC data   → communication (downlink)
  R4  Storage pressure         → payload_compress (>70% used + uncompressed data)
  R5  Compression backlog      → payload_compress (uncomp >= 1; forces observe→compress interleave)
  R5b Pipeline saturated       → charging  (obc_mb + jetson_compressed_mb > daily_downlink_budget_mb)
                                  Source: Proposal Section 6.1 — "useful obs time limited by downlink"
  R5c Detection backlog        → payload_detect (undetected_observations > 0)
                                  PDR Section 3.2.3 Detection sub-function
  R5d Jetson data to send      → payload_send (jetson_compressed_mb > 0)
                                  RS-485 one-way TX: Jetson → OBC at 50 kbps
  R6  Good battery + storage   → payload_observe (SoC > 0.6, storage < 80%)
  R7  Uncompressed + decent SoC → payload_compress (SoC > 0.45)
  default                      → charging

State fields used (from encode_observation):
  battery_soc, ground_pass_active, data_stored_mb, obc_data_mb,
  jetson_raw_mb, jetson_compressed_mb, storage_capacity_mb,
  uncompressed_observations, undetected_observations,
  daily_downlink_budget_mb, health_status
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from src.representation.base import Representation
from src.emergence.controller import register


@register("rule_based_eventsat")
class RuleBasedEventSat(Representation):
    """Priority-based rule system for EventSat mode selection."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._last_rationale: Optional[str] = None

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
                "obc_data_mb": res.get("obc_data_mb", meta.get("obc_data_mb", 0.0)),
                "jetson_raw_mb": meta.get("jetson_raw_mb", 0.0),
                "jetson_compressed_mb": meta.get("jetson_compressed_mb", 0.0),
                "storage_capacity_mb": meta.get("storage_capacity_mb", 512.0),
                "uncompressed_observations": meta.get("uncompressed_observations", 0),
                "compression_progress": meta.get("compression_progress", 0),
                "total_observation_s": meta.get("total_observation_s", 0.0),
                "health_status": meta.get("health_status", "nominal"),
                "undetected_observations": meta.get("undetected_observations", 0),
                "daily_downlink_budget_mb": meta.get("daily_downlink_budget_mb", 27.0),
            }
        return {}

    def select_action(self, state: Dict[str, Any], memory: Any = None) -> Dict[str, Any]:
        """Apply priority rules to select the next mode."""
        if not state:
            self._last_rationale = "No state available; defaulting to charging."
            return {"eventsat_0": {"mode": "charging"}}

        soc = state.get("battery_soc", 0.5)
        pass_active = state.get("ground_pass_active", False)
        data_mb = state.get("data_stored_mb", 0.0)
        obc_mb = state.get("obc_data_mb", 0.0)
        jetson_compressed_mb = state.get("jetson_compressed_mb", 0.0)
        cap_mb = state.get("storage_capacity_mb", 512.0)
        uncomp = state.get("uncompressed_observations", 0)
        undetected = state.get("undetected_observations", 0)
        health = state.get("health_status", "nominal")
        daily_budget_mb = state.get("daily_downlink_budget_mb", 27.0)

        # Rule 1: Anomaly -> safe
        if health != "nominal":
            self._last_rationale = f"R1: Anomaly active ({health}); entering safe mode."
            return {"eventsat_0": {"mode": "safe"}}

        # Rule 2: Critically low battery -> charge
        if soc < 0.35:
            self._last_rationale = f"R2: Battery critically low (SoC={soc:.2f}<0.35); charging."
            return {"eventsat_0": {"mode": "charging"}}

        # Rule 3: Ground pass active with OBC data -> communicate (P3: only OBC is downlinkable)
        if pass_active and obc_mb > 0:
            self._last_rationale = f"R3: Ground pass active with {obc_mb:.1f} MB on OBC; downlinking."
            return {"eventsat_0": {"mode": "communication"}}

        # Rule 4: Storage pressure + uncompressed data -> compress
        if data_mb > cap_mb * 0.7 and uncomp > 0:
            self._last_rationale = f"R4: Storage at {data_mb/cap_mb:.0%} with {uncomp} uncompressed; compressing."
            return {"eventsat_0": {"mode": "payload_compress"}}

        # Rule 5: Compression backlog (lowered from >=3 to >=1 for observe→compress interleave)
        if uncomp >= 1:
            self._last_rationale = f"R5: Compression backlog ({uncomp} uncompressed); compressing."
            return {"eventsat_0": {"mode": "payload_compress"}}

        # Rule 5b: Pipeline backpressure — stop observing when pipeline exceeds daily downlink budget
        pipeline_data_mb = obc_mb + jetson_compressed_mb
        if pipeline_data_mb > daily_budget_mb:
            self._last_rationale = (
                f"R5b: Pipeline saturated ({pipeline_data_mb:.1f} MB > "
                f"{daily_budget_mb:.0f} MB budget); charging to drain."
            )
            return {"eventsat_0": {"mode": "charging"}}

        # Rule 5c: Detection backlog — run CV detection on compressed observations
        # (must detect before send: Jetson needs compressed data for CV inference)
        if undetected > 0:
            self._last_rationale = f"R5c: {undetected} undetected observations; running detection."
            return {"eventsat_0": {"mode": "payload_detect"}}

        # Rule 5d: Compressed data on Jetson → send to OBC via RS-485
        if jetson_compressed_mb > 0:
            self._last_rationale = (
                f"R5d: {jetson_compressed_mb:.2f} MB compressed on Jetson; "
                f"sending to OBC via RS-485."
            )
            return {"eventsat_0": {"mode": "payload_send"}}

        # Rule 6: Good battery + storage space -> observe
        if soc > 0.6 and data_mb < cap_mb * 0.8:
            self._last_rationale = f"R6: Battery adequate (SoC={soc:.2f}) and storage available; observing."
            return {"eventsat_0": {"mode": "payload_observe"}}

        # Rule 7: Some uncompressed + decent battery -> compress
        if uncomp > 0 and soc > 0.45:
            self._last_rationale = f"R7: {uncomp} uncompressed obs with SoC={soc:.2f}; compressing."
            return {"eventsat_0": {"mode": "payload_compress"}}

        # Default: charge
        self._last_rationale = f"R-default: No priority rule matched (SoC={soc:.2f}); charging."
        return {"eventsat_0": {"mode": "charging"}}

    def get_rationale(self) -> Optional[str]:
        """Return the rationale for the last decision."""
        return self._last_rationale

"""
Rule-Based Symbolic Representation for EventSat.

Hand-designed priority rules for single-satellite mode selection.
Registered as "rule_based_eventsat" in the emergence controller.

SDA baseline rules (evaluated top-to-bottom, first match wins):
  R1  Anomaly active           → safe
  R2  Battery critically low   → charging (SoC < 0.35)
  R3  Ground pass + OBC data   → communication (downlink)
  R4  Storage pressure         → payload_compress (>70% used + uncompressed data)
  R5  Compression backlog      → payload_compress (uncomp >= 1)
  R5b Pipeline saturated       → charging
  R5c Detection backlog        → payload_detect
  R5d Jetson data to send      → payload_send
  R6  Good battery + storage   → payload_observe (SoC > 0.6, storage < 80%)
  R7  Uncompressed + decent SoC → payload_compress (SoC > 0.45)
  default                      → charging

OODA-aware modifications (active when context.loop_type == "ooda"):
  R2-OODA  Proactive charging: SoC < 0.45 when Orient detects battery_trending_down
           Rationale: Orient's trend analysis catches decline before SDA's fixed 0.35 threshold.
  R2e-OODA Eclipse preparation: entering eclipse with SoC < 0.6 → charge
           Rationale: Boyd's "genetic heritage" — physics constraint awareness.
  R3-OODA  Urgency-based pass prioritization: urgency > 0.5 during pass → communicate
           even without OBC data (pre-emptive HK downlink).
  R6-OODA Orient-confident observation: observe at SoC > 0.50 (vs SDA's 0.60) when
           Orient's trend analysis confirms battery is stable/rising. Boyd's "analysis &
           synthesis" produces a more nuanced threshold than SDA's fixed cutoff.

State fields used (from encode_observation):
  battery_soc, ground_pass_active, data_stored_mb, obc_data_mb,
  jetson_raw_mb, jetson_compressed_mb, storage_capacity_mb,
  uncompressed_observations, undetected_observations,
  daily_downlink_budget_mb, health_status
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from src.representation.base import Representation
from src.behaviour.controller import register

if TYPE_CHECKING:
    from src.decision_procedure.context import DecisionContext


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
                "storage_capacity_mb": meta.get("storage_capacity_mb", 4096.0),
                "uncompressed_observations": meta.get("uncompressed_observations", 0),
                "compression_progress": meta.get("compression_progress", 0),
                "total_observation_s": meta.get("total_observation_s", 0.0),
                "health_status": meta.get("health_status", "nominal"),
                "undetected_observations": meta.get("undetected_observations", 0),
                "daily_downlink_budget_mb": meta.get("daily_downlink_budget_mb", 27.0),
            }
        return {}

    def select_action(self, context: DecisionContext) -> Dict[str, Any]:
        """Apply priority rules to select the next mode.

        When context.loop_type == "ooda", Orient enrichments modify rule
        thresholds and introduce proactive decisions that the SDA baseline
        cannot make.
        """
        from src.decision_procedure.context import DecisionContext  # runtime import

        state = context.state
        enrichments = context.enrichments
        is_ooda = context.loop_type == "ooda" and bool(enrichments)

        if not state:
            self._last_rationale = "No state available; defaulting to charging."
            return {"eventsat_0": {"mode": "charging"}}

        soc = state.get("battery_soc", 0.5)
        pass_active = state.get("ground_pass_active", False)
        data_mb = state.get("data_stored_mb", 0.0)
        obc_mb = state.get("obc_data_mb", 0.0)
        jetson_compressed_mb = state.get("jetson_compressed_mb", 0.0)
        cap_mb = state.get("storage_capacity_mb", 4096.0)
        uncomp = state.get("uncompressed_observations", 0)
        undetected = state.get("undetected_observations", 0)
        health = state.get("health_status", "nominal")
        daily_budget_mb = state.get("daily_downlink_budget_mb", 27.0)

        # Rule 1: Anomaly -> safe
        if health != "nominal":
            self._last_rationale = f"R1: Anomaly active ({health}); entering safe mode."
            return {"eventsat_0": {"mode": "safe"}}

        # --- OODA R2e: Eclipse preparation (Boyd's genetic heritage) ---
        # Orient detects sunlight transition; proactively charge before eclipse
        # drains battery. SDA cannot do this — it lacks transition awareness.
        if is_ooda and enrichments.get("entered_eclipse", False) and soc < 0.6:
            self._last_rationale = (
                f"R2e-OODA: Entering eclipse with SoC={soc:.2f}<0.60; "
                f"proactive charging (Orient detected transition)."
            )
            return {"eventsat_0": {"mode": "charging"}}

        # --- OODA R2: Proactive charging (Orient trend-aware) ---
        # Orient's trend analysis detects declining battery before it reaches
        # SDA's fixed 0.35 critical threshold. Charge earlier at 0.45.
        if is_ooda and enrichments.get("battery_trending_down", False):
            urgency = enrichments.get("urgency", 0.0)
            if urgency > 0 and soc < 0.45:
                self._last_rationale = (
                    f"R2-OODA: Battery trending down (SoC={soc:.2f}<0.45, "
                    f"urgency={urgency:.2f}); proactive charging."
                )
                return {"eventsat_0": {"mode": "charging"}}

        # Rule 2: Critically low battery -> charge (SDA baseline)
        if soc < 0.35:
            self._last_rationale = f"R2: Battery critically low (SoC={soc:.2f}<0.35); charging."
            return {"eventsat_0": {"mode": "charging"}}

        # --- OODA R3: Urgency-based pass prioritization ---
        # When Orient signals high urgency during a pass, prioritize comms
        # even without OBC data (pre-emptive HK downlink / command uplink).
        if is_ooda and pass_active and enrichments.get("urgency", 0.0) > 0.5 and obc_mb == 0:
            self._last_rationale = (
                f"R3-OODA: Pass active with urgency={enrichments['urgency']:.2f}>0.50; "
                f"prioritizing communication (Orient situation: "
                f"{enrichments.get('situation_class', 'unknown')})."
            )
            return {"eventsat_0": {"mode": "communication"}}

        # Rule 3: Ground pass active with OBC data -> communicate
        if pass_active and obc_mb > 0:
            self._last_rationale = f"R3: Ground pass active with {obc_mb:.1f} MB on OBC; downlinking."
            return {"eventsat_0": {"mode": "communication"}}

        # Rule 4: Storage pressure + uncompressed data -> compress
        if data_mb > cap_mb * 0.7 and uncomp > 0:
            self._last_rationale = f"R4: Storage at {data_mb/cap_mb:.0%} with {uncomp} uncompressed; compressing."
            return {"eventsat_0": {"mode": "payload_compress"}}

        # --- OODA R5: Batch observations when Orient confirms safety ---
        # SDA compresses immediately after every observation (uncomp >= 1).
        # OODA's Orient analysis enables batching: when battery trend is stable/rising,
        # no imminent pass, and pipeline isn't saturated, allow 2 observations before
        # compressing. Reduces ADCS mode-switching overhead (135s settling per switch).
        # Boyd's "analysis & synthesis" — Orient's situation awareness allows more
        # efficient pipeline utilization than SDA's conservative 1:1 interleave.
        if is_ooda and uncomp >= 1 and uncomp <= 2:
            if (
                not enrichments.get("battery_trending_down", False)
                and soc > 0.55
                and not pass_active
                and enrichments.get("urgency", 0.0) < 0.5
            ):
                # Safe to defer compression — fall through to observe (R6/R6-OODA)
                pass
            else:
                self._last_rationale = (
                    f"R5-OODA: Compression backlog ({uncomp} uncompressed, "
                    f"conditions not safe for batching); compressing."
                )
                return {"eventsat_0": {"mode": "payload_compress"}}
        elif uncomp >= 1:
            # Rule 5: SDA baseline — compress immediately
            self._last_rationale = f"R5: Compression backlog ({uncomp} uncompressed); compressing."
            return {"eventsat_0": {"mode": "payload_compress"}}

        # Rule 5b: Pipeline backpressure
        pipeline_data_mb = obc_mb + jetson_compressed_mb
        if pipeline_data_mb > daily_budget_mb:
            self._last_rationale = (
                f"R5b: Pipeline saturated ({pipeline_data_mb:.1f} MB > "
                f"{daily_budget_mb:.0f} MB budget); charging to drain."
            )
            return {"eventsat_0": {"mode": "charging"}}

        # Rule 5c: Detection backlog
        if undetected > 0:
            self._last_rationale = f"R5c: {undetected} undetected observations; running detection."
            return {"eventsat_0": {"mode": "payload_detect"}}

        # Rule 5d: Compressed data on Jetson → send to OBC
        if jetson_compressed_mb > 0:
            self._last_rationale = (
                f"R5d: {jetson_compressed_mb:.2f} MB compressed on Jetson; "
                f"sending to OBC via RS-485."
            )
            return {"eventsat_0": {"mode": "payload_send"}}

        # --- OODA R6: Orient-confident observation at lower SoC ---
        # Orient's trend analysis confirms battery is stable or rising, giving
        # confidence to observe at SoC > 0.50 instead of SDA's conservative 0.60.
        # This is the key differentiation: OODA's situation awareness allows
        # more aggressive use of the observation window.
        # Boyd's "analysis & synthesis" — combining current state with trend data
        # produces a more nuanced decision than SDA's fixed threshold.
        if is_ooda and data_mb < cap_mb * 0.8:
            battery_trending_down = enrichments.get("battery_trending_down", False)
            if not battery_trending_down and soc > 0.50:
                self._last_rationale = (
                    f"R6-OODA: Orient confirms battery stable/rising "
                    f"(SoC={soc:.2f}>0.50, trend_down=False); observing."
                )
                return {"eventsat_0": {"mode": "payload_observe"}}

        # Rule 6: Good battery + storage space -> observe (SDA baseline: SoC > 0.6)
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

    def reason(self, state: Dict[str, Any], memory: Any) -> List[Dict[str, Any]]:
        """Evaluate decision factors before acting (ReAct Thought step).

        Returns a structured trace of key rule evaluation factors so the
        ReAct grounding step can validate the planned action.
        """
        if not state:
            return [{"check": "state", "value": None, "implication": "no_state_default_charging"}]

        trace: List[Dict[str, Any]] = []

        soc = state.get("battery_soc", 0.5)
        health = state.get("health_status", "nominal")
        pass_active = state.get("ground_pass_active", False)
        uncomp = state.get("uncompressed_observations", 0)
        undetected = state.get("undetected_observations", 0)
        obc_mb = state.get("obc_data_mb", 0.0)
        jetson_compressed_mb = state.get("jetson_compressed_mb", 0.0)
        data_mb = state.get("data_stored_mb", 0.0)
        cap_mb = state.get("storage_capacity_mb", 4096.0)
        daily_budget_mb = state.get("daily_downlink_budget_mb", 27.0)

        if health != "nominal":
            trace.append({"check": "health", "value": health, "implication": "safe_mode_required", "rule": "R1"})
        if soc < 0.35:
            trace.append({"check": "battery", "value": soc, "implication": "charging_required", "rule": "R2"})
        elif soc < 0.45:
            trace.append({"check": "battery", "value": soc, "implication": "charging_preferred", "rule": "R2"})
        if pass_active and obc_mb > 0:
            trace.append({"check": "pass_opportunity", "value": obc_mb, "implication": "communicate", "rule": "R3"})
        if uncomp >= 1:
            trace.append({"check": "compression_backlog", "value": uncomp, "implication": "compress", "rule": "R5"})
        if undetected > 0:
            trace.append({"check": "detection_backlog", "value": undetected, "implication": "detect", "rule": "R5c"})
        if jetson_compressed_mb > 0:
            trace.append({"check": "jetson_data", "value": jetson_compressed_mb, "implication": "send_to_obc", "rule": "R5d"})
        pipeline_mb = obc_mb + jetson_compressed_mb
        if pipeline_mb > daily_budget_mb:
            trace.append({"check": "pipeline_saturated", "value": pipeline_mb, "implication": "hold_charge", "rule": "R5b"})
        if soc > 0.60 and data_mb < cap_mb * 0.8:
            trace.append({"check": "observe_conditions", "value": soc, "implication": "observe_viable", "rule": "R6"})

        if not trace:
            trace.append({"check": "default", "value": soc, "implication": "charging"})

        return trace

    def get_rationale(self) -> Optional[str]:
        """Return the rationale for the last decision."""
        return self._last_rationale

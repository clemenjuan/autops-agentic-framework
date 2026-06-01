"""
OODA (Observe-Orient-Decide-Act) Decision Loop.

Boyd's OODA loop (1987) with feedback, grounded in:
  - Richards, C. (2018), "Boyd's OODA Loop (It's Not What You Think)"
    [Zotero: H6GKNERB]. Primary modern exposition; clarifies that Observe
    and Orient dominate the cycle (not Decide/Act) and that the loop must
    include cross-phase feedback, not just linear progression.
  - Miller, Hasbrouck & Udrea (2021), "Development of Human-Machine
    Collaborative Systems Using OODA Loops", ASCEND 2021.
  - Hartmann et al. (2024), "METIS: An AI Assistant Enabling Autonomous
    Spacecraft Operations", IEEE Aerospace Conference.

Key differences from SDA:
  1. Observe phase classifies operational situation into regimes
     (cf. METIS Monitoring Agent's telemetry categories).
  2. Orient phase implements simplified Case-Based Reasoning
     (Retrieve→Reuse→Revise→Retain) + trend analysis + urgency scoring.
     Synthesizes Boyd's "cultural traditions" (mission rules), "genetic
     heritage" (physics constraints), "new information" (telemetry), and
     "previous experience" (memory history).
  3. Feedback loops: Orient→Observe (attention guidance for next cycle),
     Orient→Act (urgency-based bypass).
  4. Memory: actively reads and updates (SDA ignores memory).
  5. OODA loop duration is a gamma-distributed TPM (Miller et al.).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from src.decision_procedure.base import DecisionProcedure
from src.decision_procedure.context import DecisionContext
from src.representation.base import Representation


# -- Situation classes (cf. METIS 7 telemetry categories) -------------------
# Ordered by priority (highest first) for classification.
SITUATION_ANOMALY = "anomaly_active"
SITUATION_BATTERY_CRITICAL = "battery_critical"
SITUATION_PASS_OPPORTUNITY = "pass_opportunity"
SITUATION_STORAGE_CRITICAL = "storage_critical"
SITUATION_ECLIPSE_CHARGING = "eclipse_charging"
SITUATION_DATA_PIPELINE = "data_pipeline_active"
SITUATION_NOMINAL = "nominal"


class OODALoop(DecisionProcedure):
    """Observe-Orient-Decide-Act decision loop (Boyd, 1987).

    Implements Boyd's four phases with feedback loops, using METIS-inspired
    situation classification in Observe and simplified Case-Based Reasoning
    in Orient.
    """

    def __init__(self, config: Dict[str, Any], representation: Representation) -> None:
        self.config = config
        self.representation = representation

        # Config parameters
        self._orient_history_window: int = config.get("orient_history_window", 10)
        self._max_orient_iterations: int = config.get("max_orient_iterations", 1)
        self._urgency_bypass_threshold: float = config.get(
            "urgency_bypass_threshold", 0.9
        )

        # Metrics state
        self._last_total_latency: float = 0.0
        self._last_observe_latency: float = 0.0
        self._last_orient_latency: float = 0.0
        self._last_decide_latency: float = 0.0
        self._last_orient_iterations: int = 0
        self._last_urgency: float = 0.0
        self._last_cases_retrieved: int = 0
        self._total_steps: int = 0
        self._last_has_rationale: bool = False

        # Feedback state (Boyd's implicit guidance & control)
        self._attention_guidance: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # DecisionProcedure interface
    # ------------------------------------------------------------------

    def process(
        self, observation: Any, memory: Any
    ) -> Tuple[Dict[str, Any], Any]:
        """Execute one OODA cycle: Observe → Orient → Decide → Act."""
        t_total = time.perf_counter()

        # === OBSERVE (cf. METIS Monitoring Agent) ===
        t0 = time.perf_counter()
        raw_obs = observation
        if hasattr(observation, "local_state") and isinstance(
            observation.local_state, dict
        ):
            raw_obs = observation.local_state.get("full_observation", observation)
        encoded = self.representation.encode_observation(raw_obs)
        situation_class = self._classify_situation(encoded)
        self._last_observe_latency = time.perf_counter() - t0

        # === ORIENT (cf. METIS Reasoning Agent + Boyd's Analysis & Synthesis) ===
        t0 = time.perf_counter()
        oriented_state, orient_iters = self._orient(
            encoded, situation_class, memory
        )
        self._last_orient_latency = time.perf_counter() - t0
        self._last_orient_iterations = orient_iters

        # === DECIDE ===
        t0 = time.perf_counter()
        context = DecisionContext(
            state=encoded,
            loop_type="ooda",
            memory=memory,
            enrichments={
                "situation_class": situation_class,
                "urgency": self._last_urgency,
                "battery_trend": oriented_state.get("orient_battery_trend", 0.0),
                "battery_trending_down": oriented_state.get(
                    "orient_battery_trending_down", False
                ),
                "data_pressure": oriented_state.get("orient_data_pressure", 0.0),
                "competing_priorities": oriented_state.get(
                    "orient_competing_priorities", []
                ),
                "similar_case": oriented_state.get("orient_similar_case_outcome"),
                "attention_guidance": oriented_state.get(
                    "orient_attention_guidance"
                ),
                "anomaly_is_new": oriented_state.get(
                    "orient_anomaly_is_new", False
                ),
                "sunlight_transition": oriented_state.get(
                    "orient_sunlight_transition", False
                ),
                "entered_eclipse": oriented_state.get(
                    "orient_entered_eclipse", False
                ),
            },
            loop_metadata={
                "orient_iterations": self._last_orient_iterations,
                "orient_latency_s": self._last_orient_latency,
            },
        )
        action = self.representation.select_action(context)
        self._last_decide_latency = time.perf_counter() - t0

        # === ACT + FEEDBACK (Boyd's implicit guidance & control) ===
        updated_memory = self._act_and_update_memory(
            memory, encoded, oriented_state, action, situation_class
        )

        # Metrics bookkeeping
        self._last_total_latency = time.perf_counter() - t_total
        self._total_steps += 1
        self._last_has_rationale = (
            hasattr(self.representation, "get_rationale")
            and self.representation.get_rationale() is not None
        )

        return action, updated_memory

    def get_metrics(self) -> Dict[str, Any]:
        """Return OODA-specific metrics (Miller et al. TPM framework)."""
        rationale = ""
        if hasattr(self.representation, "get_rationale"):
            rationale = self.representation.get_rationale() or ""
        return {
            # Total OODA loop duration (gamma-distributed TPM)
            "decision_latency_s": self._last_total_latency,
            # Per-phase breakdown
            "observe_latency_s": self._last_observe_latency,
            "orient_latency_s": self._last_orient_latency,
            "decide_latency_s": self._last_decide_latency,
            # Orient quality metrics
            "orient_iterations": float(self._last_orient_iterations),
            "orient_urgency": self._last_urgency,
            "orient_cases_retrieved": float(self._last_cases_retrieved),
            # Standard
            "total_decisions": float(self._total_steps),
            "has_rationale": float(self._last_has_rationale),
            "rationale": rationale,
        }

    def reset(self) -> None:
        """Reset internal state at episode start."""
        self._last_total_latency = 0.0
        self._last_observe_latency = 0.0
        self._last_orient_latency = 0.0
        self._last_decide_latency = 0.0
        self._last_orient_iterations = 0
        self._last_urgency = 0.0
        self._last_cases_retrieved = 0
        self._total_steps = 0
        self._last_has_rationale = False
        self._attention_guidance = None

    def get_name(self) -> str:
        return "OODALoop"

    # ------------------------------------------------------------------
    # OBSERVE: Situation Classification
    # ------------------------------------------------------------------

    def _classify_situation(self, state: Dict[str, Any]) -> str:
        """Classify operational situation into one of the defined regimes.

        Inspired by METIS Monitoring Agent's 7 telemetry categories
        (Hartmann et al. 2024), adapted for EventSat operational regimes.
        Evaluated in priority order (highest first).
        """
        if not state:
            return SITUATION_NOMINAL

        health = state.get("health_status", "nominal")
        if health != "nominal":
            return SITUATION_ANOMALY

        soc = state.get("battery_soc", 0.5)
        if soc < 0.20:
            return SITUATION_BATTERY_CRITICAL

        pass_active = state.get("ground_pass_active", False)
        obc_mb = state.get("obc_data_mb", 0.0)
        if pass_active and obc_mb > 0:
            return SITUATION_PASS_OPPORTUNITY

        jetson_raw_mb = state.get("jetson_raw_mb", 0.0)
        cap_mb = state.get("storage_capacity_mb", 512.0)
        if cap_mb > 0 and jetson_raw_mb > cap_mb * 0.8:
            return SITUATION_STORAGE_CRITICAL

        in_sunlight = state.get("in_sunlight", True)
        if not in_sunlight and soc < 0.50:
            return SITUATION_ECLIPSE_CHARGING

        uncomp = state.get("uncompressed_observations", 0)
        undetected = state.get("undetected_observations", 0)
        if uncomp > 0 or undetected > 0:
            return SITUATION_DATA_PIPELINE

        return SITUATION_NOMINAL

    # ------------------------------------------------------------------
    # ORIENT: Case-Based Reasoning + Trend Analysis + Urgency
    # ------------------------------------------------------------------

    def _orient(
        self,
        encoded_state: Dict[str, Any],
        situation_class: str,
        memory: Any,
    ) -> Tuple[Dict[str, Any], int]:
        """Build situation assessment (Boyd's Analysis & Synthesis).

        Implements simplified CBR cycle (cf. METIS Reasoning Agent):
          Retrieve → Reuse → (Revise deferred to Act phase) → (Retain in Act)

        Returns:
            Tuple of (oriented_state_dict, orient_iterations).
        """
        oriented = dict(encoded_state)
        oriented["orient_situation_class"] = situation_class

        iterations = 0
        for _ in range(self._max_orient_iterations):
            iterations += 1

            # --- CBR Retrieve: find similar past situations ---
            similar_case = self._retrieve_similar_case(
                situation_class, encoded_state, memory
            )
            oriented["orient_similar_case_outcome"] = similar_case

            # --- Trend Analysis (Boyd's analysis & synthesis) ---
            trends = self._analyze_trends(encoded_state, memory)
            oriented.update(trends)

            # --- Urgency Scoring ---
            urgency = self._compute_urgency(encoded_state, situation_class, trends)
            oriented["orient_urgency"] = urgency
            self._last_urgency = urgency

            # --- Competing Priorities ---
            competing = self._detect_competing_priorities(encoded_state)
            oriented["orient_competing_priorities"] = competing

            # --- Attention Guidance (Boyd's Orient→Observe feedback) ---
            oriented["orient_attention_guidance"] = self._generate_attention_guidance(
                situation_class, trends, urgency
            )

            # If situation is unambiguous, no need to re-orient
            if len(competing) < 2:
                break

        return oriented, iterations

    def _retrieve_similar_case(
        self,
        situation_class: str,
        current_state: Dict[str, Any],
        memory: Any,
    ) -> Optional[Dict[str, Any]]:
        """CBR Retrieve: find the most similar past situation from memory.

        Searches memory history for entries with matching situation class,
        then selects the closest by battery SoC proximity.
        """
        if memory is None or not hasattr(memory, "query"):
            self._last_cases_retrieved = 0
            return None

        history: List[Dict[str, Any]] = memory.query("history") or []
        if not history:
            self._last_cases_retrieved = 0
            return None

        recent = history[-self._orient_history_window:]
        current_soc = current_state.get("battery_soc", 0.5)

        # Filter by situation class match, then rank by state proximity
        candidates = [
            h for h in recent
            if h.get("orient_situation_class") == situation_class
        ]
        self._last_cases_retrieved = len(candidates)

        if not candidates:
            # Fallback: closest by SoC regardless of class
            best = min(
                recent,
                key=lambda h: abs(h.get("battery_soc", 0.5) - current_soc),
            )
            self._last_cases_retrieved = 1
            return {
                "matched_class": False,
                "case_situation": best.get("orient_situation_class", "unknown"),
                "case_action": best.get("last_action"),
                "case_soc": best.get("battery_soc"),
            }

        best = min(
            candidates,
            key=lambda h: abs(h.get("battery_soc", 0.5) - current_soc),
        )
        return {
            "matched_class": True,
            "case_situation": situation_class,
            "case_action": best.get("last_action"),
            "case_soc": best.get("battery_soc"),
        }

    def _analyze_trends(
        self, current_state: Dict[str, Any], memory: Any
    ) -> Dict[str, Any]:
        """Analyze trends from memory history (Boyd's analysis & synthesis).

        Maps to Boyd's concept of synthesizing "previous experience" with
        "new information" through analysis.
        """
        trends: Dict[str, Any] = {}

        if memory is None or not hasattr(memory, "query"):
            return trends

        history: List[Dict[str, Any]] = memory.query("history") or []
        if not history:
            return trends

        recent = history[-self._orient_history_window:]

        # Battery trend (Boyd's "genetic heritage" — physics constraint)
        recent_socs = [
            h.get("battery_soc") for h in recent if "battery_soc" in h
        ]
        if len(recent_socs) >= 2:
            soc_delta = recent_socs[-1] - recent_socs[0]
            trends["orient_battery_trend"] = soc_delta
            trends["orient_battery_trending_down"] = soc_delta < -0.01
        else:
            trends["orient_battery_trend"] = 0.0
            trends["orient_battery_trending_down"] = False

        # Data pipeline pressure
        recent_obc = [
            h.get("obc_data_mb", 0.0) for h in recent if "obc_data_mb" in h
        ]
        if len(recent_obc) >= 2:
            trends["orient_data_pressure"] = recent_obc[-1] - recent_obc[0]
        else:
            trends["orient_data_pressure"] = 0.0

        # Sunlight transition detection
        recent_sun = [h.get("in_sunlight") for h in recent if "in_sunlight" in h]
        if recent_sun:
            current_sun = current_state.get("in_sunlight", True)
            last_sun = recent_sun[-1]
            if current_sun != last_sun:
                trends["orient_sunlight_transition"] = True
                trends["orient_entered_sunlight"] = current_sun and not last_sun
                trends["orient_entered_eclipse"] = not current_sun and last_sun
            else:
                trends["orient_sunlight_transition"] = False
                trends["orient_entered_sunlight"] = False
                trends["orient_entered_eclipse"] = False

        # Anomaly context
        health = current_state.get("health_status", "nominal")
        if health != "nominal":
            recent_health = [
                h.get("health_status", "nominal") for h in recent
            ]
            trends["orient_anomaly_is_new"] = all(
                h == "nominal" for h in recent_health[-3:]
            )
        else:
            trends["orient_anomaly_is_new"] = False

        return trends

    def _compute_urgency(
        self,
        state: Dict[str, Any],
        situation_class: str,
        trends: Dict[str, Any],
    ) -> float:
        """Compute urgency score [0, 1].

        Higher urgency = more pressing situation. Combines situation class
        severity with trend signals.
        """
        urgency = 0.0

        # Situation class base urgency
        class_urgency = {
            SITUATION_ANOMALY: 0.9,
            SITUATION_BATTERY_CRITICAL: 0.8,
            SITUATION_PASS_OPPORTUNITY: 0.6,
            SITUATION_STORAGE_CRITICAL: 0.5,
            SITUATION_ECLIPSE_CHARGING: 0.4,
            SITUATION_DATA_PIPELINE: 0.2,
            SITUATION_NOMINAL: 0.0,
        }
        urgency = class_urgency.get(situation_class, 0.0)

        # Modifiers from trends
        if trends.get("orient_battery_trending_down", False):
            urgency = min(1.0, urgency + 0.1)

        soc = state.get("battery_soc", 0.5)
        if soc < 0.35:
            urgency = min(1.0, urgency + 0.15)

        return urgency

    def _detect_competing_priorities(
        self, state: Dict[str, Any]
    ) -> List[str]:
        """Detect conflicting operational demands.

        When multiple priorities compete, Orient may need to re-iterate
        (Boyd's feedback within Orient).
        """
        competing: List[str] = []
        soc = state.get("battery_soc", 0.5)
        pass_active = state.get("ground_pass_active", False)
        uncomp = state.get("uncompressed_observations", 0)
        obc_mb = state.get("obc_data_mb", 0.0)

        if soc < 0.50 and pass_active and obc_mb > 0:
            competing.append("charge_vs_downlink")

        if pass_active and uncomp > 0 and obc_mb > 0:
            competing.append("downlink_vs_compress")

        if soc < 0.50 and uncomp > 0:
            competing.append("charge_vs_process")

        return competing

    def _generate_attention_guidance(
        self,
        situation_class: str,
        trends: Dict[str, Any],
        urgency: float,
    ) -> Dict[str, Any]:
        """Generate attention guidance for next Observe cycle.

        Boyd's implicit guidance & control: Orient feeds back to Observe,
        directing what the agent should monitor more closely next step.
        """
        guidance: Dict[str, Any] = {
            "priority_monitor": [],
            "urgency_level": urgency,
        }

        if situation_class == SITUATION_ANOMALY:
            guidance["priority_monitor"].append("health_status")
            guidance["priority_monitor"].append("battery_soc")
        elif situation_class == SITUATION_BATTERY_CRITICAL:
            guidance["priority_monitor"].append("battery_soc")
            guidance["priority_monitor"].append("in_sunlight")
        elif situation_class == SITUATION_PASS_OPPORTUNITY:
            guidance["priority_monitor"].append("ground_pass_active")
            guidance["priority_monitor"].append("obc_data_mb")

        if trends.get("orient_battery_trending_down", False):
            if "battery_soc" not in guidance["priority_monitor"]:
                guidance["priority_monitor"].append("battery_soc")

        return guidance

    # ------------------------------------------------------------------
    # ACT: Memory Update + CBR Retain (Boyd's feedback)
    # ------------------------------------------------------------------

    def _act_and_update_memory(
        self,
        memory: Any,
        encoded_state: Dict[str, Any],
        oriented_state: Dict[str, Any],
        action: Dict[str, Any],
        situation_class: str,
    ) -> Any:
        """Act phase: update memory with current situation for future CBR.

        Implements:
          - CBR Retain: store situation + action for future Retrieve
          - Boyd's feedback: set attention guidance for next Observe
        """
        if memory is None or not hasattr(memory, "update"):
            return memory

        # Store encoded state + orient metadata for CBR Retain.
        # FixedMemory.update("constellation_state", ...) auto-pushes
        # the previous state into the history sliding window.
        enriched_state = dict(encoded_state)
        enriched_state["orient_situation_class"] = situation_class
        enriched_state["last_action"] = action
        memory.update("constellation_state", enriched_state)

        # Store orient assessment in custom slot for introspection
        orient_data = {
            k: v for k, v in oriented_state.items() if k.startswith("orient_")
        }
        memory.update("custom", {
            "last_orient_assessment": orient_data,
            "last_action": action,
            "orient_iterations": self._last_orient_iterations,
        })

        # Set attention guidance for next cycle (Boyd's Orient→Observe)
        self._attention_guidance = oriented_state.get("orient_attention_guidance")

        return memory

"""
LLM Hybrid Representation for EventSat.

Hybrid representation (subsymbolic LLM + symbolic safety constraints).
The LLM reasons about mode selection; symbolic grounding validates the
output against physical constraints and retries on invalid responses.

Works with all 3 decision loops (SDA, OODA, ReAct) and all 3 operations
paradigms (AH, AG, CG) — fully orthogonal in the morphological matrix.

Papers:
- Rodriguez-Fernandez et al. (2024) — LLM prompt design for sat ops
- Li (2025) — ReAct LLM agent architecture for satellite operations

Registered as "llm_eventsat" in the emergence controller.
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.emergence.controller import register
from src.representation.base import Representation
from src.representation.llm_client import LLMClient
from src.representation.llm_prompts import (
    SYSTEM_PROMPT,
    format_reasoning_prompt,
    format_state_prompt,
)

if TYPE_CHECKING:
    from src.decision_loop.context import DecisionContext

logger = logging.getLogger(__name__)

# Valid EventSat modes for symbolic grounding
VALID_MODES = frozenset({
    "charging",
    "communication",
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "safe",
})


@register("llm_eventsat")
class LLMEventSat(Representation):
    """LLM-based hybrid representation for EventSat mode selection.

    The LLM provides the subsymbolic reasoning core. Symbolic constraints
    validate and ground the output:
    - Mode must be one of the 7 valid EventSat modes.
    - Anomaly active → force safe mode (cannot be overridden).
    - Response must be valid JSON with 'mode' field.

    On LLM failure or invalid output, falls back to a safe symbolic default.
    """

    # Maximum LLM retries on invalid/unparseable output
    MAX_RETRIES: int = 2

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._client = LLMClient(config)
        self._last_rationale: Optional[str] = None
        self._last_raw_response: Optional[str] = None
        self._last_parse_retries: int = 0
        self._grounding_overrides: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        """Extract decision-relevant features (same as rule-based for comparability)."""
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

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        """Select mode via LLM reasoning + symbolic grounding.

        Flow:
        1. Format state into structured prompt
        2. Call LLM for mode selection + rationale
        3. Parse JSON response
        4. Validate against symbolic constraints (grounding)
        5. Retry on failure, fall back to safe default
        """
        state = context.state
        enrichments = context.enrichments

        # Symbolic safety: anomaly always → safe (no LLM override)
        health = state.get("health_status", "nominal")
        if health != "nominal":
            self._last_rationale = f"Symbolic safety: anomaly active ({health}); forced safe mode."
            self._grounding_overrides += 1
            return {"eventsat_0": {"mode": "safe"}}

        if not state:
            self._last_rationale = "No state available; defaulting to charging."
            return {"eventsat_0": {"mode": "charging"}}

        # Build prompt
        user_prompt = format_state_prompt(state, enrichments)

        # LLM call with retries
        mode = None
        rationale = None
        retries = 0

        for attempt in range(1 + self.MAX_RETRIES):
            try:
                raw = self._client.generate(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    json_mode=True,
                )
                self._last_raw_response = raw
                parsed = self._parse_response(raw)
                mode = parsed.get("mode")
                rationale = parsed.get("rationale", "")

                # Symbolic grounding: validate mode
                if mode in VALID_MODES:
                    break
                else:
                    logger.warning(
                        "LLM returned invalid mode '%s' (attempt %d/%d)",
                        mode, attempt + 1, 1 + self.MAX_RETRIES,
                    )
                    mode = None
                    retries += 1

            except Exception as e:
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1, 1 + self.MAX_RETRIES, e,
                )
                retries += 1

        self._last_parse_retries = retries

        # Fallback if LLM failed entirely
        if mode is None:
            mode = self._symbolic_fallback(state)
            rationale = f"LLM failed after {retries} retries; symbolic fallback selected '{mode}'."
            self._grounding_overrides += 1
            logger.warning("LLM fallback to symbolic: %s", rationale)

        # Additional symbolic grounding checks
        mode = self._apply_grounding(mode, state)

        self._last_rationale = f"LLM: {rationale}" if rationale else f"LLM selected: {mode}"
        return {"eventsat_0": {"mode": mode}}

    def reason(self, state: Dict[str, Any], memory: Any) -> List[Dict[str, Any]]:
        """Produce structured reasoning steps for ReAct thought phase.

        Calls the LLM with a reasoning prompt and parses the structured output.
        Falls back to empty trace on failure (base class contract).
        """
        if not state:
            return [{"check": "state", "value": None, "implication": "no_state_default_charging"}]

        prompt = format_reasoning_prompt(state, memory)

        try:
            raw = self._client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                json_mode=True,
            )
            # Try to parse as JSON array
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
            return []
        except Exception as e:
            logger.debug("LLM reasoning parse failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Optional extension points
    # ------------------------------------------------------------------

    def get_rationale(self) -> Optional[str]:
        """Return the rationale for the last decision."""
        return self._last_rationale

    def get_metrics(self) -> Dict[str, float]:
        """Return LLM-specific metrics."""
        client_metrics = self._client.get_metrics()
        client_metrics["llm_grounding_overrides"] = float(self._grounding_overrides)
        client_metrics["llm_last_parse_retries"] = float(self._last_parse_retries)
        return client_metrics

    def get_name(self) -> str:
        return "LLMEventSat"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse LLM response as JSON, handling common formatting issues."""
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        return json.loads(text)

    def _symbolic_fallback(self, state: Dict[str, Any]) -> str:
        """Minimal symbolic fallback when LLM is unavailable.

        Conservative priority: safe if anomaly, charge if low battery,
        communicate if pass active with data, else charge.
        """
        soc = state.get("battery_soc", 0.5)
        if soc < 0.35:
            return "charging"
        if state.get("ground_pass_active", False) and state.get("obc_data_mb", 0.0) > 0:
            return "communication"
        return "charging"

    def _apply_grounding(self, mode: str, state: Dict[str, Any]) -> str:
        """Apply symbolic grounding constraints to the LLM's chosen mode.

        Returns the mode (possibly overridden) after safety checks.
        """
        # Cannot communicate without an active ground pass
        if mode == "communication" and not state.get("ground_pass_active", False):
            logger.debug("Grounding: overriding 'communication' — no active pass.")
            self._grounding_overrides += 1
            return "charging"

        # Very low battery override (hard constraint)
        soc = state.get("battery_soc", 0.5)
        if soc < 0.20 and mode != "charging":
            logger.debug("Grounding: overriding '%s' — SoC %.2f below hard limit.", mode, soc)
            self._grounding_overrides += 1
            return "charging"

        return mode

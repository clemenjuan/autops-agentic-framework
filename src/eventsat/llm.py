"""
LLM Hybrid Representation for EventSat.

Hybrid representation (subsymbolic LLM + symbolic safety constraints).
The LLM reasons about mode selection; symbolic grounding validates the
output against physical constraints and retries on invalid responses.

Works with the fixed SDA decision driver across the EventSat operations
paradigms used by the benchmark.

Learned behaviour variants (behaviour_config.mechanism):
- ``hand_designed`` (default): fixed SYSTEM_PROMPT + FixedMemory.
- ``prompt_optimized``: loads an offline-optimised system prompt from
  ``data/trained_prompts/<experiment_id>/prompt.txt``; falls back to default
  with a warning if the file does not exist. Run ``autops train <config>``
  (which calls PromptOptimizer) to generate it. FixedMemory invariant preserved.

Papers:
- Rodriguez-Fernandez et al. (2024) — LLM prompt design for sat ops
- Li (2025) — LLM agent architecture for satellite operations
- Khattab et al. (2023) [DSPy] — prompt optimization reference

Registered as "llm_eventsat" in the representation factory.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.behaviour.controller import register
from src.core.representation import Representation
from src.core.llm_client import LLMClient
from src.eventsat.llm_prompts import (
    SYSTEM_PROMPT,
    format_reasoning_prompt,
    format_state_prompt,
)

if TYPE_CHECKING:
    from src.core.decision_procedure.context import DecisionContext

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
    _TRAINED_PROMPTS_DIR: str = "data/trained_prompts"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self._client = LLMClient(cfg)

        # Learned behaviour mechanism
        behaviour_cfg: Dict[str, Any] = cfg.get("behaviour_config", {})
        mechanism: str = behaviour_cfg.get("mechanism", "hand_designed")
        experiment_id: str = cfg.get("experiment_id", "")
        self._system_prompt: str = self._resolve_system_prompt(
            mechanism, experiment_id, behaviour_cfg
        )

        self._last_rationale: Optional[str] = None
        self._last_raw_response: Optional[str] = None
        self._last_parse_retries: int = 0
        self._grounding_overrides: int = 0

    @classmethod
    def _resolve_system_prompt(
        cls,
        mechanism: str,
        experiment_id: str,
        behaviour_cfg: Dict[str, Any],
    ) -> str:
        """Return the system prompt to use for this mechanism."""
        if mechanism == "prompt_optimized":
            prompt_path_str = behaviour_cfg.get(
                "trained_prompt_path",
                f"{cls._TRAINED_PROMPTS_DIR}/{experiment_id}/prompt.txt"
                if experiment_id
                else "",
            )
            if prompt_path_str:
                prompt_path = Path(prompt_path_str)
                if prompt_path.exists():
                    logger.info(
                        "Loading optimised system prompt from %s", prompt_path
                    )
                    return prompt_path.read_text(encoding="utf-8").strip()
                warnings.warn(
                    f"prompt_optimized mechanism: trained prompt not found at "
                    f"'{prompt_path_str}'. Falling back to default SYSTEM_PROMPT. "
                    f"Run `autops train <config>` to generate it.",
                    stacklevel=4,
                )
        return SYSTEM_PROMPT

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
                "storage_capacity_mb": meta.get("storage_capacity_mb", 4096.0),
                "uncompressed_observations": meta.get("uncompressed_observations", 0),
                "compression_progress": meta.get("compression_progress", 0),
                "total_observation_s": meta.get("total_observation_s", 0.0),
                "health_status": meta.get("health_status", "nominal"),
                "undetected_observations": meta.get("undetected_observations", 0),
                "daily_downlink_budget_mb": meta.get("daily_downlink_budget_mb", 27.0),
                "achievable_downlink_mb": meta.get("achievable_downlink_mb"),
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
                    system_prompt=self._system_prompt,
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

        # Substrate integrity: an LLM cell whose calls fail
        # must FAIL the episode — substituting a symbolic decision silently turns
        # the run into a mixed-substrate measurement (user decision 2026-06-11).
        if mode is None:
            raise RuntimeError(
                f"LLM cell integrity violation: LLM produced no valid mode after "
                f"{retries} retries — failing the episode instead of substituting "
                f"a symbolic decision. Check OLLAMA_HOST / model availability."
            )

        # Additional symbolic grounding checks
        mode = self._apply_grounding(mode, state)

        self._last_rationale = f"LLM: {rationale}" if rationale else f"LLM selected: {mode}"
        return {"eventsat_0": {"mode": mode}}

    def reason(self, state: Dict[str, Any], memory: Any) -> List[Dict[str, Any]]:
        """Produce structured reasoning steps for explanation/debug traces.

        Calls the LLM with a reasoning prompt and parses the structured output.
        Falls back to empty trace on failure (base class contract).
        """
        if not state:
            return [{"check": "state", "value": None, "implication": "no_state_default_charging"}]

        prompt = format_reasoning_prompt(state, memory)

        try:
            raw = self._client.generate(
                system_prompt=self._system_prompt,
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

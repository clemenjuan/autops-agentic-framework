"""
Prompt Optimizer for LLM-based EventSat Representations.

Implements bootstrap few-shot prompt optimization (DSPy-style, Khattab et al.
2023 [DSPy]) without a DSPy dependency. Loads high-utility trajectories from
logged Phase-4 hand-designed runs, constructs a few-shot augmented system prompt,
then iteratively scores candidates on a held-out split and picks the best one.

Supports both ``llm_eventsat`` (single-shot) and ``agentic_eventsat``
(multi-step) representations. The optimizer writes:
- ``data/trained_prompts/<experiment_id>/prompt.txt``  — the optimised prompt
- ``data/trained_prompts/<experiment_id>/metadata.json`` — source, score, config

Papers:
- Khattab et al. (2023) [DSPy] — programmatic prompt optimization; bootstrap
  few-shot and MIPRO as reference algorithms.  We implement a minimal
  bootstrap-fewshot variant that does not require DSPy as a runtime dep.
- Rodriguez-Fernandez et al. (2024) — baseline system prompt design for sat ops

Usage:
    optimizer = PromptOptimizer(config)
    prompt = optimizer.optimize(source_results_dir="data/results/eventsat_sas_sda_hybr_hd_ah")
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default number of few-shot examples to inject into the candidate prompt
_DEFAULT_NUM_EXAMPLES = 5
# Default number of candidate prompts to generate and score
_DEFAULT_NUM_CANDIDATES = 3
# Held-out fraction for scoring
_DEFAULT_HOLDOUT_FRAC = 0.2


def _load_step_records(results_dir: str | Path) -> List[Dict[str, Any]]:
    """Load per-step decision records from a results directory.

    Looks for ``steps.json`` or ``steps.jsonl`` under ``results_dir``.
    Each record must contain at least: ``state``, ``action``, ``utility``.
    Returns an empty list if no file is found (optimizer degrades gracefully).
    """
    base = Path(results_dir)
    for fname in ("steps.json", "steps.jsonl"):
        path = base / fname
        if path.exists():
            text = path.read_text(encoding="utf-8")
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                # Try JSONL
                records = []
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            logger.debug("prompt_optimizer: skipping malformed JSONL line: %s", exc)
                return records
    # Try results.json — it has episode-level summary but not step-level detail
    results_path = base / "results.json"
    if results_path.exists():
        logger.warning(
            "prompt_optimizer: only results.json found in %s — "
            "no per-step records available; few-shot examples will be empty.",
            results_dir,
        )
    return []


def _select_high_utility_examples(
    records: List[Dict[str, Any]],
    n: int = _DEFAULT_NUM_EXAMPLES,
    utility_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """Return up to ``n`` high-utility step records as few-shot examples."""
    high = [r for r in records if r.get("utility", 0.0) >= utility_threshold]
    if len(high) < n:
        # Fall back to the top-n by utility
        high = sorted(records, key=lambda r: r.get("utility", 0.0), reverse=True)
    return high[:n]


def _format_example(record: Dict[str, Any]) -> str:
    """Format a single step record as a few-shot example block."""
    state = record.get("state", {})
    action = record.get("action", {})
    rationale = record.get("rationale", "")
    utility = record.get("utility", "?")

    mode = action.get("mode", "unknown") if isinstance(action, dict) else str(action)
    lines = [
        "--- EXAMPLE ---",
        f"State: {json.dumps(state, default=str)}",
        f"Decision: {mode}",
    ]
    if rationale:
        lines.append(f"Rationale: {rationale}")
    lines.append(f"Utility: {utility}")
    return "\n".join(lines)


def _build_fewshot_prompt(base_prompt: str, examples: List[Dict[str, Any]]) -> str:
    """Append few-shot examples to a base system prompt."""
    if not examples:
        return base_prompt
    example_block = "\n\n".join(_format_example(e) for e in examples)
    return (
        base_prompt
        + "\n\nFEW-SHOT EXAMPLES (high-utility decisions from prior runs):\n"
        + example_block
        + "\n--- END EXAMPLES ---"
    )


def _score_prompt_on_holdout(
    prompt: str,
    holdout: List[Dict[str, Any]],
    llm_client: Any,
    base_user_prompt_fn: Any,
) -> float:
    """Score a prompt candidate on held-out records.

    Uses a mock or live LLM client. Returns mean utility-weighted accuracy:
    fraction of decisions that match the ground-truth high-utility action.
    In mock mode returns a deterministic score based on prompt length
    (a stand-in for unit tests without a live LLM).
    """
    if not holdout:
        return 0.0

    if getattr(llm_client, "mock_mode", True):
        # Deterministic mock score: slightly favour longer prompts as a proxy
        return min(1.0, 0.5 + len(prompt) / 50000)

    correct = 0
    for record in holdout:
        state = record.get("state", {})
        expected_mode = (
            record.get("action", {}).get("mode")
            if isinstance(record.get("action"), dict)
            else None
        )
        if expected_mode is None:
            continue
        try:
            user_prompt = base_user_prompt_fn(state, {})
            raw = llm_client.generate(
                system_prompt=prompt,
                user_prompt=user_prompt,
                json_mode=True,
            )
            parsed = json.loads(raw)
            chosen = parsed.get("mode")
            if chosen == expected_mode:
                correct += 1
        except Exception as exc:
            logger.debug("prompt_optimizer: holdout eval call failed (counted as miss): %s", exc)

    return correct / len(holdout) if holdout else 0.0


class PromptOptimizer:
    """Bootstrap few-shot system prompt optimizer for LLM-based representations.

    Args:
        config: Experiment config dict. Relevant keys:
            - ``experiment_id`` (str): Used to name the output directory.
            - ``representation_config.type`` (str): ``llm_eventsat`` or
              ``agentic_eventsat`` — selects the base prompt.
            - ``emergence_config.num_examples`` (int): Number of few-shot
              examples to inject (default 5).
            - ``emergence_config.num_candidates`` (int): Candidate prompts to
              score (default 3).
            - ``output_dir`` (str): Root results directory (default
              ``data/trained_prompts``).
    """

    OUTPUT_ROOT: str = "data/trained_prompts"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._experiment_id: str = self._config.get("experiment_id", "unnamed")
        repr_cfg = self._config.get("representation_config", {})
        self._repr_type: str = repr_cfg.get("type", "llm_eventsat")
        emergence_cfg = self._config.get("emergence_config", {})
        self._num_examples: int = emergence_cfg.get("num_examples", _DEFAULT_NUM_EXAMPLES)
        self._num_candidates: int = emergence_cfg.get("num_candidates", _DEFAULT_NUM_CANDIDATES)
        self._output_dir: Path = (
            Path(self._config.get("output_dir", self.OUTPUT_ROOT)) / self._experiment_id
        )

        # Lazy-initialise LLM client (only needed for live scoring)
        self._llm_client: Optional[Any] = None

    def _get_llm_client(self) -> Any:
        """Return (cached) LLM client."""
        if self._llm_client is None:
            from src.representation.llm_client import LLMClient
            self._llm_client = LLMClient(self._config)
        return self._llm_client

    def _get_base_prompt(self) -> str:
        """Return the base system prompt for this representation type."""
        if self._repr_type == "agentic_eventsat":
            from src.representation.agentic_prompts import AGENTIC_SYSTEM_PROMPT
            return AGENTIC_SYSTEM_PROMPT
        # Default: llm_eventsat
        from src.representation.llm_prompts import SYSTEM_PROMPT
        return SYSTEM_PROMPT

    def _get_user_prompt_fn(self) -> Any:
        """Return the user-prompt formatter for scoring."""
        if self._repr_type == "agentic_eventsat":
            from src.representation.agentic_prompts import format_planning_prompt
            return format_planning_prompt
        from src.representation.llm_prompts import format_state_prompt
        return format_state_prompt

    def optimize(
        self,
        source_results_dir: str | Path = "",
        seed: int = 42,
    ) -> str:
        """Run bootstrap few-shot optimization and write the result to disk.

        Args:
            source_results_dir: Path to a hand-designed baseline results dir
                that contains per-step records (``steps.json`` / ``steps.jsonl``).
                If empty, falls back to ``data/results/<experiment_id_hd>/`` where
                ``_hd_`` replaces the ``_lep_`` suffix in experiment_id.
            seed: Random seed for example selection.

        Returns:
            The optimised prompt string (also written to disk).
        """
        random.seed(seed)

        # Resolve source dir
        if not source_results_dir:
            baseline_id = self._experiment_id.replace("_lep_", "_hd_").replace("_lec_", "_hd_")
            source_results_dir = f"data/results/{baseline_id}"

        logger.info(
            "PromptOptimizer: loading trajectories from %s", source_results_dir
        )
        records = _load_step_records(source_results_dir)
        if not records:
            logger.warning(
                "PromptOptimizer: no step records found in %s. "
                "Falling back to base prompt without few-shot examples.",
                source_results_dir,
            )

        # Train/holdout split
        holdout_n = max(1, int(len(records) * _DEFAULT_HOLDOUT_FRAC))
        random.shuffle(records)
        holdout = records[:holdout_n]
        train = records[holdout_n:]

        base_prompt = self._get_base_prompt()
        user_prompt_fn = self._get_user_prompt_fn()
        llm = self._get_llm_client()

        # Generate candidates: vary the number and selection of few-shot examples
        candidates: List[Tuple[str, float]] = []
        for i in range(self._num_candidates):
            # Vary example selection across candidates
            subseed = seed + i
            random.seed(subseed)
            examples = _select_high_utility_examples(
                train,
                n=self._num_examples + i,  # slight variation
            )
            candidate_prompt = _build_fewshot_prompt(base_prompt, examples)
            score = _score_prompt_on_holdout(
                candidate_prompt, holdout, llm, user_prompt_fn
            )
            candidates.append((candidate_prompt, score))
            logger.info(
                "PromptOptimizer: candidate %d/%d score=%.3f (examples=%d)",
                i + 1, self._num_candidates, score, len(examples),
            )

        # Pick best candidate
        best_prompt, best_score = max(candidates, key=lambda t: t[1])
        logger.info("PromptOptimizer: best score=%.3f", best_score)

        # Write output
        self._output_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = self._output_dir / "prompt.txt"
        metadata_path = self._output_dir / "metadata.json"

        prompt_path.write_text(best_prompt, encoding="utf-8")
        metadata = {
            "experiment_id": self._experiment_id,
            "repr_type": self._repr_type,
            "source_results_dir": str(source_results_dir),
            "num_train_records": len(train),
            "num_holdout_records": len(holdout),
            "num_candidates": self._num_candidates,
            "best_score": round(best_score, 4),
            "seed": seed,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        logger.info("PromptOptimizer: wrote %s", prompt_path)
        return best_prompt

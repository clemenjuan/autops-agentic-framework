"""
Tests for PromptOptimizer — bootstrap few-shot prompt optimization.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.behaviour.prompt_optimizer import PromptOptimizer, _load_step_records


# ======================================================================
# Helper fixtures
# ======================================================================


def _make_step_records(n: int = 20) -> list:
    """Generate synthetic step records for testing."""
    records = []
    for i in range(n):
        records.append({
            "state": {"battery_soc": 0.5 + (i % 5) * 0.05, "ground_pass_active": i % 3 == 0},
            "action": {"mode": "charging" if i % 2 == 0 else "payload_observe"},
            "rationale": f"Step {i}: example rationale",
            "utility": 0.4 + (i % 6) * 0.1,
        })
    return records


# ======================================================================
# Unit tests: _load_step_records
# ======================================================================


class TestLoadStepRecords:
    def test_load_json_file(self, tmp_path: Path) -> None:
        records = _make_step_records(5)
        (tmp_path / "steps.json").write_text(json.dumps(records), encoding="utf-8")
        loaded = _load_step_records(tmp_path)
        assert len(loaded) == 5

    def test_load_jsonl_file(self, tmp_path: Path) -> None:
        records = _make_step_records(3)
        lines = "\n".join(json.dumps(r) for r in records)
        (tmp_path / "steps.jsonl").write_text(lines, encoding="utf-8")
        loaded = _load_step_records(tmp_path)
        assert len(loaded) == 3

    def test_no_file_returns_empty(self, tmp_path: Path) -> None:
        loaded = _load_step_records(tmp_path)
        assert loaded == []

    def test_results_json_returns_empty_with_warning(self, tmp_path: Path) -> None:
        (tmp_path / "results.json").write_text('{"experiment_id": "x"}', encoding="utf-8")
        loaded = _load_step_records(tmp_path)
        assert loaded == []  # results.json is episode-level only


# ======================================================================
# Unit tests: PromptOptimizer core
# ======================================================================


class TestPromptOptimizer:
    def test_optimize_returns_non_empty_prompt(self, tmp_path: Path) -> None:
        records = _make_step_records(20)
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "steps.json").write_text(json.dumps(records), encoding="utf-8")

        optimizer = PromptOptimizer(config={
            "experiment_id": "test_exp",
            "llm_mock": True,
            "output_dir": str(tmp_path / "trained_prompts"),
        })
        prompt = optimizer.optimize(source_results_dir=source_dir)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_optimize_writes_prompt_txt(self, tmp_path: Path) -> None:
        records = _make_step_records(10)
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "steps.json").write_text(json.dumps(records), encoding="utf-8")

        optimizer = PromptOptimizer(config={
            "experiment_id": "test_exp_write",
            "llm_mock": True,
            "output_dir": str(tmp_path / "trained_prompts"),
        })
        optimizer.optimize(source_results_dir=source_dir)

        prompt_path = tmp_path / "trained_prompts" / "test_exp_write" / "prompt.txt"
        assert prompt_path.exists()
        assert len(prompt_path.read_text()) > 0

    def test_optimize_writes_metadata_json(self, tmp_path: Path) -> None:
        records = _make_step_records(10)
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "steps.json").write_text(json.dumps(records), encoding="utf-8")

        optimizer = PromptOptimizer(config={
            "experiment_id": "meta_exp",
            "llm_mock": True,
            "output_dir": str(tmp_path / "trained_prompts"),
        })
        optimizer.optimize(source_results_dir=source_dir)

        meta_path = tmp_path / "trained_prompts" / "meta_exp" / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["experiment_id"] == "meta_exp"
        assert "best_score" in meta
        assert "num_train_records" in meta

    def test_optimize_empty_source_uses_base_prompt(self, tmp_path: Path) -> None:
        """When no step records exist, still produces a valid prompt (no crash)."""
        optimizer = PromptOptimizer(config={
            "experiment_id": "empty_exp",
            "llm_mock": True,
            "output_dir": str(tmp_path / "trained_prompts"),
        })
        prompt = optimizer.optimize(source_results_dir=tmp_path / "nonexistent")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_optimize_agentic_uses_agentic_system_prompt(self, tmp_path: Path) -> None:
        from src.representation.agentic_prompts import AGENTIC_SYSTEM_PROMPT

        optimizer = PromptOptimizer(config={
            "experiment_id": "agnt_exp",
            "llm_mock": True,
            "representation_config": {"type": "agentic_eventsat"},
            "output_dir": str(tmp_path / "trained_prompts"),
        })
        prompt = optimizer.optimize(source_results_dir=tmp_path / "nonexistent")
        # Base should be the agentic prompt
        assert AGENTIC_SYSTEM_PROMPT in prompt

    def test_optimize_llm_uses_llm_system_prompt(self, tmp_path: Path) -> None:
        from src.representation.llm_prompts import SYSTEM_PROMPT

        optimizer = PromptOptimizer(config={
            "experiment_id": "llm_exp",
            "llm_mock": True,
            "representation_config": {"type": "llm_eventsat"},
            "output_dir": str(tmp_path / "trained_prompts"),
        })
        prompt = optimizer.optimize(source_results_dir=tmp_path / "nonexistent")
        assert SYSTEM_PROMPT in prompt

    def test_mock_score_increases_with_prompt_length(self, tmp_path: Path) -> None:
        """In mock mode, the best candidate should have a score between 0 and 1."""
        records = _make_step_records(15)
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        (source_dir / "steps.json").write_text(json.dumps(records), encoding="utf-8")

        optimizer = PromptOptimizer(config={
            "experiment_id": "score_exp",
            "llm_mock": True,
            "behaviour_config": {"num_candidates": 2},
            "output_dir": str(tmp_path / "trained_prompts"),
        })
        optimizer.optimize(source_results_dir=source_dir)
        meta_path = tmp_path / "trained_prompts" / "score_exp" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert 0.0 <= meta["best_score"] <= 1.0


# ======================================================================
# Integration: LLMEventSat + prompt_optimized
# ======================================================================


class TestLLMEventSatPromptOptimized:
    def test_llm_prompt_optimized_loads_from_file(self, tmp_path: Path) -> None:
        from src.representation.llm_eventsat import LLMEventSat

        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("Custom LLM prompt for test", encoding="utf-8")

        rep = LLMEventSat(config={
            "llm_mock": True,
            "behaviour_config": {
                "mechanism": "prompt_optimized",
                "trained_prompt_path": str(prompt_path),
            },
        })
        assert rep._system_prompt == "Custom LLM prompt for test"

    def test_llm_hand_designed_uses_default_prompt(self) -> None:
        from src.representation.llm_eventsat import LLMEventSat
        from src.representation.llm_prompts import SYSTEM_PROMPT

        rep = LLMEventSat(config={"llm_mock": True})
        assert rep._system_prompt == SYSTEM_PROMPT

    def test_llm_prompt_optimized_fallback_on_missing(self) -> None:
        import warnings
        from src.representation.llm_eventsat import LLMEventSat
        from src.representation.llm_prompts import SYSTEM_PROMPT

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = LLMEventSat(config={
                "llm_mock": True,
                "experiment_id": "missing_exp",
                "behaviour_config": {"mechanism": "prompt_optimized"},
            })
        assert rep._system_prompt == SYSTEM_PROMPT
        assert any("trained prompt not found" in str(warning.message) for warning in w)

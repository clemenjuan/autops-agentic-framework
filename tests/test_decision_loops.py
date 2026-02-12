"""
Tests for Decision Loop base class.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest

from src.decision_loop.base import DecisionLoop


class DummyDecisionLoop(DecisionLoop):
    """Minimal concrete implementation for testing."""

    def process(self, observation: Any, memory: Any) -> Tuple[Any, Any]:
        return "noop", memory

    def get_metrics(self) -> Dict[str, float]:
        return {"latency_ms": 0.0}


class TestDecisionLoopABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            DecisionLoop(representation=None)  # type: ignore[abstract]

    def test_dummy_process(self) -> None:
        loop = DummyDecisionLoop(representation=None)
        action, memory = loop.process("obs", {"state": "init"})
        assert action == "noop"
        assert memory == {"state": "init"}

    def test_dummy_metrics(self) -> None:
        loop = DummyDecisionLoop(representation=None)
        metrics = loop.get_metrics()
        assert "latency_ms" in metrics

    def test_get_name(self) -> None:
        loop = DummyDecisionLoop(representation=None)
        assert loop.get_name() == "DummyDecisionLoop"

    def test_reset_does_not_raise(self) -> None:
        loop = DummyDecisionLoop(representation=None)
        loop.reset()  # Should be a no-op but must not raise

    def test_config_defaults_to_empty(self) -> None:
        loop = DummyDecisionLoop(representation=None)
        assert loop.config == {}

    def test_config_passed_through(self) -> None:
        loop = DummyDecisionLoop(representation=None, config={"max_iter": 5})
        assert loop.config["max_iter"] == 5

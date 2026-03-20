"""
Tests for Representation base class.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.decision_loop.context import DecisionContext
from src.representation.base import Representation


class DummyRepresentation(Representation):
    """Minimal concrete implementation for testing."""

    def encode_observation(self, observation: Any) -> Any:
        return {"encoded": observation}

    def select_action(self, context: DecisionContext) -> Any:
        return "default_action"


class TestRepresentationABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            Representation()  # type: ignore[abstract]

    def test_dummy_encode(self) -> None:
        rep = DummyRepresentation()
        encoded = rep.encode_observation({"raw": True})
        assert encoded == {"encoded": {"raw": True}}

    def test_dummy_select_action(self) -> None:
        rep = DummyRepresentation()
        ctx = DecisionContext(state={"state": 1}, memory={"memory": 2})
        action = rep.select_action(ctx)
        assert action == "default_action"

    def test_update_is_noop_by_default(self) -> None:
        rep = DummyRepresentation()
        rep.update(None)  # Should not raise

    def test_get_metrics_default_empty(self) -> None:
        rep = DummyRepresentation()
        assert rep.get_metrics() == {}

    def test_get_name(self) -> None:
        rep = DummyRepresentation()
        assert rep.get_name() == "DummyRepresentation"

    def test_config_default_empty(self) -> None:
        rep = DummyRepresentation()
        assert rep.config == {}

    def test_config_passed_through(self) -> None:
        rep = DummyRepresentation(config={"lr": 0.01})
        assert rep.config["lr"] == 0.01

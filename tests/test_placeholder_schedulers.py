"""Tests for placeholder schedule-producer representations and the ground guard."""

from __future__ import annotations

import warnings

import pytest

from src.decision_procedure.context import DecisionContext
from src.behaviour.controller import BehaviourController
from src.orchestration.config_loader import ExperimentConfig
from src.representation.placeholder_schedulers import (
    SubsymbolicSchedulerEventSat,
)


def _fresh_pass_state():
    """State during a pass with fresh telemetry (triggers schedule generation)."""
    return {
        "battery_soc": 0.7,
        "current_mode": "communication",
        "in_sunlight": True,
        "ground_pass_active": True,
        "obc_data_mb": 5.0,
        "jetson_raw_mb": 9.41,
        "jetson_compressed_mb": 0.0,
        "uncompressed_observations": 1,
        "undetected_observations": 0,
        "staleness_steps": 1,  # fresh (<= staleness_threshold)
        "estimated_gap_steps": 93,
        "daily_downlink_budget_mb": 27.0,
    }


# agentic_scheduler_eventsat is now a REAL core (agentic_scheduler_eventsat.py) — no
# longer a placeholder; see tests/test_agentic_scheduler.py. subsymbolic (RL schedule
# producer) remains the only placeholder scheduler.
PLACEHOLDERS = [
    (SubsymbolicSchedulerEventSat, "subsymbolic_scheduler_eventsat", "subsymbolic"),
]


class TestPlaceholderSchedulers:
    @pytest.mark.parametrize("cls,reg_name,family", PLACEHOLDERS)
    def test_emits_schedule_on_fresh_pass(self, cls, reg_name, family):
        rep = cls()
        ctx = DecisionContext(state=_fresh_pass_state(), loop_type="sda")
        action = rep.select_action(ctx)
        sat = action["eventsat_0"]
        assert "schedule" in sat
        assert sat["schedule"]  # non-empty
        assert sat["mode"] == "communication"

    @pytest.mark.parametrize("cls,reg_name,family", PLACEHOLDERS)
    def test_is_placeholder_flag(self, cls, reg_name, family):
        assert cls.is_placeholder is True

    @pytest.mark.parametrize("cls,reg_name,family", PLACEHOLDERS)
    def test_rationale_marked_placeholder(self, cls, reg_name, family):
        rep = cls()
        rep.select_action(DecisionContext(state=_fresh_pass_state(), loop_type="sda"))
        assert "PLACEHOLDER" in (rep.get_rationale() or "")
        assert family in (rep.get_rationale() or "")

    @pytest.mark.parametrize("cls,reg_name,family", PLACEHOLDERS)
    def test_registered_in_controller(self, cls, reg_name, family):
        import src.representation.placeholder_schedulers  # noqa: F401  (trigger @register)

        controller = BehaviourController(config={})
        rep = controller.get_representation(repr_type=reg_name, repr_config={})
        assert isinstance(rep, cls)


class TestGroundScheduleGuard:
    """The validator rejects non-schedule representations under ground paradigms."""

    def _cfg(self, rep, rep_type, ops, **extra):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ExperimentConfig(
                representation=rep,
                representation_config={"type": rep_type},
                operations_paradigm=ops,
                environment={"scenario": "eventsat"},
                **extra,
            )

    @pytest.mark.parametrize("ops", ["autonomous_ground", "conventional_ground"])
    def test_non_schedule_rep_under_ground_raises(self, ops):
        with pytest.raises(ValueError, match="schedule-producing"):
            self._cfg("subsymbolic", "subsymbolic_eventsat", ops)

    @pytest.mark.parametrize("ops", ["autonomous_ground", "conventional_ground"])
    def test_scheduler_placeholder_under_ground_ok(self, ops):
        cfg = self._cfg("subsymbolic", "subsymbolic_scheduler_eventsat", ops)
        assert cfg.operations_paradigm == ops

    def test_symbolic_schedule_under_ground_ok(self):
        cfg = self._cfg("symbolic", "schedule_based_eventsat", "autonomous_ground")
        assert cfg.representation_config["type"] == "schedule_based_eventsat"

    def test_non_schedule_rep_under_ah_ok(self):
        # The guard only applies to ground paradigms.
        cfg = self._cfg("subsymbolic", "subsymbolic_eventsat", "autonomous_hybrid")
        assert cfg.operations_paradigm == "autonomous_hybrid"

    def test_writable_coala_accepts_agentic_scheduler(self):
        cfg = self._cfg(
            "hybrid",
            "agentic_scheduler_eventsat",
            "autonomous_ground",
            behaviour="emergent",
            behaviour_config={"mode": "emergent", "mechanism": "writable_coala"},
        )
        assert cfg.behaviour_config["mechanism"] == "writable_coala"

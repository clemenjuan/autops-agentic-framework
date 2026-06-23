"""Tests for ScheduleBasedEventSat representation and schedule generation."""

import pytest

from src.core.decision_procedure.context import DecisionContext
from src.core.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteState,
)
from src.eventsat.schedule_symbolic import ScheduleBasedEventSat, _merge_schedule


def _ctx(state, **kwargs):
    """Helper to wrap a state dict in a DecisionContext for tests."""
    return DecisionContext(state=state, **kwargs)


# -----------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------


def _make_observation(
    step=0,
    battery_soc=0.8,
    data_stored_mb=10.0,
    obc_data_mb=5.0,
    jetson_raw_mb=0.0,
    jetson_compressed_mb=0.0,
    uncompressed_observations=0,
    undetected_observations=0,
    mode="charging",
    in_sunlight=True,
    ground_pass_active=False,
    staleness_steps=0,
    estimated_gap_steps=93,
    health_status="nominal",
):
    metadata = {
        "in_sunlight": in_sunlight,
        "ground_pass_active": ground_pass_active,
        "uncompressed_observations": uncompressed_observations,
        "undetected_observations": undetected_observations,
        "total_observation_s": 0.0,
        "storage_capacity_mb": 512.0,
        "health_status": health_status,
        "staleness_steps": staleness_steps,
        "last_update_step": step - staleness_steps,
        "jetson_raw_mb": jetson_raw_mb,
        "jetson_compressed_mb": jetson_compressed_mb,
        "obc_data_mb": obc_data_mb,
        "daily_downlink_budget_mb": 27.0,
    }
    if ground_pass_active:
        metadata["estimated_gap_steps"] = estimated_gap_steps

    sat = SatelliteState(
        satellite_id="eventsat_0",
        position=[0.0, 0.0, 500.0],
        velocity=[0.0, 0.0, 0.0],
        resources={
            "battery_soc": battery_soc,
            "data_stored_mb": data_stored_mb,
            "obc_data_mb": obc_data_mb,
            "data_downlinked_mb": 0.0,
        },
        status=mode,
        metadata=metadata,
    )
    constellation = ConstellationState(
        timestep=step,
        epoch_seconds=step * 60.0,
        satellites={"eventsat_0": sat},
        global_info={"max_steps": 10080},
    )
    return EnvironmentObservation(
        constellation_state=constellation, tasks=[], events=[]
    )


# -----------------------------------------------------------------
# _merge_schedule helper
# -----------------------------------------------------------------


class TestMergeSchedule:
    def test_empty(self):
        assert _merge_schedule([]) == []

    def test_single_entry(self):
        assert _merge_schedule([("charging", 5)]) == [("charging", 5)]

    def test_merges_consecutive_same_mode(self):
        result = _merge_schedule([("charging", 3), ("charging", 4)])
        assert result == [("charging", 7)]

    def test_preserves_different_modes(self):
        sched = [("charging", 2), ("payload_observe", 1), ("charging", 3)]
        result = _merge_schedule(sched)
        assert result == [("charging", 2), ("payload_observe", 1), ("charging", 3)]

    def test_merges_only_adjacent(self):
        sched = [("charging", 2), ("payload_observe", 1), ("charging", 3), ("payload_observe", 2)]
        result = _merge_schedule(sched)
        assert result == [
            ("charging", 2), ("payload_observe", 1),
            ("charging", 3), ("payload_observe", 2),
        ]


# -----------------------------------------------------------------
# ScheduleBasedEventSat.encode_observation
# -----------------------------------------------------------------


class TestEncodeObservation:
    def test_extracts_all_fields(self):
        rep = ScheduleBasedEventSat()
        obs = _make_observation(
            battery_soc=0.75,
            obc_data_mb=10.0,
            jetson_raw_mb=5.0,
            uncompressed_observations=2,
            ground_pass_active=True,
            staleness_steps=3,
            estimated_gap_steps=90,
        )
        state = rep.encode_observation(obs)
        assert state["battery_soc"] == pytest.approx(0.75)
        assert state["obc_data_mb"] == pytest.approx(10.0)
        assert state["jetson_raw_mb"] == pytest.approx(5.0)
        assert state["uncompressed_observations"] == 2
        assert state["ground_pass_active"] is True
        assert state["staleness_steps"] == 3
        assert state["estimated_gap_steps"] == 90

    def test_returns_empty_for_none(self):
        rep = ScheduleBasedEventSat()
        assert rep.encode_observation(None) == {}

    def test_returns_empty_for_missing_satellite(self):
        rep = ScheduleBasedEventSat()
        empty_constellation = ConstellationState(
            timestep=0, epoch_seconds=0.0,
            satellites={},
            global_info={},
        )
        obs = EnvironmentObservation(
            constellation_state=empty_constellation, tasks=[], events=[]
        )
        assert rep.encode_observation(obs) == {}


# -----------------------------------------------------------------
# ScheduleBasedEventSat.select_action
# -----------------------------------------------------------------


class TestSelectAction:
    def test_between_passes_returns_charging(self):
        rep = ScheduleBasedEventSat()
        obs = _make_observation(ground_pass_active=False, staleness_steps=50)
        state = rep.encode_observation(obs)
        action = rep.select_action(_ctx(state))
        assert action == {"eventsat_0": {"mode": "charging"}}

    def test_pass_with_stale_telemetry_communicates_only(self):
        """When pass starts but telemetry is stale, communicate first (no schedule)."""
        rep = ScheduleBasedEventSat(config={"staleness_threshold": 5})
        obs = _make_observation(ground_pass_active=True, staleness_steps=50)
        state = rep.encode_observation(obs)
        action = rep.select_action(_ctx(state))
        assert action["eventsat_0"]["mode"] == "communication"
        assert "schedule" not in action["eventsat_0"]

    def test_pass_with_fresh_telemetry_generates_schedule(self):
        """After telemetry received (low staleness), schedule is generated."""
        rep = ScheduleBasedEventSat(config={"staleness_threshold": 5})
        obs = _make_observation(
            battery_soc=0.8,
            ground_pass_active=True,
            staleness_steps=1,
            estimated_gap_steps=93,
        )
        state = rep.encode_observation(obs)
        action = rep.select_action(_ctx(state))
        assert action["eventsat_0"]["mode"] == "communication"
        assert "schedule" in action["eventsat_0"]
        assert len(action["eventsat_0"]["schedule"]) > 0

    def test_schedule_not_regenerated_within_same_pass(self):
        """Once schedule is generated, subsequent pass steps don't re-generate."""
        rep = ScheduleBasedEventSat(config={"staleness_threshold": 5})
        # Step 1: fresh telemetry → generates schedule
        obs1 = _make_observation(ground_pass_active=True, staleness_steps=1, estimated_gap_steps=50)
        state1 = rep.encode_observation(obs1)
        action1 = rep.select_action(_ctx(state1))
        assert "schedule" in action1["eventsat_0"]

        # Step 2: still in pass, staleness now 2 (still fresh) → no schedule
        obs2 = _make_observation(ground_pass_active=True, staleness_steps=2, estimated_gap_steps=50)
        state2 = rep.encode_observation(obs2)
        action2 = rep.select_action(_ctx(state2))
        assert action2["eventsat_0"]["mode"] == "communication"
        assert "schedule" not in action2["eventsat_0"]

    def test_schedule_resets_between_passes(self):
        """After pass ends and new pass starts, schedule is generated fresh."""
        rep = ScheduleBasedEventSat(config={"staleness_threshold": 5})
        # First pass
        obs_pass = _make_observation(ground_pass_active=True, staleness_steps=1, estimated_gap_steps=50)
        rep.select_action(_ctx(rep.encode_observation(obs_pass)))

        # Between passes
        obs_between = _make_observation(ground_pass_active=False, staleness_steps=40)
        rep.select_action(_ctx(rep.encode_observation(obs_between)))

        # Second pass
        obs_pass2 = _make_observation(ground_pass_active=True, staleness_steps=1, estimated_gap_steps=50)
        action2 = rep.select_action(_ctx(rep.encode_observation(obs_pass2)))
        assert "schedule" in action2["eventsat_0"]

    def test_empty_state_defaults_to_charging(self):
        rep = ScheduleBasedEventSat()
        action = rep.select_action(_ctx({}))
        assert action == {"eventsat_0": {"mode": "charging"}}

    def test_get_rationale_is_set_after_action(self):
        rep = ScheduleBasedEventSat()
        obs = _make_observation(ground_pass_active=False)
        state = rep.encode_observation(obs)
        rep.select_action(_ctx(state))
        assert rep.get_rationale() is not None


# -----------------------------------------------------------------
# Schedule generation correctness
# -----------------------------------------------------------------


class TestScheduleGeneration:
    def _make_rep(self, **kwargs):
        defaults = {
            "solar_generation_w": 24.0,
            "battery_capacity_wh": 84.0,
            "eclipse_fraction": 0.36,
            "step_duration_s": 60.0,
            "compression_time_factor": 2.0,
            "detection_steps": 5,
            "observation_size_mb": 9.41,
            "compression_ratio": 5.11,
            "jetson_to_obc_rate_kbps": 50.0,
            "daily_downlink_budget_mb": 27.0,
            "charge_reserve_fraction": 0.12,
            "min_soc_for_operations": 0.40,
        }
        defaults.update(kwargs)
        return ScheduleBasedEventSat(config=defaults)

    def test_schedule_total_steps_equals_gap(self):
        rep = self._make_rep()
        gap = 80
        state = {
            "battery_soc": 0.8,
            "uncompressed_observations": 0,
            "undetected_observations": 0,
            "jetson_compressed_mb": 0.0,
            "jetson_raw_mb": 0.0,
            "obc_data_mb": 0.0,
            "daily_downlink_budget_mb": 27.0,
        }
        schedule = rep._generate_schedule(state, gap)
        total = sum(steps for _, steps in schedule)
        assert total == gap

    def test_schedule_ends_with_charging_reserve(self):
        rep = self._make_rep(charge_reserve_fraction=0.10)
        gap = 100
        state = {
            "battery_soc": 0.9,
            "uncompressed_observations": 0,
            "undetected_observations": 0,
            "jetson_compressed_mb": 0.0,
            "jetson_raw_mb": 0.0,
            "obc_data_mb": 0.0,
            "daily_downlink_budget_mb": 27.0,
        }
        schedule = rep._generate_schedule(state, gap)
        # Last entry should be charging (reserve block)
        assert schedule[-1][0] == "charging"
        # Reserve is at least the configured fraction
        reserve = max(5, int(gap * 0.10))
        assert schedule[-1][1] >= reserve

    def test_low_battery_charges_first(self):
        rep = self._make_rep(min_soc_for_operations=0.40)
        state = {
            "battery_soc": 0.2,  # critically low
            "uncompressed_observations": 0,
            "undetected_observations": 0,
            "jetson_compressed_mb": 0.0,
            "jetson_raw_mb": 0.0,
            "obc_data_mb": 0.0,
            "daily_downlink_budget_mb": 27.0,
        }
        schedule = rep._generate_schedule(state, 50)
        # First activity must be charging
        assert schedule[0][0] == "charging"

    def test_uncompressed_data_triggers_compression(self):
        rep = self._make_rep()
        state = {
            "battery_soc": 0.8,
            "uncompressed_observations": 2,
            "undetected_observations": 0,
            "jetson_compressed_mb": 0.0,
            "jetson_raw_mb": 18.82,  # 2 observations
            "obc_data_mb": 0.0,
            "daily_downlink_budget_mb": 27.0,
        }
        schedule = rep._generate_schedule(state, 50)
        modes = [m for m, _ in schedule]
        assert "payload_compress" in modes

    def test_observe_when_pipeline_empty_and_battery_good(self):
        rep = self._make_rep()
        state = {
            "battery_soc": 0.9,
            "uncompressed_observations": 0,
            "undetected_observations": 0,
            "jetson_compressed_mb": 0.0,
            "jetson_raw_mb": 0.0,
            "obc_data_mb": 0.0,
            "daily_downlink_budget_mb": 27.0,
        }
        schedule = rep._generate_schedule(state, 80)
        modes = [m for m, _ in schedule]
        assert "payload_observe" in modes

    def test_no_observe_when_pipeline_saturated(self):
        rep = self._make_rep()
        state = {
            "battery_soc": 0.9,
            "uncompressed_observations": 0,
            "undetected_observations": 0,
            "jetson_compressed_mb": 20.0,  # saturated pipeline
            "jetson_raw_mb": 0.0,
            "obc_data_mb": 20.0,           # > daily_downlink_budget_mb (27)
            "daily_downlink_budget_mb": 27.0,
        }
        schedule = rep._generate_schedule(state, 50)
        modes = [m for m, _ in schedule]
        assert "payload_observe" not in modes

    def test_all_modes_are_valid(self):
        valid_modes = {
            "charging", "communication", "payload_observe", "payload_compress",
            "payload_detect", "payload_send", "safe",
        }
        rep = self._make_rep()
        state = {
            "battery_soc": 0.8,
            "uncompressed_observations": 1,
            "undetected_observations": 1,
            "jetson_compressed_mb": 2.0,
            "jetson_raw_mb": 9.41,
            "obc_data_mb": 5.0,
            "daily_downlink_budget_mb": 27.0,
        }
        schedule = rep._generate_schedule(state, 60)
        for mode, steps in schedule:
            assert mode in valid_modes, f"Invalid mode in schedule: {mode}"
            assert steps > 0, f"Schedule entry has non-positive steps: {steps}"

    def test_schedule_steps_always_positive(self):
        rep = self._make_rep()
        for battery_soc in [0.1, 0.5, 0.9]:
            state = {
                "battery_soc": battery_soc,
                "uncompressed_observations": 0,
                "undetected_observations": 0,
                "jetson_compressed_mb": 0.0,
                "jetson_raw_mb": 0.0,
                "obc_data_mb": 0.0,
                "daily_downlink_budget_mb": 27.0,
            }
            schedule = rep._generate_schedule(state, 93)
            for mode, steps in schedule:
                assert steps > 0

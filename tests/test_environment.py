"""
Tests for the Satellite Environment base class and data structures.
"""

from __future__ import annotations

import pytest

from src.core.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteEnvironment,
    SatelliteState,
    StepResult,
)


# ======================================================================
# Data structure tests
# ======================================================================


class TestSatelliteState:
    """Tests for the SatelliteState dataclass."""

    def test_default_construction(self) -> None:
        state = SatelliteState(satellite_id="sat_0")
        assert state.satellite_id == "sat_0"
        assert state.position == [0.0, 0.0, 0.0]
        assert state.velocity == [0.0, 0.0, 0.0]
        assert state.status == "nominal"
        assert state.resources == {}
        assert state.metadata == {}

    def test_custom_construction(self) -> None:
        state = SatelliteState(
            satellite_id="sat_1",
            position=[7000.0, 0.0, 0.0],
            velocity=[0.0, 7.5, 0.0],
            resources={"power_w": 100.0},
            status="safe_mode",
            metadata={"subsystem": "payload"},
        )
        assert state.position[0] == 7000.0
        assert state.resources["power_w"] == 100.0
        assert state.status == "safe_mode"


class TestConstellationState:
    """Tests for the ConstellationState dataclass."""

    def test_default_construction(self) -> None:
        cs = ConstellationState(timestep=0, epoch_seconds=0.0)
        assert cs.timestep == 0
        assert cs.satellites == {}

    def test_with_satellites(self) -> None:
        sat = SatelliteState(satellite_id="sat_0")
        cs = ConstellationState(
            timestep=10,
            epoch_seconds=600.0,
            satellites={"sat_0": sat},
        )
        assert len(cs.satellites) == 1
        assert cs.satellites["sat_0"].satellite_id == "sat_0"


class TestStepResult:
    """Tests for the StepResult dataclass."""

    def test_defaults(self) -> None:
        obs = EnvironmentObservation(
            constellation_state=ConstellationState(timestep=0, epoch_seconds=0.0)
        )
        sr = StepResult(observation=obs)
        assert sr.done is False
        assert sr.truncated is False
        assert sr.rewards == {}


# ======================================================================
# Abstract class contract tests
# ======================================================================


class DummyEnvironment(SatelliteEnvironment):
    """Minimal concrete implementation for testing the ABC."""

    def reset(self, seed=None):
        self.current_step = 0
        return EnvironmentObservation(
            constellation_state=ConstellationState(timestep=0, epoch_seconds=0.0)
        )

    def step(self, actions):
        self.current_step += 1
        obs = EnvironmentObservation(
            constellation_state=ConstellationState(
                timestep=self.current_step,
                epoch_seconds=self.current_step * self.timestep_seconds,
            )
        )
        return StepResult(observation=obs)

    def get_observation(self):
        return EnvironmentObservation(
            constellation_state=ConstellationState(
                timestep=self.current_step,
                epoch_seconds=self.current_step * self.timestep_seconds,
            )
        )

    def get_metrics(self):
        return {"steps_taken": float(self.current_step)}


class TestSatelliteEnvironmentABC:
    """Tests to verify the abstract base class contract."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            SatelliteEnvironment(config={})  # type: ignore[abstract]

    def test_dummy_reset(self) -> None:
        env = DummyEnvironment(config={"constellation_size": 3, "timestep_seconds": 30})
        obs = env.reset()
        assert obs.constellation_state.timestep == 0
        assert env.constellation_size == 3

    def test_dummy_step(self) -> None:
        env = DummyEnvironment(config={"max_steps": 5})
        env.reset()
        result = env.step({})
        assert result.observation.constellation_state.timestep == 1

    def test_is_done(self) -> None:
        env = DummyEnvironment(config={"max_steps": 2})
        env.reset()
        assert not env.is_done()
        env.step({})
        env.step({})
        assert env.is_done()

    def test_get_config(self) -> None:
        cfg = {"constellation_size": 5, "timestep_seconds": 60}
        env = DummyEnvironment(config=cfg)
        assert env.get_config()["constellation_size"] == 5

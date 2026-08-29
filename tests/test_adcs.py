"""Tests for the ADCS simulation.

Structural tests confirm the closed loop runs through all modules and produces a
state history of the expected shape — their job is to catch wiring and interface
breakage as real implementations replace the stubs.

Physics tests check invariants of the real implementations: the disturbance
torques (always run, built on hand-made environment samples so they need no
Orekit) and the Orekit-backed propagator (run only when Orekit is available, via
the requires_orekit guard).
"""
from __future__ import annotations

from dataclasses import replace
from typing import List

import numpy as np
import pytest

from src.environment.orbital import propagator as P
from src.environment.orbital.adcs.control import initial_setpoint
from src.environment.orbital.adcs.dynamics import (
    _aerodynamic_torque,
    _gravity_gradient_torque,
    _residual_dipole_torque,
    _srp_torque,
    disturbance_torque,
    dcm_eci_to_body,
)
from src.environment.orbital.adcs.estimator import initial_estimator_state
from src.environment.orbital.adcs.eventsat import actuators, sim, orbit, satellite, sensors
from src.environment.orbital.adcs.simulation import initial_state, run, step
from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.adcs.actuators import apply_magnetorquer, apply_reaction_wheel
from src.environment.orbital.adcs.configs import SimulationConfig, MagnetometerConfig
from src.environment.orbital.adcs.sensors import initial_sensor_state, read_magnetometer

requires_orekit = pytest.mark.skipif(
    not P.is_available(), reason="Orekit unavailable; skipping physics checks."
)

START_STEP = 0
END_STEP = 10


@pytest.fixture
def history() -> List[SatState]:
    """Run the skeleton once and share the resulting state history."""
    return run(
        sensors, actuators, satellite, sim, start_step=START_STEP, end_step=END_STEP, orbit=orbit
    )


def test_eventsat_config_counts() -> None:
    """The EventSat suite has the expected instrument counts."""
    assert len(sensors.magnetometers) == 3 
    assert len(sensors.fine_sun_sensors) == 2
    assert len(sensors.star_trackers) == 0
    assert len(actuators.reaction_wheels) == 4
    assert len(actuators.magnetorquers) == 3
    assert len(sensors.rate_gyros) == 1


def test_run_executes_end_to_end(history: List[SatState]) -> None:
    """run() completes and returns one state per step boundary."""
    assert len(history) == END_STEP - START_STEP + 1
    assert all(isinstance(s, SatState) for s in history)


def test_run_advances_time(history: List[SatState]) -> None:
    """Time runs from start_step * step_s to end_step * step_s."""
    assert history[0].t == START_STEP * sim.step_s
    assert history[-1].t == END_STEP * sim.step_s


def test_final_state_shapes(history: List[SatState]) -> None:
    """The final state has the expected vector shapes."""
    final = history[-1]
    assert final.q_eci_body.shape == (4,)
    assert final.omega_body.shape == (3,)
    assert final.wheel_speeds.shape == (len(actuators.reaction_wheels),)
    assert final.r_eci.shape == (3,)
    assert final.v_eci.shape == (3,)


def test_single_step_returns_state_and_estimator() -> None:
    """One step returns an advanced state and an estimator with a 6x6 covariance."""
    P.configure(orbit)
    rng = np.random.default_rng(0)
    state = initial_state(0.0, len(actuators.reaction_wheels))
    sensor_state = initial_sensor_state(sensors, rng)
    estimator = initial_estimator_state()
    new_state, new_sensor_state, new_estimator = step(
        state, sensor_state, estimator, sensors, actuators,
        satellite, initial_setpoint(), sim, rng
    )
    assert isinstance(new_state, SatState)
    assert new_state.t == sim.step_s
    assert new_estimator.covariance.shape == (6, 6)
    assert new_sensor_state.gyro_bias.shape == (len(sensors.rate_gyros), 3)


ALT_RADIUS = 6.828e6  # m, ~450 km altitude


def _sample_state(q: np.ndarray, r: np.ndarray, v: np.ndarray) -> SatState:
    return SatState(
        t=0.0,
        q_eci_body=q,
        omega_body=np.zeros(3),
        wheel_speeds=np.zeros(len(actuators.reaction_wheels)),
        r_eci=r,
        v_eci=v,
    )


def _sample_env(
    r: np.ndarray,
    v: np.ndarray,
    b: np.ndarray,
    sun: np.ndarray,
    eclipse: bool = False,
    rho: float = 1e-12,
) -> P.EnvironmentData:
    return P.EnvironmentData(
        r_eci=r,
        v_eci=v,
        b_field_eci=b,
        sun_vector_eci=sun,
        eclipse=eclipse,
        atmospheric_density=rho,
    )


class TestDisturbances:
    """Physics invariants for the four environmental disturbance torques."""

    def test_gravity_gradient_magnitude(self) -> None:
        """Gravity gradient sits in the ~1e-7 N·m band for a 6U at 450 km."""
        r = np.array([ALT_RADIUS, 0.0, 0.0])
        state = _sample_state(np.array([1.0, 0.0, 0.0, 0.0]), r, np.zeros(3))
        env = _sample_env(r, np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]))
        tau = _gravity_gradient_torque(state, env, satellite)
        assert 1e-8 < np.linalg.norm(tau) < 1e-6

    def test_gravity_gradient_principal_axis_null(self) -> None:
        """Torque vanishes when a principal axis of the inertia points at nadir."""
        _, evecs = np.linalg.eigh(satellite.inertia_full)
        r = ALT_RADIUS * evecs[:, 1]  # nadir along a principal axis (identity attitude)
        state = _sample_state(np.array([1.0, 0.0, 0.0, 0.0]), r, np.zeros(3))
        env = _sample_env(r, np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]))
        assert np.linalg.norm(_gravity_gradient_torque(state, env, satellite)) < 1e-15

    def test_residual_dipole_perpendicular_and_band(self) -> None:
        """τ = m × B is perpendicular to both, in the ~1e-6 band for a real dipole."""
        sat = replace(satellite, residual_dipole=np.array([0.05, -0.02, 0.03]))
        b = np.array([2.0e-5, 1.0e-5, -3.0e-5])
        r = np.array([ALT_RADIUS, 0.0, 0.0])
        state = _sample_state(np.array([1.0, 0.0, 0.0, 0.0]), r, np.zeros(3))
        env = _sample_env(r, np.zeros(3), b, np.array([1.0, 0.0, 0.0]))
        tau = _residual_dipole_torque(state, env, sat)
        assert abs(tau @ sat.residual_dipole) < 1e-18
        assert abs(tau @ b) < 1e-18
        assert 1e-7 < np.linalg.norm(tau) < 1e-5

    def test_aerodynamic_drag_magnitude(self) -> None:
        """Drag torque sits in the ~1e-8 N·m band for a 6U at 450 km."""
        r = np.array([ALT_RADIUS, 0.0, 0.0])
        v = np.array([0.0, 1500.0, 7400.0])
        state = _sample_state(np.array([1.0, 0.0, 0.0, 0.0]), r, v)
        env = _sample_env(r, v, np.zeros(3), np.array([1.0, 0.0, 0.0]))
        tau = _aerodynamic_torque(state, env, satellite)
        assert 1e-9 < np.linalg.norm(tau) < 1e-7

    def test_srp_eclipse_gate(self) -> None:
        """SRP is nonzero in sunlight and exactly zero in eclipse."""
        r = np.array([ALT_RADIUS, 0.0, 0.0])
        sun = np.array([0.6, 0.0, 0.8])
        state = _sample_state(np.array([1.0, 0.0, 0.0, 0.0]), r, np.zeros(3))
        lit = _sample_env(r, np.zeros(3), np.zeros(3), sun, eclipse=False)
        dark = _sample_env(r, np.zeros(3), np.zeros(3), sun, eclipse=True)
        assert np.linalg.norm(_srp_torque(state, lit, satellite)) > 0.0
        assert np.linalg.norm(_srp_torque(state, dark, satellite)) == 0.0

    def test_total_is_sum_of_terms(self) -> None:
        """disturbance_torque equals the sum of the four individual terms."""
        sat = replace(satellite, residual_dipole=np.array([0.02, -0.01, 0.015]))
        r = np.array([6.6e6, 1.5e6, 0.8e6])
        v = np.array([-200.0, 1000.0, 7570.0])
        sun = np.array([0.3, -0.5, 0.81])
        sun = sun / np.linalg.norm(sun)
        q = np.array([0.98, 0.10, -0.05, 0.15])
        q = q / np.linalg.norm(q)
        state = _sample_state(q, r, v)
        env = _sample_env(r, v, np.array([1.5e-5, -1.0e-5, 2.8e-5]), sun)
        total = disturbance_torque(state, env, sat)
        parts = (
            _gravity_gradient_torque(state, env, sat)
            + _residual_dipole_torque(state, env, sat)
            + _aerodynamic_torque(state, env, sat)
            + _srp_torque(state, env, sat)
        )
        assert total.shape == (3,)
        assert np.allclose(total, parts)


def test_actuator_return_shapes() -> None:
    """Wheels return a scalar motor torque; magnetorquers a (3,) body torque."""
    state = initial_state(0.0, len(actuators.reaction_wheels))
    env = _sample_env(np.zeros(3), np.zeros(3), np.array([2e-5, 1e-5, -3e-5]),
                np.array([1.0, 0.0, 0.0]))
    u = apply_reaction_wheel(state, actuators.reaction_wheels[0], 1e-3, 0, sim.step_s)
    tau = apply_magnetorquer(state, env, actuators.magnetorquers[0], 0.6)
    assert isinstance(u, float)
    assert tau.shape == (3,)


@requires_orekit
def test_orbit_propagation_physics() -> None:
    """With an orbit configured, the propagator yields a physically correct SSO.

    Checks (tolerances loose, only catching gross errors):
      - |r|, |v| match a circular orbit at the configured altitude
      - r . v ~ 0 (circular: position perpendicular to velocity)
      - position roughly reverses after half a period
      - RAAN drifts at the sun-synchronous rate (~+0.986 deg/day) -> J2 is active
    """
    P.configure(orbit)

    # Expectations derived from the same config, not hard-coded.
    mu = 3.986004418e14  # WGS84 Earth mu [m^3/s^2]
    r_e = 6378137.0      # WGS84 equatorial radius [m]
    a = r_e + orbit.altitude_km * 1000.0
    v_circular = np.sqrt(mu / a)
    period = 2.0 * np.pi * np.sqrt(a**3 / mu)

    def raan_deg(r: np.ndarray, v: np.ndarray) -> float:
        h = np.cross(r, v)                    # orbit normal
        node = np.array([-h[1], h[0], 0.0])   # z x h -> ascending-node direction
        return float(np.degrees(np.arctan2(node[1], node[0])))

    e0 = P.get_environment(0.0)
    r0 = np.linalg.norm(e0.r_eci)
    v0 = np.linalg.norm(e0.v_eci)

    # Magnitudes within ~10 km / ~10 m/s of the circular ideal.
    assert abs(r0 - a) < 1.0e4
    assert abs(v0 - v_circular) < 1.0e1
    # Circular: r perpendicular to v (dot product small relative to r*v).
    assert abs(np.dot(e0.r_eci, e0.v_eci)) < 1.0e-3 * r0 * v0

    # Half a period later, position points roughly the opposite way.
    eh = P.get_environment(period / 2.0)
    cos_ang = np.dot(e0.r_eci, eh.r_eci) / (r0 * np.linalg.norm(eh.r_eci))
    assert cos_ang < -0.99

    # RAAN drift over one day ~ +0.986 deg/day confirms J2 precession is on.
    # (A Keplerian fallback would give ~0 and fail this.)
    ed = P.get_environment(86400.0)
    drift = (raan_deg(ed.r_eci, ed.v_eci) - raan_deg(e0.r_eci, e0.v_eci) + 180.0) % 360.0 - 180.0
    assert 0.9 < drift < 1.1


@requires_orekit
def test_magnetic_field() -> None:
    """B field has a LEO-plausible magnitude (in Tesla) and varies along the orbit."""
    P.configure(orbit)
    e0 = P.get_environment(0.0)
    mag = np.linalg.norm(e0.b_field_eci)

    # ~20,000-50,000 nT at LEO = 2e-5 to 5e-5 T. This band also catches a
    # nanoTesla/Tesla (1e9) unit slip: unconverted it would be ~3e4, far outside.
    assert 1.0e-5 < mag < 6.0e-5

    # Field changes as the satellite moves around the orbit.
    mu = 3.986004418e14
    a = 6378137.0 + orbit.altitude_km * 1000.0
    period = 2.0 * np.pi * np.sqrt(a**3 / mu)
    bq = P.get_environment(period / 4.0).b_field_eci
    assert np.linalg.norm(bq - e0.b_field_eci) > 1.0e-6


def test_simulation_config_rejects_nonpositive_step() -> None:
    """step_s must be positive — a zero step would divide by zero downstream."""
    with pytest.raises(ValueError):
        SimulationConfig(step_s=0.0)
    with pytest.raises(ValueError):
        SimulationConfig(step_s=-1.0)


class TestMagnetometer:
    """Physics invariants for the magnetometer measurement model."""

    _R = np.array([ALT_RADIUS, 0.0, 0.0])
    _B = np.array([2.0e-5, 1.0e-5, -3.0e-5])
    _SUN = np.array([1.0, 0.0, 0.0])

    def _env(self) -> P.EnvironmentData:
        return _sample_env(self._R, np.zeros(3), self._B, self._SUN)

    def _config(self, **kwargs) -> MagnetometerConfig:
        """A zero-error magnetometer, overridable field by field."""
        base = dict(
            name="test",
            body_to_sensor=np.eye(3),
            noise_std=np.zeros(3),
            bias=np.zeros(3),
            update_rate_hz=5.0,
            measurement_range=8e-4,
        )
        base.update(kwargs)
        return MagnetometerConfig(**base)

    def test_shape_and_dtype(self) -> None:
        """Returns a (3,) float array."""
        b = read_magnetometer(
            _sample_state(np.array([1.0, 0.0, 0.0, 0.0]), self._R, np.zeros(3)),
            self._env(),
            self._config(),
            np.random.default_rng(0),
        )
        assert b.shape == (3,)
        assert b.dtype == np.float64

    def test_noise_free_identity_mounting_is_truth(self) -> None:
        """Zero noise, zero bias, identity attitude and mounting: measures B_eci."""
        state = _sample_state(np.array([1.0, 0.0, 0.0, 0.0]), self._R, np.zeros(3))
        b = read_magnetometer(state, self._env(), self._config(), np.random.default_rng(0))
        assert np.allclose(b, self._B)

    def test_rotation_composition(self) -> None:
        """The measurement is R_S @ C(q) @ B_eci, in that order.

        Catches a reversed composition (C @ R_S). A transposed DCM would pass
        here, since the expected value uses dcm_eci_to_body too - see
        test_dcm_eci_to_body_convention for that.
        """
        q = np.array([0.98, 0.10, -0.05, 0.15])
        q = q / np.linalg.norm(q)
        r_s = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        state = _sample_state(q, self._R, np.zeros(3))
        b = read_magnetometer(
            state, self._env(), self._config(body_to_sensor=r_s), np.random.default_rng(0)
        )
        assert np.allclose(b, r_s @ dcm_eci_to_body(q) @ self._B)

    def test_mounting_preserves_magnitude(self) -> None:
        """A rotated mounting sees the same physical field, expressed differently."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        theta = np.deg2rad(37.0)
        r_s = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        state = _sample_state(q, self._R, np.zeros(3))
        env, rng = self._env(), np.random.default_rng(0)
        straight = read_magnetometer(state, env, self._config(), rng)
        rotated = read_magnetometer(state, env, self._config(body_to_sensor=r_s), rng)
        assert np.isclose(np.linalg.norm(straight), np.linalg.norm(rotated))
        assert not np.allclose(straight, rotated)

    def test_bias_is_additive_in_sensor_frame(self) -> None:
        """Bias adds after rotation, so it shifts the measurement by exactly b_0."""
        bias = np.array([1e-7, -2e-7, 3e-7])
        state = _sample_state(np.array([1.0, 0.0, 0.0, 0.0]), self._R, np.zeros(3))
        env = self._env()
        clean = read_magnetometer(state, env, self._config(), np.random.default_rng(0))
        biased = read_magnetometer(
            state, env, self._config(bias=bias), np.random.default_rng(0)
        )
        assert np.allclose(biased - clean, bias)

    def test_seeded_reproducibility(self) -> None:
        """Identical seeds give identical noise; different seeds do not."""
        state = _sample_state(np.array([1.0, 0.0, 0.0, 0.0]), self._R, np.zeros(3))
        env = self._env()
        cfg = self._config(noise_std=np.full(3, 1e-8))
        a = read_magnetometer(state, env, cfg, np.random.default_rng(42))
        b = read_magnetometer(state, env, cfg, np.random.default_rng(42))
        c = read_magnetometer(state, env, cfg, np.random.default_rng(43))
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_noise_statistics(self) -> None:
        """Over many samples the mean approaches truth and the spread noise_std.

        This is the only test that would catch a sigma-convention error, e.g.
        feeding a 3-sigma datasheet figure into a 1-sigma field.
        """
        sigma = np.array([1e-8, 2e-8, 3e-8])
        state = _sample_state(np.array([1.0, 0.0, 0.0, 0.0]), self._R, np.zeros(3))
        env, cfg = self._env(), self._config(noise_std=sigma)
        rng = np.random.default_rng(7)
        samples = np.array([read_magnetometer(state, env, cfg, rng) for _ in range(20000)])
        assert np.allclose(samples.mean(axis=0), self._B, atol=5.0 * sigma / np.sqrt(20000))
        assert np.allclose(samples.std(axis=0), sigma, rtol=0.05)

    def test_saturation_clips_to_range(self) -> None:
        """A field beyond the measurement range comes back clipped, not wrapped."""
        huge = np.array([2e-3, -5e-3, 1e-2])   # well beyond +/-800 uT
        state = _sample_state(np.array([1.0, 0.0, 0.0, 0.0]), self._R, np.zeros(3))
        env = _sample_env(self._R, np.zeros(3), huge, self._SUN)
        b = read_magnetometer(state, env, self._config(), np.random.default_rng(0))
        assert np.allclose(b, np.array([8e-4, -8e-4, 8e-4]))

    def test_eventsat_units_are_plausible(self) -> None:
        """The three configured units read a LEO-plausible field.

        Guards the 3-sigma conversion: at 50 nT/120 nT as 1-sigma the noise would be
        three times larger, which this does not catch - see test_noise_statistics
        - but a gross unit slip in measurement_range or noise_std would show here.
        """
        state = _sample_state(np.array([1.0, 0.0, 0.0, 0.0]), self._R, np.zeros(3))
        env, rng = self._env(), np.random.default_rng(0)
        for cfg in sensors.magnetometers:
            b = read_magnetometer(state, env, cfg, rng)
            assert 1.0e-5 < np.linalg.norm(b) < 6.0e-5

def test_dcm_eci_to_body_convention() -> None:
    """C(q) satisfies v_body = C @ v_eci for a hand-computed rotation.

    Pins the frame convention rather than assuming it: a 90 deg rotation about
    ECI Z carries body X onto ECI +Y, so an inertially fixed [1,0,0] reads
    [0,-1,0] in body coordinates. Every other test computes its expected value
    with dcm_eci_to_body itself, so a transposed C would pass them all -
    consistently, across dynamics, actuators and sensors.
    """
    s = np.sqrt(0.5)
    q = np.array([s, 0.0, 0.0, s])          # 90 deg about ECI Z
    c = dcm_eci_to_body(q)
    assert np.allclose(c @ np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))
    assert np.allclose(c @ np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(c @ np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(c @ c.T, np.eye(3))
    assert np.isclose(np.linalg.det(c), 1.0)   # proper rotation, not a reflection


def test_initial_state_has_real_orbit(history: List[SatState]) -> None:
    """history[0] carries the propagator's orbit state, not zeros."""
    assert np.linalg.norm(history[0].r_eci) == pytest.approx(ALT_RADIUS, rel=1e-3)
    assert np.linalg.norm(history[0].v_eci) > 7000.0

def test_get_environment_requires_configure(monkeypatch) -> None:
    """No neutral default environment: an unconfigured propagator fails loudly."""
    monkeypatch.setattr(P, "_ctx", None)
    with pytest.raises(RuntimeError):
        P.get_environment(0.0)
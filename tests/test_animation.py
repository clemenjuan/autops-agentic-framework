"""Tests for the animated ground track module."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.orchestration.animation import (
    _ground_station_footprint_deg,
    _propagate_keplerian,
    _reconstruct_orbital_elements,
)


# ---------------------------------------------------------------------------
# Minimal scenario config used across tests
# ---------------------------------------------------------------------------

_SCENARIO = {
    "orbit": {
        "altitude_km": 400.0,
        "inclination_deg": 97.4,
        "eccentricity": 0.001,
        "raan_deg": 0.0,
        "arg_perigee_deg": 0.0,
        "true_anomaly_deg": 0.0,
        "launch_lottery": True,
    },
    "communications": {
        "ground_station": {
            "latitude_deg": 48.0483,
            "longitude_deg": 11.6567,
            "min_elevation_deg": 10.0,
        }
    },
}

_SEED = 42
_EP = 0


class TestReconstructOrbitalElements:
    def test_matches_environment_rng(self):
        """_reconstruct_orbital_elements must draw the same RAAN/ArgP/TA as the env.

        The environment calls ``random.seed(seed)`` then three
        ``random.uniform(0, 360)`` draws.  We replicate this here.
        """
        seed = _SEED + _EP
        random.seed(seed)
        expected_raan = random.uniform(0, 360)
        expected_argp = random.uniform(0, 360)
        expected_ta = random.uniform(0, 360)

        result = _reconstruct_orbital_elements(_SCENARIO, _SEED, _EP)

        assert result["raan_deg"] == pytest.approx(expected_raan, abs=1e-12)
        assert result["arg_perigee_deg"] == pytest.approx(expected_argp, abs=1e-12)
        assert result["true_anomaly_deg"] == pytest.approx(expected_ta, abs=1e-12)

    def test_no_lottery_returns_config_values(self):
        scenario = {"orbit": {"raan_deg": 42.0, "arg_perigee_deg": 10.0,
                               "true_anomaly_deg": 180.0, "launch_lottery": False}}
        result = _reconstruct_orbital_elements(scenario, 99, 3)
        assert result["raan_deg"] == pytest.approx(42.0)
        assert result["arg_perigee_deg"] == pytest.approx(10.0)
        assert result["true_anomaly_deg"] == pytest.approx(180.0)

    def test_different_episodes_give_different_elements(self):
        r0 = _reconstruct_orbital_elements(_SCENARIO, _SEED, 0)
        r1 = _reconstruct_orbital_elements(_SCENARIO, _SEED, 1)
        assert r0["raan_deg"] != pytest.approx(r1["raan_deg"])


class TestKeplerianPropagation:
    def test_lat_in_range(self):
        orbit = _reconstruct_orbital_elements(_SCENARIO, _SEED, _EP)
        lats, lons = _propagate_keplerian(orbit, n_steps=1000, step_s=60.0)
        assert lats.shape == (1000,)
        assert np.all(lats >= -90.0) and np.all(lats <= 90.0)

    def test_lon_in_range(self):
        orbit = _reconstruct_orbital_elements(_SCENARIO, _SEED, _EP)
        lats, lons = _propagate_keplerian(orbit, n_steps=1000, step_s=60.0)
        assert lons.shape == (1000,)
        assert np.all(lons >= -180.0) and np.all(lons <= 180.0)

    def test_max_lat_bounded_by_inclination(self):
        """For SSO (97.4°), inclination > 90° → max latitude ≈ 180−97.4 = 82.6°."""
        orbit = _reconstruct_orbital_elements(_SCENARIO, _SEED, _EP)
        lats, _ = _propagate_keplerian(orbit, n_steps=2000, step_s=60.0)
        # sin(lat) = sin(inc)*sin(u) → |lat| ≤ min(inc, 180°-inc) for SSO
        max_lat = abs(lats).max()
        assert max_lat <= 90.0

    def test_period_roughly_correct(self):
        """Satellite should complete approximately 1 orbit in ~5554 seconds."""
        orbit = {
            "altitude_km": 400.0, "inclination_deg": 97.4,
            "raan_deg": 0.0, "arg_perigee_deg": 0.0, "true_anomaly_deg": 0.0,
        }
        lats, lons = _propagate_keplerian(orbit, n_steps=6000, step_s=1.0)
        # Find the first zero-crossing of latitude (ascending node) after t=0
        zero_crossings = np.where(np.diff(np.sign(lats)) > 0)[0]
        assert len(zero_crossings) >= 2
        period_s = zero_crossings[1] - zero_crossings[0]
        # Period at 400 km ≈ 5554 s; allow ±5%
        assert 5000 < period_s < 6100, f"Estimated period {period_s}s outside [5000, 6100]"


class TestFootprintGeometry:
    def test_400km_10deg_approx_14deg(self):
        """Known result: 400 km altitude, 10° min elevation → ~14° footprint radius."""
        fp = _ground_station_footprint_deg(400.0, 10.0)
        assert 12.0 < fp < 16.0, f"Footprint {fp:.1f}° outside expected range"

    def test_larger_altitude_larger_footprint(self):
        fp_low = _ground_station_footprint_deg(400.0, 10.0)
        fp_high = _ground_station_footprint_deg(800.0, 10.0)
        assert fp_high > fp_low

    def test_higher_elevation_smaller_footprint(self):
        fp_10 = _ground_station_footprint_deg(400.0, 10.0)
        fp_30 = _ground_station_footprint_deg(400.0, 30.0)
        assert fp_30 < fp_10


class TestAnimateSmokeTest:
    """Smoke test: generate a small animation without saving, verify object returned."""

    def _make_step_df(self, n_steps: int = 50) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        modes = np.random.choice(
            ["charging", "payload_observe", "communication"], size=n_steps
        )
        return pd.DataFrame({
            "experiment_id": "test_exp",
            "episode_id": 0,
            "step": np.arange(n_steps),
            "resolved_mode": modes,
            "battery_soc": rng.uniform(0.4, 1.0, n_steps),
            "anomaly": [None] * n_steps,
            "ground_pass_active": rng.integers(0, 2, n_steps),
            "data_downlinked_mb": np.cumsum(rng.uniform(0, 0.5, n_steps)),
        })

    def _make_results_raw(self) -> Dict[str, Any]:
        return {
            "experiment_id": "test_exp",
            "config": {
                "seed": 42,
                "environment": {
                    "timestep_seconds": 60,
                    "scenario_config": {},
                },
            },
        }

    def test_returns_func_animation(self, tmp_path: Path):
        """Generate a tiny GIF to ensure the animation renders without error."""
        import matplotlib
        matplotlib.use("Agg")
        pytest.importorskip("PIL", reason="pillow required for GIF smoke test")
        from matplotlib.animation import FuncAnimation
        from src.orchestration.animation import animate_ground_track

        step_df = self._make_step_df(50)
        results_raw = self._make_results_raw()
        out = tmp_path / "smoke.gif"

        anim = animate_ground_track(
            step_df, results_raw,
            episode_id=0,
            duration_steps=50,
            step_stride=10,   # 5 frames total — fast
            fps=5,
            save_path=out,
        )
        assert isinstance(anim, FuncAnimation)
        assert out.exists()


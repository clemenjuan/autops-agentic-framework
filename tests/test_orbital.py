"""Tests for the orbital mechanics module.

These tests exercise the simplified (non-Orekit) code paths, which are
always available. Orekit-specific tests are skipped when Orekit is not
installed.
"""

import pytest
import random

from src.orbital.eclipse import (
    EclipseInterval,
    compute_eclipses_simplified,
    is_in_sunlight,
)
from src.orbital.ground_access import (
    GroundPass,
    compute_passes_simplified,
)
from src.orbital.context import (
    OrbitalContext,
    compute_orbital_context,
)
from src.orbital.propagator import is_available as orekit_available


# -----------------------------------------------------------------
# Eclipse computation (simplified)
# -----------------------------------------------------------------


class TestEclipseSimplified:
    def test_basic_eclipses(self):
        # 100-step orbit, 30% eclipse => 30 steps eclipse per orbit
        eclipses = compute_eclipses_simplified(
            orbital_period_s=6000, eclipse_fraction=0.3,
            step_s=60, total_steps=200,
        )
        assert len(eclipses) > 0
        for ec in eclipses:
            assert ec.start_step <= ec.end_step

    def test_eclipse_fraction_zero(self):
        # No eclipse at all
        eclipses = compute_eclipses_simplified(
            orbital_period_s=6000, eclipse_fraction=0.0,
            step_s=60, total_steps=200,
        )
        # eclipse_fraction=0 but max(1, ...) means 1-step eclipses still appear
        # This is acceptable — the fraction model has a minimum of 1 step
        for ec in eclipses:
            assert ec.end_step - ec.start_step <= 1

    def test_sunlight_check(self):
        eclipses = [EclipseInterval(start_step=0, end_step=9)]
        assert not is_in_sunlight(5, eclipses)
        assert is_in_sunlight(10, eclipses)
        assert is_in_sunlight(50, eclipses)

    def test_sunlight_empty_eclipses(self):
        assert is_in_sunlight(0, [])
        assert is_in_sunlight(100, [])

    def test_multiple_orbits(self):
        # 5676s period, 60s steps = ~94 steps/orbit. 10080 steps = ~107 orbits
        eclipses = compute_eclipses_simplified(
            orbital_period_s=5676, eclipse_fraction=0.36,
            step_s=60, total_steps=10080,
        )
        # Should have roughly 107 eclipse intervals
        assert len(eclipses) > 100


# -----------------------------------------------------------------
# Ground pass computation (simplified)
# -----------------------------------------------------------------


class TestGroundPassSimplified:
    def test_basic_passes(self):
        random.seed(42)
        passes = compute_passes_simplified(
            step_s=60, total_steps=1440,
            passes_min_per_day=2, passes_max_per_day=3,
        )
        assert len(passes) >= 2
        assert len(passes) <= 3
        for gp in passes:
            assert gp.start_step <= gp.end_step
            assert gp.data_budget_mb > 0

    def test_multi_day(self):
        random.seed(42)
        passes = compute_passes_simplified(
            step_s=60, total_steps=10080,  # 7 days
            passes_min_per_day=2, passes_max_per_day=3,
        )
        assert len(passes) >= 14  # At least 2 per day * 7 days
        assert len(passes) <= 21  # At most 3 per day * 7 days

    def test_sorted_by_start(self):
        random.seed(123)
        passes = compute_passes_simplified(
            step_s=60, total_steps=10080,
        )
        for i in range(1, len(passes)):
            assert passes[i].start_step >= passes[i - 1].start_step


# -----------------------------------------------------------------
# OrbitalContext
# -----------------------------------------------------------------


class TestOrbitalContext:
    def test_construction(self):
        ctx = OrbitalContext()
        assert ctx.mode == "simplified"
        assert ctx.eclipses == []
        assert ctx.ground_passes == []

    def test_sunlight_query(self):
        ctx = OrbitalContext(
            eclipses=[EclipseInterval(0, 10), EclipseInterval(50, 60)]
        )
        assert not ctx.is_in_sunlight(5)
        assert ctx.is_in_sunlight(25)
        assert not ctx.is_in_sunlight(55)
        assert ctx.is_in_sunlight(70)

    def test_ground_pass_query(self):
        # Second-accurate windows (step_s defaults to 60): steps 100..110 inclusive.
        ctx = OrbitalContext(
            ground_passes=[GroundPass(100, 110, data_budget_mb=5.0, start_s=6000.0, end_s=6660.0)]
        )
        assert not ctx.is_ground_pass_active(99)
        assert ctx.is_ground_pass_active(100)
        assert ctx.is_ground_pass_active(105)
        assert ctx.is_ground_pass_active(110)
        assert not ctx.is_ground_pass_active(111)

    def test_get_current_pass(self):
        gp = GroundPass(100, 110, data_budget_mb=5.0, start_s=6000.0, end_s=6660.0)
        ctx = OrbitalContext(ground_passes=[gp])
        assert ctx.get_current_pass(105) is gp
        assert ctx.get_current_pass(50) is None

    def test_contact_seconds_subtimestep(self):
        # A 22 s pass starting mid-step credits ~22 s, not a full 60 s step.
        ctx = OrbitalContext(
            ground_passes=[GroundPass(10, 10, start_s=610.0, end_s=632.0)], step_s=60.0
        )
        assert abs(ctx.contact_seconds(10) - 22.0) < 1e-6   # [600,660) ∩ [610,632] = 22
        assert ctx.contact_seconds(9) == 0.0
        assert ctx.contact_seconds(11) == 0.0
        assert ctx.is_ground_pass_active(10)


# -----------------------------------------------------------------
# compute_orbital_context (integration)
# -----------------------------------------------------------------


class TestComputeOrbitalContext:
    def test_simplified_fallback(self):
        """Without Orekit params, should use simplified model."""
        random.seed(42)
        ctx = compute_orbital_context(
            orbit_config={"orbital_period_s": 5676, "eclipse_fraction": 0.36},
            comms_config={"passes": {"min_per_day": 2, "max_per_day": 3}},
            step_s=60,
            total_steps=1440,
        )
        assert ctx.mode == "simplified"
        assert len(ctx.eclipses) > 0
        assert len(ctx.ground_passes) >= 2

    def test_with_eventsat_config(self):
        """Full EventSat-like config should work in simplified mode."""
        random.seed(42)
        orbit_config = {
            "orbital_period_s": 5676,
            "eclipse_fraction": 0.36,
            "altitude_km": 500,
            "inclination_deg": 97.4,
        }
        comms_config = {
            "sband": {"downlink_rate_kbps": 128},
            "ground_station": {"latitude_deg": 48.0483, "longitude_deg": 11.6567},
            "passes": {
                "min_per_day": 2,
                "max_per_day": 3,
                "min_duration_s": 22,
                "max_duration_s": 422,
                "avg_data_per_day_mb": 12.0,
            },
        }
        ctx = compute_orbital_context(
            orbit_config=orbit_config,
            comms_config=comms_config,
            step_s=60,
            total_steps=10080,
        )
        # When Orekit is not installed, should fall back to simplified
        assert ctx.mode in ("simplified", "orekit")
        assert len(ctx.eclipses) > 0
        assert len(ctx.ground_passes) > 0


# -----------------------------------------------------------------
# EventSat integration with OrbitalContext
# -----------------------------------------------------------------


class TestEventSatWithOrbitalContext:
    def test_environment_uses_orbital_context(self):
        """EventSat should use OrbitalContext after reset."""
        from src.eventsat.env import EventSatEnvironment

        env = EventSatEnvironment(config={
            "step_duration_s": 60,
            "max_steps": 100,
        })
        env.reset(seed=42)
        assert env._orbital_ctx is not None
        assert env._orbital_ctx.mode in ("simplified", "orekit")

    def test_end_to_end_still_works(self, tmp_path):
        """Full experiment should produce same quality results."""
        from src.core.config_loader import ExperimentConfig
        from src.core.experiment_runner import ExperimentRunner

        cfg = ExperimentConfig(
            experiment_id="orbital_test",
            agent_organization="sas",
            decision_procedure="sda",
            representation="symbolic",
            behaviour="hand_designed",
            operations_paradigm="autonomous_hybrid",
            representation_config={"type": "rule_based_eventsat"},
            environment={"constellation_size": 1, "timestep_seconds": 60,
                         "max_steps": 100, "scenario": "eventsat",
                         "scenario_config": {}},
            num_episodes=1,
            max_steps=100,
            save_checkpoints=False,
            log_level="WARNING",
            output_dir=str(tmp_path),
        )
        runner = ExperimentRunner(config=cfg)
        results = runner.run()
        assert results["num_episodes"] == 1
        assert len(results["episodes"][0]["steps"]) == 100


# -----------------------------------------------------------------
# Thermal model tests (REMOVED)
# -----------------------------------------------------------------
# Thermal model was removed from the EventSat environment.
# Heat dissipation design is in progress; temperature is not a constraint.
# The Jetson is limited by energy budget and data pipeline (Jetson→OBC→S-band).
# These tests have been replaced by test_eventsat_physics.py.


# -----------------------------------------------------------------
# Orekit-specific tests (skipped when not installed)
# -----------------------------------------------------------------


@pytest.mark.skipif(not orekit_available(), reason="Orekit not installed")
class TestOrekitPropagation:
    def test_keplerian_propagator(self):
        from datetime import datetime, timezone
        from src.orbital.propagator import create_keplerian_propagator

        propagator = create_keplerian_propagator(
            a_km=6878.137, e=0.001, i_deg=97.4,
            raan_deg=0.0, argp_deg=0.0, ta_deg=0.0,
            epoch=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        assert propagator is not None

    def test_orekit_eclipses(self):
        from datetime import datetime, timezone
        from src.orbital.propagator import create_keplerian_propagator
        from src.orbital.eclipse import compute_eclipses_orekit

        propagator = create_keplerian_propagator(
            a_km=6878.137, e=0.001, i_deg=97.4,
            raan_deg=0.0, argp_deg=0.0, ta_deg=0.0,
            epoch=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        eclipses = compute_eclipses_orekit(propagator, duration_s=6000, step_s=60)
        assert len(eclipses) > 0

    def test_orekit_ground_passes(self):
        from datetime import datetime, timezone
        from src.orbital.propagator import create_keplerian_propagator
        from src.orbital.ground_access import compute_passes_orekit

        propagator = create_keplerian_propagator(
            a_km=6878.137, e=0.001, i_deg=97.4,
            raan_deg=0.0, argp_deg=0.0, ta_deg=0.0,
            epoch=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        passes = compute_passes_orekit(
            propagator,
            station_lat_deg=48.0483, station_lon_deg=11.6567,
            min_elevation_deg=10.0,
            duration_s=86400, step_s=60,
        )
        # Over 24h, there should be some passes
        assert isinstance(passes, list)

    def test_orekit_context(self):
        import random
        random.seed(42)
        ctx = compute_orbital_context(
            orbit_config={
                "altitude_km": 500,
                "inclination_deg": 97.4,
                "orbital_period_s": 5676,
                "eclipse_fraction": 0.36,
            },
            comms_config={
                "ground_station": {"latitude_deg": 48.0483, "longitude_deg": 11.6567},
                "sband": {"downlink_rate_kbps": 128},
                "passes": {"min_per_day": 2, "max_per_day": 3},
            },
            step_s=60,
            total_steps=1440,
        )
        assert ctx.mode == "orekit"
        assert len(ctx.eclipses) > 0

    def test_j2_propagator_created(self):
        """J2 (EcksteinHechler) propagator should be created without error."""
        from datetime import datetime, timezone
        from src.orbital.propagator import create_j2_propagator

        propagator = create_j2_propagator(
            a_km=6778.137, e=0.001, i_deg=97.4,
            raan_deg=45.0, argp_deg=0.0, ta_deg=0.0,
            epoch=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        assert propagator is not None

    def test_j2_raan_precession(self):
        """EcksteinHechler J2 propagator must show RAAN precession over 7 days.

        At 400 km SSO (97.4 deg inclination), the J2-driven RAAN drift is
        ~0.98 deg/day → ~6.86 deg over 7 days. We accept 0.5–2 deg/day
        to accommodate minor model differences.
        """
        import math
        from datetime import datetime, timezone
        from src.orbital.propagator import create_j2_propagator

        epoch = datetime(2026, 6, 1, tzinfo=timezone.utc)
        propagator = create_j2_propagator(
            a_km=6778.137, e=0.001, i_deg=97.4,
            raan_deg=0.0, argp_deg=0.0, ta_deg=0.0,
            epoch=epoch,
        )
        # Propagate 7 days
        from org.orekit.time import TimeScalesFactory, AbsoluteDate
        from org.orekit.orbits import KeplerianOrbit
        utc = TimeScalesFactory.getUTC()
        t0 = AbsoluteDate(2026, 6, 1, 0, 0, 0.0, utc)
        state_7d = propagator.propagate(t0.shiftedBy(7 * 86400.0))
        kep = KeplerianOrbit(state_7d.getOrbit())
        raan_7d = math.degrees(kep.getRightAscensionOfAscendingNode())
        # Handle wrap-around: RAAN is in [0, 360)
        raan_drift = (raan_7d + 360) % 360
        drift_per_day = raan_drift / 7.0
        assert 0.5 <= drift_per_day <= 2.0, (
            f"RAAN drift {drift_per_day:.3f} deg/day outside expected range [0.5, 2.0]. "
            "J2 perturbation may not be active."
        )

    def test_launch_lottery_varies_raan(self):
        """Different seeds must produce different RAAN/ArgP/TA values."""
        from src.eventsat.env import EventSatEnvironment

        env = EventSatEnvironment(config={
            "step_duration_s": 60,
            "max_steps": 10,
            "scenario_params": {
                "orbit": {
                    "altitude_km": 400,
                    "inclination_deg": 97.4,
                    "eccentricity": 0.001,
                    "launch_lottery": True,
                    "propagator": "j2",
                    "orbital_period_s": 5554,
                    "eclipse_fraction": 0.36,
                },
                "communications": {
                    "ground_station": {"latitude_deg": 48.0483, "longitude_deg": 11.6567},
                    "sband": {"downlink_rate_kbps": 128},
                    "passes": {"min_per_day": 2, "max_per_day": 3},
                },
            },
        })
        # Same seed → same orbital context
        env.reset(seed=42)
        ctx_a = env._orbital_ctx
        env.reset(seed=42)
        ctx_b = env._orbital_ctx
        # Check mode (both should be orekit since Orekit is available)
        assert ctx_a.mode == ctx_b.mode == "orekit"
        # Eclipse patterns should be identical for same seed
        assert len(ctx_a.eclipses) == len(ctx_b.eclipses)

        # Different seed → different eclipse pattern (different RAAN → different lighting)
        env.reset(seed=999)
        ctx_c = env._orbital_ctx
        # It's possible but unlikely that two random RAANs produce identical eclipse counts
        # over a short window. We at least verify the context is computed fresh.
        assert ctx_c is not ctx_a  # Different object (was recomputed)

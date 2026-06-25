"""Tests for SSA physics primitives."""
from __future__ import annotations

import math

import pytest

from src.orbital.isl import effective_data_rate_bps, is_isl_feasible, vector_range_km
from src.ssa.targets import (
    detect_targets_in_fov,
    diffraction_limited_range_km,
    generate_sso_catalog,
    propagate_rso_position_km,
)


def test_diffraction_limited_range_matches_autops_rl_optic_payload() -> None:
    assert diffraction_limited_range_km() == pytest.approx(52.7, rel=2e-3)


def test_vector_range_helper_uses_3d_euclidean_distance() -> None:
    assert vector_range_km([0.0, 0.0, 0.0], [3.0, 4.0, 12.0]) == pytest.approx(13.0)


def test_isl_closes_in_range_and_fails_out_of_range() -> None:
    assert is_isl_feasible([0.0, 0.0, 0.0], [1000.0, 0.0, 0.0])
    assert effective_data_rate_bps(1000.0 * 1000.0) > 0.0
    assert not is_isl_feasible([0.0, 0.0, 0.0], [5000.0, 0.0, 0.0])


def test_isl_requires_both_endpoints_idle() -> None:
    assert not is_isl_feasible(
        [0.0, 0.0, 0.0],
        [1000.0, 0.0, 0.0],
        endpoint_a_idle=False,
        endpoint_b_idle=True,
    )


def test_anti_nadir_fov_returns_multiple_targets_without_target_action() -> None:
    observer = (7000.0, 0.0, 0.0)
    angle_rad = math.radians(3.0)
    target_positions = {
        "rso_a": (7020.0, 0.0, 0.0),
        "rso_b": (7000.0 + 20.0 * math.cos(angle_rad), 20.0 * math.sin(angle_rad), 0.0),
        "too_wide": (7000.0 + 20.0 * math.cos(math.radians(8.0)), 20.0 * math.sin(math.radians(8.0)), 0.0),
        "too_far": (7060.0, 0.0, 0.0),
    }

    detections = detect_targets_in_fov(observer, target_positions)

    assert [d.object_id for d in detections] == ["rso_a", "rso_b"]
    assert all(d.quality > 0.0 for d in detections)


def test_synthetic_sso_catalog_is_seeded_and_fixed_size() -> None:
    first = generate_sso_catalog(5, seed=7)
    second = generate_sso_catalog(5, seed=7)
    third = generate_sso_catalog(5, seed=8)

    assert len(first) == 5
    assert first == second
    assert first != third
    assert all(6971.0 <= target.semi_major_axis_km <= 7271.0 for target in first)


def test_target_two_body_propagation_returns_finite_position() -> None:
    target = generate_sso_catalog(1, seed=3)[0]
    position = propagate_rso_position_km(target, 120.0, prefer_orekit=False)

    assert len(position) == 3
    assert all(math.isfinite(value) for value in position)
    assert 6500.0 < vector_range_km([0.0, 0.0, 0.0], position) < 7600.0

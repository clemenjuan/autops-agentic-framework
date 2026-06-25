"""Tests for SSA physics primitives."""
from __future__ import annotations

import math

import pytest

from src.core.config_loader import ExperimentConfig, apply_overrides, load_config
from src.core.experiment_runner import ExperimentRunner
from src.core.satellite_env import scope_observation
from src.orbital.isl import effective_data_rate_bps, is_isl_feasible, vector_range_km
from src.ssa.env import SSAEnvironment, SSA_MODES
from src.ssa.metrics import SSAMetricsCollector
from src.ssa.rewards import SSARewardFunction
from src.ssa.symbolic import RuleBasedSSA
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


def _ssa_env_config(*, n: int = 2, settling_s: float = 0.0) -> dict:
    fixed_positions = {
        "rso_0": [0.0, 0.0, 530.0],
        "rso_1": [0.0, 1.0, 530.0],
    }
    return {
        "constellation_size": n,
        "step_duration_s": 60,
        "max_steps": 20,
        "satellite_positions_km": {
            "sat_0": [0.0, 0.0, 500.0],
            "sat_1": [100.0, 0.0, 500.0],
        },
        "scenario_params": {
            "modes": {
                "transition_overhead": {
                    "settling_time_s": settling_s,
                    "attitude_maneuver_modes": ["payload_observe", "communication"],
                }
            },
            "payload": {"compression_time_factor": 1.0, "detection_time_s": 60.0},
        },
        "targets": {
            "fixed_positions_km": fixed_positions,
            "fov_half_angle_deg": 5.0,
            "max_range_km": 52.7,
        },
        "ground_station": {"always_visible": True},
        "reward_config": {"local_weight": 1.0, "team_weight": 0.0, "collective_weight": 1.0},
    }


def test_ssa_observe_updates_fixed_binary_detection_matrix() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)

    result = env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})

    assert env.detection_matrix == [[1, 1], [0, 0]]
    assert result.info["ssa_onboard_coverage"] == pytest.approx(1.0)
    assert result.info["ssa_delivered_coverage"] == 0.0
    row = result.observation.constellation_state.satellites["sat_0"].metadata["ssa_detection_row"]
    assert row == [1, 1]


def test_one_hot_binary_action_selects_payload_observe_mode() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)
    one_hot = [0] * len(SSA_MODES)
    one_hot[SSA_MODES.index("payload_observe")] = 1

    env.step({"sat_0": one_hot, "sat_1": {"mode": "charging"}})

    assert env.detection_matrix[0] == [1, 1]


def test_adcs_settling_blocks_observation_on_observe_entry() -> None:
    env = SSAEnvironment(_ssa_env_config(settling_s=135.0))
    env.reset(seed=1)

    result = env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})

    assert env.detection_matrix == [[0, 0], [0, 0]]
    assert result.info["per_satellite"]["sat_0"]["in_transition"] is True


def test_onboard_keeps_best_estimate_while_ground_archives_all_records() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)

    env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})
    env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})
    env.step({"sat_0": {"mode": "communication"}, "sat_1": {"mode": "charging"}})

    assert set(env.onboard_estimates["sat_0"]) == {"rso_0", "rso_1"}
    assert len(env.onboard_estimates["sat_0"]) == 2
    assert len(env.ground_archive["rso_0"]) == 2
    assert len(env.ground_archive["rso_1"]) == 2


def test_isl_merge_ors_matrix_and_keeps_higher_quality_estimate() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)
    env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})

    env.step({"sat_0": {"mode": "isl_share"}, "sat_1": {"mode": "charging"}})

    assert env.detection_matrix == [[1, 1], [1, 1]]
    assert set(env.onboard_estimates["sat_1"]) == {"rso_0", "rso_1"}
    assert env.get_metrics()["isl_connectivity"] > 0.0


def test_delivered_utility_credits_only_downlinked_objects() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)

    observe = env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})
    downlink = env.step({"sat_0": {"mode": "communication"}, "sat_1": {"mode": "charging"}})

    assert observe.info["ssa_delivered_coverage"] == 0.0
    assert downlink.info["ssa_delivered_coverage"] == pytest.approx(1.0)
    assert downlink.rewards["sat_0"] > observe.rewards["sat_0"]


def test_ssa_reward_collective_negative_uses_delivered_coverage() -> None:
    rf = SSARewardFunction({"collective_weight": 1.0, "mission_scale": 2.0})

    empty = rf.compute_rewards({"sat_0": 0.0}, {"_global": {"delivered_coverage": 0.0}})
    delivered = rf.compute_rewards({"sat_0": 0.0}, {"_global": {"delivered_coverage": 1.0}})

    assert empty["sat_0"] == pytest.approx(-2.0)
    assert delivered["sat_0"] == pytest.approx(0.0)


def test_ssa_metrics_adds_coverage_duplicate_connectivity_and_m10() -> None:
    collector = SSAMetricsCollector({
        "max_steps": 2,
        "step_duration_s": 60,
        "constellation_size": 2,
        "baseline_utility_n1": 0.5,
    })
    for step, delivered in enumerate((0.0, 1.0)):
        collector.record_step(
            timestep=step,
            wall_clock_seconds=0.01,
            env_state=None,
            actions={},
            rewards={"sat_0": 1.0},
            info={
                "battery_soc": 0.8,
                "prev_battery_soc": 0.81,
                "ssa_onboard_coverage": 1.0,
                "ssa_delivered_coverage": delivered,
                "duplicate_observation_rate": 0.25,
                "mean_revisit_steps": 3.0,
                "isl_connectivity": 0.5,
                "ssa_delivered_objects": delivered * 2,
                "ssa_known_objects": 2,
            },
            decision_metrics={"inference_allowed": True, "has_rationale": True},
        )

    episode = collector.finalise_episode(0)

    assert episode.aggregated["utility"] == 1.0
    assert episode.aggregated["ssa_delivered_coverage"] == 1.0
    assert episode.aggregated["duplicate_observation_rate"] == 0.25
    assert episode.aggregated["eta_scale"] == pytest.approx(1.0)


def test_rule_based_ssa_deconflicts_full_scope_but_local_scope_observes() -> None:
    env = SSAEnvironment({
        **_ssa_env_config(n=2),
        "satellite_positions_km": {
            "sat_0": [0.0, 0.0, 500.0],
            "sat_1": [0.4, 0.0, 500.0],
        },
        "targets": {
            "fixed_positions_km": {"rso_0": [0.0, 0.0, 530.0]},
            "fov_half_angle_deg": 5.0,
            "max_range_km": 52.7,
        },
    })
    observation = env.reset(seed=1)
    central = RuleBasedSSA({})

    central_action = central.select_action(type("Context", (), {"state": central.encode_observation(observation)}))

    assert central_action["sat_0"]["mode"] == "payload_observe"
    assert central_action["sat_1"]["mode"] != "payload_observe"

    step_result = env.step(central_action)
    local = RuleBasedSSA({"satellite_id": "sat_1"})
    scoped = scope_observation(step_result.observation, ["sat_1"])
    local_state = local.encode_observation(scoped)

    assert local.select_action(type("Context", (), {"state": local_state}))["sat_1"]["mode"] == "payload_observe"


def test_ssa_symbolic_runner_sas_deconflicts_imas_duplicates(tmp_path) -> None:
    sas_cfg = apply_overrides(
        load_config("configs/experiments/ssa_sas_ao_symb_n3.yaml"),
        episodes=1,
        steps=8,
        output_dir=str(tmp_path / "sas"),
    )
    imas_cfg = apply_overrides(
        load_config("configs/experiments/ssa_imas_ao_symb_n3.yaml"),
        episodes=1,
        steps=8,
        output_dir=str(tmp_path / "imas"),
    )

    sas = ExperimentRunner(config=sas_cfg).run()
    imas = ExperimentRunner(config=imas_cfg).run()

    sas_dupes = sas["experiment_statistics"].mean["duplicate_observation_rate"]
    imas_dupes = imas["experiment_statistics"].mean["duplicate_observation_rate"]
    assert sas_dupes == 0.0
    assert imas_dupes > 0.0


def test_ssa_ground_paradigms_reject_distributed_organizations() -> None:
    with pytest.raises(ValueError, match="SSA ground paradigms"):
        ExperimentConfig(
            experiment_id="ssa_invalid_ground_imas",
            agent_organization="independent_mas",
            decision_procedure="sda",
            representation="symbolic",
            behaviour="hand_designed",
            operations_paradigm="autonomous_ground",
            behaviour_config={"mode": "hand_designed"},
            environment={
                "constellation_size": 3,
                "timestep_seconds": 60,
                "max_steps": 10,
                "scenario": "ssa",
                "scenario_config": {"scenario_file": "configs/scenarios/ssa.yaml"},
            },
            num_episodes=1,
            max_steps=10,
        )

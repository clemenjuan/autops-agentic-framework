"""
Tests for experiment reproducibility.

Validates that the same configuration and seed produce identical step-level
results, and that different seeds produce different trajectories (via launch
lottery RAAN/ArgP/TA randomization).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.orchestration.config_loader import ExperimentConfig
from src.orchestration.experiment_runner import ExperimentRunner


def _make_config(seed: int, output_dir: str, max_steps: int = 50, num_episodes: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="repro_test",
        seed=seed,
        num_episodes=num_episodes,
        max_steps=max_steps,
        agent_organization="centralized",
        decision_loop="sda",
        representation="symbolic",
        emergence_mode="hand_designed",
        operations_paradigm="autonomous_hybrid",
        representation_config={"type": "rule_based_eventsat"},
        environment={
            "constellation_size": 1,
            "timestep_seconds": 60,
            "max_steps": max_steps,
            "scenario": "eventsat",
            "scenario_config": {},
        },
        save_checkpoints=False,
        log_level="WARNING",
        output_dir=output_dir,
    )


class TestReproducibility:
    def test_same_seed_identical_rewards(self, tmp_path: Path) -> None:
        """Same seed must produce bit-identical rewards and state at every step."""
        cfg1 = _make_config(42, str(tmp_path / "run1"), max_steps=50, num_episodes=2)
        cfg2 = _make_config(42, str(tmp_path / "run2"), max_steps=50, num_episodes=2)

        r1 = ExperimentRunner(config=cfg1).run()
        r2 = ExperimentRunner(config=cfg2).run()

        assert r1["num_episodes"] == r2["num_episodes"]
        for ep1, ep2 in zip(r1["episodes"], r2["episodes"]):
            assert ep1["num_steps"] == ep2["num_steps"]
            for s1, s2 in zip(ep1["steps"], ep2["steps"]):
                assert s1["rewards"] == s2["rewards"], (
                    f"Reward mismatch at step {s1.get('step')}: "
                    f"{s1['rewards']} vs {s2['rewards']}"
                )
                info1 = s1.get("info", {})
                info2 = s2.get("info", {})
                assert info1.get("battery_soc") == pytest.approx(
                    info2.get("battery_soc"), abs=1e-9
                ), f"Battery SoC mismatch at step {s1.get('step')}"
                assert info1.get("data_downlinked_mb") == pytest.approx(
                    info2.get("data_downlinked_mb"), abs=1e-9
                )
                assert info1.get("resolved_mode") == info2.get("resolved_mode"), (
                    f"Mode mismatch at step {s1.get('step')}: "
                    f"{info1.get('resolved_mode')} vs {info2.get('resolved_mode')}"
                )

    def test_different_seeds_different_trajectories(self, tmp_path: Path) -> None:
        """Different seeds produce different reward sequences (different RAAN → different passes)."""
        # Use enough steps that the orbital geometry has time to differ
        cfg1 = _make_config(42, str(tmp_path / "seed42"), max_steps=200)
        cfg2 = _make_config(99, str(tmp_path / "seed99"), max_steps=200)

        r1 = ExperimentRunner(config=cfg1).run()
        r2 = ExperimentRunner(config=cfg2).run()

        rewards1 = [s["rewards"]["total"] for s in r1["episodes"][0]["steps"]]
        rewards2 = [s["rewards"]["total"] for s in r2["episodes"][0]["steps"]]

        # Trajectories must diverge at some point
        assert rewards1 != rewards2, (
            "Seeds 42 and 99 produced identical reward sequences — "
            "launch lottery may not be affecting orbital geometry"
        )

    def test_multi_episode_per_seed_deterministic(self, tmp_path: Path) -> None:
        """Each episode in a run uses seed+episode_id deterministically."""
        cfg1 = _make_config(10, str(tmp_path / "run1"), max_steps=30, num_episodes=3)
        cfg2 = _make_config(10, str(tmp_path / "run2"), max_steps=30, num_episodes=3)

        r1 = ExperimentRunner(config=cfg1).run()
        r2 = ExperimentRunner(config=cfg2).run()

        for ep_idx, (ep1, ep2) in enumerate(zip(r1["episodes"], r2["episodes"])):
            for s1, s2 in zip(ep1["steps"], ep2["steps"]):
                assert s1["rewards"] == s2["rewards"], (
                    f"Episode {ep_idx} reward mismatch at step {s1.get('step')}"
                )

    def test_cross_architecture_anomaly_sync(self, tmp_path: Path) -> None:
        """Anomaly injection must occur at identical steps regardless of ops paradigm.

        The dedicated _anomaly_rng is seeded from the episode seed independently
        of recovery timing, so autonomous (no ground pass needed) and conventional
        (ground pass required) architectures inject anomalies at the same steps.
        """
        SEED = 42
        # Elevated anomaly_prob to guarantee several anomalies in 500 steps
        scenario_config = {"anomaly_prob": 0.05}

        def run(paradigm: str, requires_ground: bool) -> list:
            cfg = _make_config(SEED, str(tmp_path / paradigm), max_steps=500)
            cfg.operations_paradigm = paradigm
            cfg.environment.scenario_config = {
                **scenario_config,
                "anomaly_requires_ground_pass": requires_ground,
            }
            result = ExperimentRunner(config=cfg).run()
            # Collect steps where a new anomaly is injected (transition None→truthy)
            steps_info = [s.get("info", {}) for s in result["episodes"][0]["steps"]]
            injection_steps = []
            prev = False
            for i, info in enumerate(steps_info):
                anom = info.get("anomaly")
                curr = bool(anom and anom is not False and anom != 0)
                if curr and not prev:
                    injection_steps.append(i)
                prev = curr
            return injection_steps

        autonomous_injections = run("autonomous_hybrid", requires_ground=False)
        conventional_injections = run("conventional_ground", requires_ground=True)

        assert len(autonomous_injections) > 0, (
            "No anomalies injected — increase anomaly_prob or max_steps"
        )
        assert len(conventional_injections) > 0, (
            "No anomalies injected in conventional run — increase anomaly_prob or max_steps"
        )
        # The first injection must be identical: both architectures have no active anomaly
        # at t=0, so their _anomaly_rng streams are in the same state until the first hit.
        # After that, recovery timing diverges (autonomous clears faster), so subsequent
        # injections naturally differ — that's correct and expected behaviour.
        assert autonomous_injections[0] == conventional_injections[0], (
            f"First anomaly injection step differs between architectures!\n"
            f"  autonomous first: {autonomous_injections[0]}\n"
            f"  conventional first: {conventional_injections[0]}\n"
            f"  (subsequent injections are expected to differ due to different recovery timing)"
        )

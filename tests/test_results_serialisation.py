"""results.json serialisation contract.

results.json is the compact, experiment-level artifact: it must hold the
aggregated statistics and per-episode AGGREGATED metrics, but NOT the raw
per-step observation/state snapshots (full ``ConstellationState`` dumps) nor
the per-step metric lists. Those balloon the file to multi-GB on long symbolic
runs and merely duplicate the ``decisions_ep*.jsonl`` trace (DEBUG only), which
``scripts/recompute_metrics.py`` reads instead.

See ``ExperimentRunner._strip_per_step_data``.
"""

from __future__ import annotations

import json
from pathlib import Path


def _run_short_experiment(tmp_path, *, episodes: int = 2, steps: int = 30):
    from src.orchestration.config_loader import ExperimentConfig
    from src.orchestration.experiment_runner import ExperimentRunner

    cfg = ExperimentConfig(
        experiment_id="results_serialisation_test",
        agent_organization="sas",
        decision_procedure="ooda",
        representation="symbolic",
        behaviour="hand_designed",
        operations_paradigm="autonomous_hybrid",
        decision_procedure_config={
            "orient_history_window": 10,
            "max_orient_iterations": 1,
        },
        representation_config={"type": "rule_based_eventsat"},
        environment={
            "constellation_size": 1,
            "timestep_seconds": 60,
            "max_steps": steps,
            "scenario": "eventsat",
            "scenario_config": {},
        },
        num_episodes=episodes,
        max_steps=steps,
        save_checkpoints=False,
        log_level="WARNING",
        output_dir=str(tmp_path),
    )
    runner = ExperimentRunner(config=cfg)
    runner.run()
    return tmp_path / "results.json"


class TestResultsJsonStaysCompact:
    def test_results_json_has_no_raw_per_step_dumps(self, tmp_path) -> None:
        """results.json must not embed raw constellation_state / satellites dumps."""
        results_path = _run_short_experiment(tmp_path)
        assert results_path.exists()

        text = results_path.read_text(encoding="utf-8")
        # The raw per-step observation carries a ConstellationState with a
        # per-satellite 'satellites' mapping — neither should reach results.json.
        assert "constellation_state" not in text
        assert "satellites" not in text

        results = json.loads(text)
        # No raw per-step list on episodes, and no per-step metric lists.
        for ep in results["episodes"]:
            assert "steps" not in ep
            em = ep.get("episode_metrics")
            if em is not None:
                assert em.get("step_metrics") == []
        # experiment_statistics keeps its per-episode entries, but those too
        # must be stripped of their per-step metric lists.
        for em in results["experiment_statistics"].get("raw_episodes", []):
            assert em.get("step_metrics") == []

    def test_results_json_keeps_aggregated_metrics(self, tmp_path) -> None:
        """The aggregated stats the board/analysis read must survive stripping."""
        results_path = _run_short_experiment(tmp_path)
        results = json.loads(results_path.read_text(encoding="utf-8"))

        assert results["experiment_id"] == "results_serialisation_test"
        assert results["num_episodes"] == 2
        assert "config" in results
        assert "timestamp" in results
        # Experiment-level mean (consumed by scripts/refresh_board.py).
        assert results["experiment_statistics"]["mean"]
        # Per-episode aggregated metrics (also consumed by the board).
        for ep in results["episodes"]:
            assert ep["episode_metrics"]["aggregated"]

    def test_results_json_is_small(self, tmp_path) -> None:
        """A 2x30 run produces a KB-scale file, not the multi-GB raw dump."""
        results_path = _run_short_experiment(tmp_path)
        size_bytes = results_path.stat().st_size
        # Generous bound: without stripping this would be many MB even at 2x30
        # (full ConstellationState per step). Compact form is well under 1 MB.
        assert size_bytes < 1_000_000, f"results.json too large: {size_bytes} bytes"


class TestCompactTelemetryBlock:
    """Sample episodes keep a compact, scalar-only per-step telemetry block for
    graphs / the Episode inspector — present regardless of log level, and small.
    """

    def test_sample_episodes_carry_telemetry(self, tmp_path) -> None:
        results_path = _run_short_experiment(tmp_path, episodes=2, steps=30)
        results = json.loads(results_path.read_text(encoding="utf-8"))

        # 2 episodes < TELEMETRY_SAMPLE_EPISODES (3) → both carry telemetry.
        expected_keys = {
            "steps", "mode", "gpass", "sunlight", "anomaly",
            "soc", "stored", "downlinked", "jetson_raw", "obc",
        }
        for ep in results["episodes"]:
            tel = ep.get("telemetry")
            assert tel, "sample episode missing telemetry block"
            assert expected_keys.issubset(tel.keys())
            n = len(tel["steps"])
            assert n > 0
            # All series are aligned and scalar (graph-ready).
            for key in expected_keys:
                assert len(tel[key]) == n
            assert all(isinstance(m, str) for m in tel["mode"])
            assert all(isinstance(s, (int, float)) for s in tel["soc"])
            # Telemetry is the WARNING-level run (no DEBUG) — proves it is not
            # gated on the decision-trace flag.
            assert not (tmp_path / "decisions_ep0.jsonl").exists()

    def test_telemetry_only_for_leading_sample(self, tmp_path, monkeypatch) -> None:
        """Episodes beyond the sample window carry no telemetry (keeps file small)."""
        from src.orchestration.experiment_runner import ExperimentRunner

        monkeypatch.setattr(ExperimentRunner, "TELEMETRY_SAMPLE_EPISODES", 1)
        results_path = _run_short_experiment(tmp_path, episodes=2, steps=30)
        results = json.loads(results_path.read_text(encoding="utf-8"))

        assert results["episodes"][0].get("telemetry")
        assert "telemetry" not in results["episodes"][1]


class TestResidentMemoryBounded:
    """Raw per-step data is freed eagerly for episodes beyond the sample window,
    so resident memory does not grow with episode count (parallel/long runs).
    The first TELEMETRY_SAMPLE_EPISODES keep their steps in memory (telemetry +
    the tests that inspect the first few episodes); the rest are stripped.
    """

    def test_in_memory_steps_stripped_past_sample(self, tmp_path, monkeypatch) -> None:
        from src.orchestration.config_loader import ExperimentConfig
        from src.orchestration.experiment_runner import ExperimentRunner

        monkeypatch.setattr(ExperimentRunner, "TELEMETRY_SAMPLE_EPISODES", 2)
        cfg = ExperimentConfig(
            experiment_id="ram_bound_test",
            agent_organization="sas",
            decision_procedure="sda",
            representation="symbolic",
            behaviour="hand_designed",
            operations_paradigm="autonomous_hybrid",
            representation_config={"type": "rule_based_eventsat"},
            environment={"constellation_size": 1, "timestep_seconds": 60,
                         "max_steps": 20, "scenario": "eventsat", "scenario_config": {}},
            num_episodes=4,
            max_steps=20,
            save_checkpoints=False,
            log_level="WARNING",
            output_dir=str(tmp_path),
        )
        results = ExperimentRunner(config=cfg).run()
        eps = results["episodes"]
        # First 2 (sample window) keep full steps + step_metrics in memory.
        for i in (0, 1):
            assert len(eps[i]["steps"]) == 20
            assert len(eps[i]["episode_metrics"].step_metrics) == 20
        # Beyond the window: raw steps dropped and step_metrics cleared in place.
        for i in (2, 3):
            assert "steps" not in eps[i]
            assert eps[i]["episode_metrics"].step_metrics == []
        # Aggregated metrics survive for every episode (the analysis payload).
        for ep in eps:
            assert ep["episode_metrics"].aggregated

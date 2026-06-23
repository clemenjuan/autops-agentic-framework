from __future__ import annotations

import numpy as np

from src.core.config_loader import ExperimentConfig
from src.core.experiment_runner import ExperimentRunner


def _wm_config(tmp_path, repr_type: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=f"test_{repr_type}",
        seed=3,
        agent_organization="sas",
        decision_procedure="sda",
        representation="rl",
        representation_config={
            "type": repr_type,
            "horizon": 3,
            "samples": 8,
            "elites": 2,
            "cem_iterations": 2,
            "mission_mode": "science",
        },
        behaviour="hand_designed",
        behaviour_config={"mode": "hand_designed"},
        operations_paradigm="autonomous_onboard",
        environment={
            "constellation_size": 1,
            "timestep_seconds": 60,
            "max_steps": 6,
            "scenario": "eventsat",
            "scenario_config": {"scenario_file": "configs/scenarios/eventsat.yaml"},
        },
        num_episodes=1,
        max_steps=6,
        output_dir=str(tmp_path / repr_type),
    )


def test_world_model_representations_register_and_smoke_run(tmp_path):
    import src.eventsat.world_model  # noqa: F401
    from src.core.behaviour.controller import _REPRESENTATION_REGISTRY

    assert "lewm_cem_eventsat" in _REPRESENTATION_REGISTRY
    assert "dreamerv3_eventsat" in _REPRESENTATION_REGISTRY

    cfg = _wm_config(tmp_path, "lewm_cem_eventsat")
    results = ExperimentRunner(config=cfg).run()
    mean = results["experiment_statistics"].mean
    assert results["num_episodes"] == 1
    assert mean["candidate_count"] == 8.0
    assert mean["cem_iterations"] == 2.0
    assert mean["planner_latency_s"] >= 0.0


def test_eventsat_world_model_trace_schema(tmp_path):
    from src.eventsat.env import EventSatEnvironment
    from src.eventsat.trace import (
        ACTION11_NAMES,
        OBS25_NAMES,
        STATE_NAMES,
        WorldModelTraceEpisode,
    )

    env = EventSatEnvironment(
        {
            "scenario_file": "configs/scenarios/eventsat.yaml",
            "max_steps": 4,
            "step_duration_s": 60,
            "anomaly_prob": 0.0,
        }
    )
    obs = env.reset(seed=11)
    action = {"eventsat_0": {"mode": "charging", "data_priority": 0, "pipeline_routing": 0}}
    result = env.step(action)

    trace = WorldModelTraceEpisode(episode_id=0, seed=11)
    trace.record(step=0, observation=obs, env_actions=action, rewards=result.rewards, info=result.info)
    arrays = trace.as_arrays()

    assert arrays["obs"].shape == (1, len(OBS25_NAMES))
    assert arrays["action"].shape == (1, len(ACTION11_NAMES))
    assert arrays["state"].shape == (1, len(STATE_NAMES))
    assert np.isfinite(arrays["obs"]).all()
    assert np.isclose(arrays["action"].sum(), 3.0)

    out = tmp_path / "episode.npz"
    trace.write_npz(out)
    loaded = np.load(out)
    assert loaded["obs"].shape == arrays["obs"].shape

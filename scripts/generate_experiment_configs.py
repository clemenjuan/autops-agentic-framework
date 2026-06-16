"""
Generate the EventSat·SAS experiment matrix — the 32 O-framework experiments
(morphological_matrix.md §4): conventional 1 + ag 7 + ao 3 + ah 21.

This script is the single source of truth for the matrix. It (re)writes every
``eventsat_sas_<paradigm>_<rep>`` config into ``configs/experiments/`` from
per-cell fragments, so the set is complete, consistent, and reproducible.

  paradigm: conventional | ag | ao | ah
  rep (7 cells): symb · rl · hrl · llm-s · llm-a · hllm-s · hllm-a
  onboard (ao, ah-onboard) ∈ {symb, rl, hrl} (no per-step LLM onboard)
  ah names both cores onboard-first: eventsat_sas_ah_<onboard>_<ground>

Cells without a real core yet (hrl, llm-s, llm-a) resolve to documented
placeholders (is_placeholder=True); they are part of the named matrix.

Usage::

    uv run python scripts/generate_experiment_configs.py            # → configs/experiments/
    uv run python scripts/generate_experiment_configs.py --output-dir /tmp/check
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

CELLS = ["symb", "rl", "hrl", "llm-s", "llm-a", "hllm-s", "hllm-a"]
ONBOARD_CELLS = ["symb", "rl", "hrl"]  # no per-step LLM onboard
PARADIGM_OPS = {
    "conventional": "conventional_ground",
    "ag": "autonomous_ground",
    "ao": "autonomous_onboard",
    "ah": "autonomous_hybrid",
}
AGENTIC_CELLS = {"llm-a", "hllm-a"}
HYBRID_CELLS = {"hllm-s", "hllm-a"}  # LLM + symbolic safety shield (needs scenario physics)

_POWER_CONSUMPTION = {
    "charging": {"sun_w": 4.72, "eclipse_w": 4.32},
    "communication": {"sun_w": 33.65, "eclipse_w": 33.24},
    "payload_observe": {"sun_w": 17.94, "eclipse_w": 17.55},
    "payload_compress": {"sun_w": 12.77, "eclipse_w": 12.37},
    "payload_detect": {"sun_w": 12.77, "eclipse_w": 12.37},
    "payload_send": {"sun_w": 12.77, "eclipse_w": 12.37},
    "safe": {"sun_w": 9.58, "eclipse_w": 9.58},
}


def _symbolic_power_block(conventional: bool) -> dict:
    """EventSat power/scenario params the symbolic schedule planner needs.
    The conventional planner additionally models human cognitive constraints."""
    block: dict = {
        "solar_generation_w": 24.0, "battery_capacity_wh": 70.0, "eclipse_fraction": 0.36,
        "step_duration_s": 60.0, "charge_efficiency": 0.9, "compression_time_factor": 2.0,
        "detection_steps": 5, "observation_size_mb": 9.41, "compression_ratio": 5.11,
        "jetson_to_obc_rate_kbps": 8000, "obc_capacity_mb": 4096.0,  # CAN ~1 MB/s; OBC 4 GB (match scenario)
        "charge_reserve_fraction": 0.12, "min_soc_for_operations": 0.4,
        "staleness_threshold": 5, "settling_time_steps": 2,
    }
    if conventional:
        block.update({
            "conservative_margin": 1.3, "planning_horizon_discount": 0.85,
            "max_observations_per_gap": 2, "shift_handover_probability": 0.1,
            "shift_handover_soc_penalty": 0.1,
        })
    block["power_consumption"] = {k: dict(v) for k, v in _POWER_CONSUMPTION.items()}
    return block


def _rl_block() -> dict:
    return {
        "rl_mock": False, "deterministic": False, "checkpoint_path": "",
        "orbital_period_steps": 94, "max_steps": 10080, "compression_time_factor": 2.0,
        "detection_steps": 5, "jetson_capacity_mb": 249036.8,
    }


def _llm_block(agentic: bool) -> dict:
    block = {
        "llm_provider": "auto", "llm_model": "qwen3.6:35b", "openai_model": "gpt-4o-mini",
        "llm_temperature": 0.0, "llm_mock": False, "llm_cache_dir": "data/llm_cache",
    }
    if agentic:
        block["max_agentic_steps"] = 5
    return block


_PPO_BLOCK = {
    "mechanism": "ppo", "mode": "emergent", "rollout_fragment": 128, "lr": 0.0001,
    "lr_schedule": [[0, 0.0001], [3000000, 1.0e-05]], "gamma": 0.97, "gae_lambda": 0.95,
    "clip_ratio": 0.3, "ppo_epochs": 30, "entropy_coef": 0.01, "value_coef": 1.0,
    "max_grad_norm": 0.5, "minibatch_size": 256,
}


def repr_config(cell: str, role: str, paradigm: str) -> dict:
    """representation_config for a cell in a given role (onboard | ground)."""
    if cell == "symb":
        return {} if role == "onboard" else _symbolic_power_block(paradigm == "conventional")
    if cell == "rl":
        return _rl_block()
    if cell == "hrl":
        return {}  # placeholder → symbolic stand-in
    # Hybrid LLM ground cells (hllm-s / hllm-a) run a symbolic SAFETY shield over the
    # LLM schedule, which needs the scenario physics (OBC capacity, SoC/power model).
    # Pure LLM cells (llm-s / llm-a) have no shield, so only the LLM block.
    if cell in HYBRID_CELLS:
        return {**_symbolic_power_block(paradigm == "conventional"),
                **_llm_block(agentic=cell in AGENTIC_CELLS)}
    return _llm_block(agentic=cell in AGENTIC_CELLS)  # llm-s / llm-a


def _is_learned_rl(cell: str) -> bool:
    return cell == "rl"


def _ops_config(paradigm: str) -> dict:
    if paradigm in ("conventional", "ag"):
        return {"default_mode": "charging", "orbital_period_steps": 93}
    if paradigm == "ah":
        return {"orbital_period_steps": 93}
    return {}  # ao: onboard-only


def _common(experiment_id: str, description: str, paradigm: str, learned_rl: bool) -> dict:
    """Scaffold shared by every config (canonical key order)."""
    behaviour = "emergent" if learned_rl else "hand_designed"
    behaviour_config = dict(_PPO_BLOCK) if learned_rl else {"mode": "hand_designed"}
    return {
        "experiment_id": experiment_id,
        "description": description,
        "seed": 42,
        "agent_organization": "sas",
        "decision_procedure": "sda",
        "behaviour": behaviour,
        "operations_paradigm": PARADIGM_OPS[paradigm],
        "agent_organization_config": {},
        "decision_procedure_config": {"sense_encode": True, "decide_act": True},
        "behaviour_config": behaviour_config,
        "operations_paradigm_config": _ops_config(paradigm),
        "environment": {
            "constellation_size": 1, "timestep_seconds": 60, "max_steps": 10080,
            "scenario": "eventsat",
            "scenario_config": {"scenario_file": "configs/scenarios/eventsat.yaml"},
        },
        "memory_config": {"history_depth": 100},
        "num_episodes": 5,
        "max_steps": 10080,
        "metrics": {
            "enabled": ["utility", "latency", "resource_efficiency", "operator_load", "explainability"],
            "collection_frequency": "per_step",
        },
        "output_dir": "data/results/${experiment_id}",
        "save_checkpoints": False,
        "log_level": "INFO",
    }


def build_matrix() -> dict[str, dict]:
    """Return {experiment_id: config_dict} for the full 32-experiment matrix."""
    configs: dict[str, dict] = {}

    # conventional (symbolic only), ag (7), ao (3) — single-core
    for paradigm, cells in (("conventional", ["symb"]), ("ag", CELLS), ("ao", ONBOARD_CELLS)):
        role = "onboard" if paradigm == "ao" else "ground"
        for cell in cells:
            eid = f"eventsat_sas_{paradigm}_{cell}"
            desc = f"EventSat \xB7 SAS \xB7 {paradigm} \xB7 {cell}.\n"
            cfg = _common(eid, desc, paradigm, _is_learned_rl(cell))
            cfg["representation"] = cell
            cfg["representation_config"] = repr_config(cell, role, paradigm)
            configs[eid] = cfg

    # ah (21) — dual-core: onboard {symb, rl, hrl} × ground {7}
    for onboard in ONBOARD_CELLS:
        for ground in CELLS:
            eid = f"eventsat_sas_ah_{onboard}_{ground}"
            desc = f"EventSat \xB7 SAS \xB7 AH \xB7 {onboard} onboard \xB7 {ground} ground.\n"
            cfg = _common(eid, desc, "ah", _is_learned_rl(onboard))
            cfg["onboard"] = {"representation": onboard,
                              "representation_config": repr_config(onboard, "onboard", "ah")}
            cfg["ground"] = {"representation": ground,
                             "representation_config": repr_config(ground, "ground", "ah")}
            configs[eid] = cfg

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 32 EventSat·SAS experiment configs.")
    parser.add_argument("--output-dir", default="configs/experiments/",
                        help="Directory to write configs into (default: configs/experiments/).")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    configs = build_matrix()
    for eid, cfg in configs.items():
        with open(out / f"{eid}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=False)

    counts: dict[str, int] = {}
    for eid in configs:
        counts[eid.split("_")[2]] = counts.get(eid.split("_")[2], 0) + 1
    print(f"Wrote {len(configs)} configs to {out} — "
          + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()

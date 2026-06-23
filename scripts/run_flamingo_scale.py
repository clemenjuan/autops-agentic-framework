"""Run the Flamingo organisation **scale sweep** (M-10) into ``data/results/``.

M-10 scale efficiency = ``(U(N) / N) / U(1)``: does per-satellite productivity
hold as the constellation grows, or does coordination overhead erode it?

For each organisation we run N in {3, 6, 12} plus a shared N = 1 anchor, all on
the stochastic scenario with paired seeds (``seed = 42``) so the catalogs match
across organisations at each N. The RSO catalog **scales with N** (``count =
2·N``): this holds the per-satellite task load roughly constant so M-10 isolates
coordination cost rather than mere target scarcity. Contention is on
(``satellite_phase_shift = 0`` → the constellation shares visibility windows),
which is what lets an uncoordinated organisation waste capacity as it grows.

Result ids: ``flamingo_<org>_ag_symb_n<N>`` (the no-suffix configs remain the
canonical N = 3 organisation sweep used by board section 1). After running,
refresh the board:  ``uv run python scripts/refresh_board.py``.

Usage:  uv run python scripts/run_flamingo_scale.py
"""
from __future__ import annotations

from src.core.config_loader import ExperimentConfig
from src.core.experiment_runner import ExperimentRunner

# token -> internal agent_organization name
ORGS = {
    "sas": "sas",
    "cmas": "centralized_mas",
    "imas": "independent_mas",
    "dmas": "decentralized_mas",
    "hmas": "hybrid_mas",
}
SCALE_NS = [3, 6, 12]
EPISODES = 8
SEED = 42
STEPS = 120


def _scenario_params(n: int) -> dict:
    """Stochastic, contended catalog that scales with the constellation."""
    return {
        "stochastic": True,
        "satellite_phase_shift": 0,           # shared windows → contention
        "visibility_period_steps": 12,
        "visibility_window_steps": 3,
        "targets": {"count": 2 * n, "priorities": [5.0, 3.0, 2.0, 1.0]},
    }


def run(token: str, org: str, n: int) -> float:
    rid = f"flamingo_{token}_ag_symb_n{n}"
    cfg = ExperimentConfig(
        experiment_id=rid,
        seed=SEED,
        agent_organization=org,
        decision_procedure="sda",
        representation="symb",
        representation_config={"type": "rule_based_flamingo"},
        behaviour="hand_designed",
        operations_paradigm="autonomous_ground",
        operations_paradigm_config={"pass_through_observation": True},
        environment={
            "scenario": "flamingo",
            "constellation_size": n,
            "timestep_seconds": 60,
            "max_steps": STEPS,
            "scenario_config": {"scenario_params": _scenario_params(n)},
        },
        num_episodes=EPISODES,
        max_steps=STEPS,
        output_dir=f"data/results/{rid}",
    )
    stats = ExperimentRunner(config=cfg).run()["experiment_statistics"]
    return float(stats.mean["utility"])


def main() -> None:
    # Shared N = 1 anchor (a lone satellite; every organisation degenerates to it).
    anchor = run("sas", "sas", 1)
    print(f"anchor U(1) = {anchor:.1f}")
    print(f"{'org':5} {'N':>3} {'U(N)':>9} {'U/N':>8} {'M-10':>7}")
    for n in SCALE_NS:
        for token, org in ORGS.items():
            u = run(token, org, n)
            m10 = (u / n) / anchor if anchor else float("nan")
            print(f"{token:5} {n:>3} {u:>9.1f} {u / n:>8.2f} {m10:>7.3f}")


if __name__ == "__main__":
    main()

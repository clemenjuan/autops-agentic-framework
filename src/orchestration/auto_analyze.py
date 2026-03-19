"""
Auto-analysis: generate figures and print summary from experiment results.

Scriptable alternative to running the analysis/telemetry notebooks manually.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CLI
import matplotlib.pyplot as plt


def run_analysis(results_path: str | Path, output_dir: str | Path | None = None) -> None:
    """Load experiment results, print summary, and save figures.

    Args:
        results_path: Path to results.json or its parent directory.
        output_dir: Directory for figure output. Defaults to ``data/figures/``.
    """
    from src.orchestration.results_loader import load_experiment
    from src.orchestration import plotting

    results_path = Path(results_path)
    if results_path.is_dir():
        results_path = results_path / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    if output_dir is None:
        # Walk up from results.json to find the project data/ directory
        candidate = results_path.resolve().parent
        while candidate != candidate.parent:
            if (candidate / "data" / "results").is_dir():
                output_dir = candidate / "data" / "figures"
                break
            candidate = candidate.parent
        else:
            output_dir = results_path.parent / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, step_df, episode_df = load_experiment(results_path)

    # Print experiment info
    cfg_env = raw["config"]["environment"]
    total_sec = cfg_env["timestep_seconds"] * cfg_env["max_steps"]
    print(f"Experiment : {raw['experiment_id']}")
    print(f"Episodes   : {raw['num_episodes']}")
    print(f"Sim length : {cfg_env['max_steps']} steps x {cfg_env['timestep_seconds']}s "
          f"= {total_sec / 3600:.1f} h")
    print()

    # Episode summary table
    summary = plotting.format_episode_summary(episode_df)
    print(summary)
    print()

    # Dashboard figure
    fig = plotting.plot_experiment_dashboard(
        step_df, episode_df,
        save_path=output_dir / "dashboard.pdf",
    )
    plt.close(fig)

    # Battery SoC time series
    ax = plotting.plot_timeseries(
        step_df, "battery_soc",
        ylabel="Battery SoC",
        title="Battery State of Charge",
        save_path=output_dir / "battery_soc.pdf",
    )
    plt.close(ax.figure)

    # Data stored time series
    ax = plotting.plot_timeseries(
        step_df, "data_stored_mb",
        ylabel="Data Stored (MB)",
        title="On-board Data Storage",
        save_path=output_dir / "data_stored.pdf",
    )
    plt.close(ax.figure)

    print(f"Figures saved to {output_dir}")

"""Publication-quality plotting for experiment results.

Uses matplotlib + seaborn with journal-ready defaults (serif fonts,
tight layouts, high DPI). All functions accept DataFrames produced
by results_loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Journal-ready style defaults
# ---------------------------------------------------------------------------

STYLE_DEFAULTS: Dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Column width ~3.5in, full width ~7.2in (IEEE / typical Q1 journal)
FIG_COL = (3.5, 2.6)
FIG_FULL = (7.2, 3.2)

# Colour palette (colourblind-safe)
PALETTE = sns.color_palette("colorblind")


def apply_style() -> None:
    """Apply journal-ready matplotlib style globally."""
    plt.rcParams.update(STYLE_DEFAULTS)


def _savefig(fig: plt.Figure, path: Optional[Union[str, Path]]) -> None:
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)


# ---------------------------------------------------------------------------
# Time-series plots (per-step)
# ---------------------------------------------------------------------------


def plot_timeseries(
    step_df: pd.DataFrame,
    y_col: str,
    *,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    episode_ids: Optional[Sequence[int]] = None,
    x_col: str = "step",
    hue_col: str = "episode_id",
    figsize: Tuple[float, float] = FIG_FULL,
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a per-step metric over time, one line per episode."""
    apply_style()
    df = step_df.copy()
    if episode_ids is not None:
        df = df[df["episode_id"].isin(episode_ids)]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for i, (ep_id, grp) in enumerate(df.groupby(hue_col)):
        ax.plot(
            grp[x_col], grp[y_col],
            color=PALETTE[i % len(PALETTE)],
            alpha=0.7,
            label=f"Ep {ep_id}",
        )

    ax.set_xlabel("Simulation step")
    ax.set_ylabel(ylabel or y_col.replace("_", " ").title())
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    if save_path:
        _savefig(ax.figure, save_path)
    return ax


def plot_timeseries_mean(
    step_df: pd.DataFrame,
    y_col: str,
    *,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    window: int = 60,
    figsize: Tuple[float, float] = FIG_COL,
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot mean +/- std across episodes with rolling smoothing."""
    apply_style()
    pivot = step_df.pivot_table(
        index="step", columns="episode_id", values=y_col,
    )
    mean = pivot.mean(axis=1).rolling(window, min_periods=1).mean()
    std = pivot.std(axis=1).rolling(window, min_periods=1).mean()

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(mean.index, mean.values, color=PALETTE[0])
    ax.fill_between(
        mean.index, (mean - std).values, (mean + std).values,
        alpha=0.2, color=PALETTE[0],
    )
    ax.set_xlabel("Simulation step")
    ax.set_ylabel(ylabel or y_col.replace("_", " ").title())
    if title:
        ax.set_title(title)
    if save_path:
        _savefig(ax.figure, save_path)
    return ax


# ---------------------------------------------------------------------------
# Episode-level comparison plots
# ---------------------------------------------------------------------------


def plot_episode_metric_boxes(
    episode_df: pd.DataFrame,
    metrics: List[str],
    *,
    group_col: str = "experiment_id",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Box plots for multiple metrics, one subplot each, grouped by experiment."""
    apply_style()
    n = len(metrics)
    if figsize is None:
        figsize = (FIG_FULL[0], 2.2 * n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sns.boxplot(
            data=episode_df, x=group_col, y=metric,
            hue=group_col, palette=PALETTE, legend=False,
            ax=ax, width=0.5, linewidth=0.8,
        )
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel("")

    axes[-1].set_xlabel(group_col.replace("_", " ").title())
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


def plot_grouped_bars(
    episode_df: pd.DataFrame,
    metric: str,
    *,
    group_col: str = "experiment_id",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = FIG_COL,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Axes:
    """Bar chart of mean metric ± std across groups."""
    apply_style()
    summary = episode_df.groupby(group_col)[metric].agg(["mean", "std"])
    _, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(summary))
    ax.bar(
        x, summary["mean"], yerr=summary["std"],
        color=PALETTE[:len(summary)], capsize=3, edgecolor="black",
        linewidth=0.5, error_kw={"linewidth": 0.8},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=30, ha="right")
    ax.set_ylabel(ylabel or metric.replace("_", " ").title())
    if title:
        ax.set_title(title)
    _savefig(ax.figure, save_path)
    return ax


# ---------------------------------------------------------------------------
# Multi-metric comparison: normalized horizontal grouped bars
# ---------------------------------------------------------------------------

# Human-readable labels for research metrics
METRIC_LABELS: Dict[str, str] = {
    "utility": "Utility",
    "data_downlink_efficiency": "Downlink Efficiency",
    "resource_efficiency": "Resource Efficiency",
    "operator_load": "Operator Load",
    "explainability_score": "Explainability",
    "mean_latency_s": "Mean Latency (ms)",
    "robustness_cv": "Robustness (CV)",
    # kept for backwards-compatibility with old results files
    "robustness_mean_recovery_steps": "Recovery Steps",
}

# Scale factors applied before display (raw value × scale → displayed value)
METRIC_DISPLAY_SCALE: Dict[str, float] = {
    "mean_latency_s": 1000.0,  # seconds → milliseconds
}

# Metrics where lower is better (inverted before normalisation so bar length
# always means "better")
LOWER_IS_BETTER: set = {"operator_load", "mean_latency_s",
                         "robustness_cv", "robustness_mean_recovery_steps"}


def format_comparison_table(
    episode_df: pd.DataFrame,
    metrics: List[str],
    *,
    group_col: str = "experiment_id",
    confidence: float = 0.95,
    latex_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Return a publication-ready comparison table with mean ± CI and p-values.

    Columns: Metric | <exp_A> mean ± CI | <exp_B> mean ± CI | ... | p-value | sig

    Args:
        episode_df: Episode-level DataFrame from ``load_multiple_experiments``.
        metrics: Research metric column names to include.
        group_col: Column used to distinguish experiments.
        confidence: Confidence level for interval computation.
        latex_path: Optional path to save a LaTeX ``.tex`` file directly.

    Returns:
        Formatted DataFrame ready for ``display()`` or ``to_latex()``.
    """
    from scipy import stats as scipy_stats

    def _fmt(v: float) -> str:
        """3 significant figures for notebook display."""
        if np.isnan(v):
            return "nan"
        if v == 0.0:
            return "0"
        if abs(v) < 0.001 or abs(v) >= 100_000:
            return f"{v:.3e}"
        return f"{v:.4g}"

    def _fmt_latex_num(v: float) -> str:
        """LaTeX math fragment for a single number (no surrounding $)."""
        if np.isnan(v):
            return r"\text{nan}"
        if v == 0.0:
            return "0"
        if abs(v) < 0.001 or abs(v) >= 100_000:
            exp = int(np.floor(np.log10(abs(v))))
            mantissa = v / 10 ** exp
            return rf"{mantissa:.3g}\times 10^{{{exp}}}"
        return f"{v:.4g}"

    groups = episode_df[group_col].unique().tolist()
    rows: list = []
    rows_latex: list = []
    for m in metrics:
        if m not in episode_df.columns:
            continue
        label = METRIC_LABELS.get(m, m.replace("_", " ").title())
        row: Dict[str, Any] = {"Metric": label}
        row_latex: Dict[str, Any] = {"Metric": label}
        scale = METRIC_DISPLAY_SCALE.get(m, 1.0)
        vals_per_group = []
        for g in groups:
            vals = episode_df.loc[episode_df[group_col] == g, m].dropna().values
            n = len(vals)
            mean = float(np.mean(vals)) * scale if n > 0 else float("nan")
            if n >= 2:
                se = float(scipy_stats.sem(vals)) * scale
                ci = se * scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
            else:
                ci = float("nan")
            # Notebook display
            row[g] = f"{_fmt(mean)} +/- {_fmt(ci)}" if not np.isnan(ci) else _fmt(mean)
            # LaTeX display: $mean \pm ci$
            if not np.isnan(ci):
                row_latex[g] = f"${_fmt_latex_num(mean)} \\pm {_fmt_latex_num(ci)}$"
            else:
                row_latex[g] = f"${_fmt_latex_num(mean)}$"
            vals_per_group.append(vals)

        # Kruskal-Wallis p-value
        all_metric_vals = np.concatenate(vals_per_group) if vals_per_group else np.array([])
        is_constant = len(all_metric_vals) > 0 and np.std(all_metric_vals) < 1e-10
        valid = [v for v in vals_per_group if len(v) >= 2]
        if is_constant:
            row["p-value"] = "-"
            row["sig"] = "="
            row_latex["p-value"] = "---"
            row_latex["sig"] = "="
        elif len(valid) >= 2:
            try:
                _, p = scipy_stats.kruskal(*valid)
                if np.isnan(p):
                    row["p-value"] = row_latex["p-value"] = "-"
                    row["sig"] = row_latex["sig"] = "-"
                else:
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                    row["p-value"] = f"{p:.4f}"
                    row_latex["p-value"] = f"${p:.4f}$"
                    row["sig"] = row_latex["sig"] = sig
            except Exception:
                row["p-value"] = row_latex["p-value"] = "-"
                row["sig"] = row_latex["sig"] = "-"
        else:
            row["p-value"] = row_latex["p-value"] = "-"
            row["sig"] = row_latex["sig"] = "-"
        rows.append(row)
        rows_latex.append(row_latex)

    df = pd.DataFrame(rows).set_index("Metric")

    if latex_path is not None:
        df_latex = pd.DataFrame(rows_latex).set_index("Metric")
        p = Path(latex_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            df_latex.to_latex(
                escape=False,
                caption="Research metric comparison (mean $\\pm$ 95\\% CI).",
                label="tab:metrics",
            ),
            encoding="utf-8",
        )

    return df


def plot_metric_comparison_bars(
    episode_df: pd.DataFrame,
    metrics: List[str],
    *,
    group_col: str = "experiment_id",
    group_labels: Optional[Dict[str, str]] = None,
    confidence: float = 0.95,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Grouped horizontal bar chart comparing architectures across metrics.

    One subplot per metric (each with its own x-axis scale).  Raw values are
    displayed as bars and annotated.  LOWER_IS_BETTER metrics are bar-inverted
    so a longer bar always means better performance; the raw value annotation
    still shows the original number.  The ↓ symbol on the y-axis label marks
    these metrics.  Figure height is auto-computed from the number of metrics.

    Args:
        episode_df: Episode-level DataFrame.
        metrics: Ordered list of metric column names to display.
        group_col: Column distinguishing experiments / architectures.
        group_labels: Optional mapping of group values → display names.
        confidence: Confidence level for CI error bars.
        figsize: Figure size in inches.  Auto-computed if *None*.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    from scipy import stats as scipy_stats

    apply_style()

    avail = [m for m in metrics if m in episode_df.columns]
    groups = episode_df[group_col].unique().tolist()
    n_metrics = len(avail)
    n_groups = len(groups)

    # Scale row height with number of groups so bars don't overlap
    row_height = max(0.85, 0.35 * n_groups + 0.15)
    if figsize is None:
        figsize = (3.5, max(2.5, row_height * n_metrics + 0.9))

    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, squeeze=False)

    total_bar = min(0.85, 0.22 * n_groups)  # spread bars across y-range
    bar_height = total_bar / n_groups

    for i, m in enumerate(avail):
        ax = axes[i, 0]
        is_lib = m in LOWER_IS_BETTER

        # Collect raw means and 95 % CI half-widths
        raw_data: list = []   # (y_pos, mu, ci_half, group_idx)
        for gi, g in enumerate(groups):
            vals = episode_df.loc[episode_df[group_col] == g, m].dropna().values
            mu = float(np.mean(vals)) if vals.size > 0 else 0.0
            if vals.size >= 2:
                se = float(scipy_stats.sem(vals))
                h = se * scipy_stats.t.ppf((1 + confidence) / 2, vals.size - 1)
            else:
                h = 0.0
            y = (gi - (n_groups - 1) / 2) * bar_height
            raw_data.append((y, mu, h, gi))

        max_right = max((mu + h for _, mu, h, _ in raw_data), default=1.0)
        if max_right <= 0:
            max_right = 1.0
        ax.set_xlim(0, max_right * 1.55)

        # Draw bars; raw values shown directly (↓ on label conveys direction)
        for y, mu, h, gi in raw_data:
            lbl = (group_labels or {}).get(groups[gi], groups[gi]) if i == 0 else None
            color = PALETTE[gi % len(PALETTE)]
            if mu < max_right * 1e-6:
                # Zero-length bar: draw a visible tick at the axis and annotate
                ax.plot(
                    0, y, marker="|",
                    color=color, markersize=10, markeredgewidth=1.8,
                    label=lbl,
                )
                ax.text(
                    max_right * 0.04, y,
                    "0",
                    va="center", ha="left", fontsize=6.5, color="#333333",
                )
            else:
                ax.barh(
                    y, mu, height=bar_height * 0.75,
                    xerr=h,
                    color=color,
                    edgecolor="black", linewidth=0.4,
                    error_kw={"linewidth": 0.7, "capsize": 2},
                    label=lbl,
                )
                ax.text(
                    mu + h + max_right * 0.04, y,
                    f"{mu:.3g}",
                    va="center", ha="left", fontsize=6.5, color="#333333",
                )

        # Y-axis: metric name; ↓ marks lower-is-better (bar inverted)
        metric_label = METRIC_LABELS.get(m, m.replace("_", " ").title())
        if is_lib:
            metric_label += "  \u2193"
        ax.set_yticks([0])
        ax.set_yticklabels([metric_label], fontsize=7.5)
        ax.tick_params(axis="y", length=0, pad=4)
        y_margin = total_bar / 2 + bar_height
        ax.set_ylim(-y_margin, y_margin)

        ax.tick_params(axis="x", labelsize=6)

    # Legend in a dedicated strip above all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center", ncol=n_groups,
            fontsize=7, frameon=True, fancybox=False,
            edgecolor="#cccccc", framealpha=0.9,
            handlelength=1.0, handletextpad=0.4,
            columnspacing=1.0, borderpad=0.4,
            bbox_to_anchor=(0.5, 1.0),
        )

    # Leave room at the top for the legend (rect top < 1)
    fig.tight_layout(h_pad=3.0, rect=[0, 0, 1, 0.93])
    _savefig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Operations timeline (hero figure)
# ---------------------------------------------------------------------------

# Canonical mode colours — consistent across all timeline figures
MODE_COLORS: Dict[str, str] = {
    "charging": "#4dac26",
    "communication": "#0571b0",
    "payload_observe": "#ca0020",
    "payload_compress": "#f4a582",
    "payload_detect": "#d01c8b",
    "payload_send": "#7b2d8b",
    "safe": "#bababa",
}
MODE_ORDER = [
    "payload_observe", "payload_detect", "payload_compress",
    "payload_send", "communication", "charging", "safe",
]


def plot_operations_timeline(
    step_dfs: Dict[str, pd.DataFrame],
    *,
    steps_per_day: int = 1440,
    window: int = 60,
    figsize: Tuple[float, float] = (7.2, 4.0),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Aggregate operations timeline: mean ± std across all episodes.

    Layout: **rows = architectures, columns = [mode distribution %, battery
    SoC mean±std, cumulative downlink mean±std]**. Mode column shows the
    average fraction of time spent in each mode as a horizontal stacked bar.
    Continuous metrics show mean line + ±1 std shaded band (rolling-smoothed).

    Args:
        step_dfs: Mapping of {label: step_df} — one entry per architecture.
            Each DataFrame must have ``step``, ``battery_soc``,
            ``data_downlinked_mb``, ``resolved_mode``, and ``episode_id``
            columns.
        steps_per_day: Number of simulation steps per day (default 1440 for
            60 s steps).
        window: Rolling window in steps for smoothing mean/std bands.
        figsize: Figure size.
        save_path: Optional path to save the figure (PDF recommended).

    Returns:
        Matplotlib Figure.
    """
    apply_style()

    n_archs = len(step_dfs)
    labels = list(step_dfs.keys())

    # 3-column layout: mode % | battery | downlink
    # Col 0 is a bar chart (different x-domain), cols 1-2 share x-axis
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = gridspec.GridSpec(n_archs, 3, figure=fig,
                           width_ratios=[0.22, 0.40, 0.38],
                           hspace=0.18, wspace=0.32)

    # Build axes — cols 1 and 2 share x per row, and share across rows
    ax_mode_col: List[plt.Axes] = []
    ax_batt_col: List[plt.Axes] = []
    ax_dl_col: List[plt.Axes] = []
    for row in range(n_archs):
        ax_mode_col.append(fig.add_subplot(gs[row, 0]))
        if row == 0:
            ax_batt_col.append(fig.add_subplot(gs[row, 1]))
            ax_dl_col.append(fig.add_subplot(gs[row, 2]))
        else:
            ax_batt_col.append(fig.add_subplot(gs[row, 1],
                                                sharex=ax_batt_col[0],
                                                sharey=ax_batt_col[0]))
            ax_dl_col.append(fig.add_subplot(gs[row, 2],
                                              sharex=ax_dl_col[0],
                                              sharey=ax_dl_col[0]))

    for row, (label, sdf) in enumerate(step_dfs.items()):
        ax_mode = ax_mode_col[row]
        ax_batt = ax_batt_col[row]
        ax_dl = ax_dl_col[row]

        # --- Col 0: Mode fraction — one horizontal bar per mode ---
        if "resolved_mode" in sdf.columns and "episode_id" in sdf.columns:
            fracs = (
                sdf.groupby("episode_id")["resolved_mode"]
                .value_counts(normalize=True)
                .unstack(fill_value=0.0)
                .reindex(columns=MODE_ORDER, fill_value=0.0)
                .mean(axis=0)
            )
            # Plot in reverse order so first mode in MODE_ORDER is at top
            modes_to_plot = [m for m in reversed(MODE_ORDER) if fracs.get(m, 0.0) > 0]
            y_positions = list(range(len(modes_to_plot)))
            for y, mode in zip(y_positions, modes_to_plot):
                frac = fracs.get(mode, 0.0)
                short = mode.replace("payload_", "").replace("_", " ").title()
                ax_mode.barh(
                    y, frac, height=0.65,
                    color=MODE_COLORS.get(mode, "#cccccc"),
                    alpha=0.88,
                )
                # % label inside bar if wide enough, otherwise just outside
                if frac >= 0.05:
                    ax_mode.text(
                        frac / 2, y, f"{frac:.0%}",
                        ha="center", va="center",
                        fontsize=5.5, color="white", fontweight="bold",
                    )
                else:
                    ax_mode.text(
                        frac + 0.01, y, f"{frac:.1%}",
                        ha="left", va="center",
                        fontsize=5.0, color="#444444",
                    )
            ax_mode.set_yticks(y_positions)
            ax_mode.set_yticklabels(
                [m.replace("payload_", "").replace("_", " ").title() for m in modes_to_plot],
                fontsize=5.5,
            )
        ax_mode.set_xlim(0, 1)
        ax_mode.set_ylabel(label, fontsize=7.5, fontweight="bold")
        ax_mode.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        ax_mode.tick_params(axis="x", labelsize=6)
        ax_mode.tick_params(axis="y", length=0, pad=2)

        # Helper: pivot → rolling mean/std → convert index to days
        def _mean_std_days(col: str):
            if col not in sdf.columns or "episode_id" not in sdf.columns:
                return None, None, None
            pivot = sdf.pivot_table(index="step", columns="episode_id", values=col)
            mean = pivot.mean(axis=1).rolling(window, min_periods=1).mean()
            std = pivot.std(axis=1).rolling(window, min_periods=1).mean().fillna(0)
            x_days = mean.index / steps_per_day
            return x_days, mean, std

        # --- Col 1: Battery SoC ---
        x_days, mean, std = _mean_std_days("battery_soc")
        if mean is not None:
            ax_batt.plot(x_days, mean.values, color=PALETTE[0], linewidth=0.9)
            ax_batt.fill_between(
                x_days,
                (mean - std).clip(lower=0).values,
                (mean + std).clip(upper=1).values,
                alpha=0.20, color=PALETTE[0],
            )
            ax_batt.axhline(0.2, color="red", linewidth=0.6, linestyle="--", alpha=0.7)
            ax_batt.set_ylim(0, 1.05)
            ax_batt.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax_batt.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.4)

        # --- Col 2: Cumulative downlink ---
        dl_col_name = next(
            (c for c in ["data_downlinked_mb", "downlinked_mb"] if c in sdf.columns),
            None,
        )
        if dl_col_name is not None:
            x_days, mean, std = _mean_std_days(dl_col_name)
            if mean is not None:
                ax_dl.plot(x_days, mean.values, color=PALETTE[1], linewidth=0.9)
                ax_dl.fill_between(
                    x_days,
                    (mean - std).clip(lower=0).values,
                    (mean + std).values,
                    alpha=0.20, color=PALETTE[1],
                )
        ax_dl.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.4)

    # Column titles (top row only)
    ax_mode_col[0].set_title("Mode Distribution", fontsize=8, pad=3)
    ax_batt_col[0].set_title("Battery SoC (mean ± std)", fontsize=8, pad=3)
    ax_dl_col[0].set_title("Cumul. Downlink MB (mean ± std)", fontsize=8, pad=3)

    # x-labels on bottom row only
    ax_mode_col[-1].set_xlabel("Fraction of time", fontsize=7)
    ax_batt_col[-1].set_xlabel("Mission day", fontsize=7)
    ax_dl_col[-1].set_xlabel("Mission day", fontsize=7)

    # Hide x-tick labels on non-bottom rows for shared axes
    for row in range(n_archs - 1):
        plt.setp(ax_batt_col[row].get_xticklabels(), visible=False)
        plt.setp(ax_dl_col[row].get_xticklabels(), visible=False)

    _savefig(fig, save_path)
    return fig


def plot_operations_timeline_episode(
    step_dfs: Dict[str, pd.DataFrame],
    *,
    episode_id: Optional[int] = None,
    steps_per_day: int = 1440,
    figsize: Tuple[float, float] = (7.2, 4.0),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Single-episode operations timeline comparing multiple architectures.

    Layout: **rows = architectures, columns = [mode strip, battery SoC, cumul.
    downlink]**.  All architectures share the same episode so anomaly onset
    markers are directly comparable.

    Args:
        step_dfs: Mapping of {label: step_df} — one entry per architecture.
        episode_id: Episode to plot. If None, selects the median-reward episode
            from the first architecture (shared across all architectures so
            anomaly timing is comparable).
        steps_per_day: Number of simulation steps per day (default 1440 for 60 s steps).
        figsize: Figure size.
        save_path: Optional path to save the figure (PDF recommended).

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    apply_style()

    n_archs = len(step_dfs)
    labels = list(step_dfs.keys())

    # Build mode index mapping (consistent across all architectures)
    mode_to_idx: Dict[str, int] = {m: i for i, m in enumerate(MODE_ORDER)}
    n_modes = len(MODE_ORDER)

    # Layout: n_archs rows × 3 columns (mode | battery | downlink)
    fig, axes = plt.subplots(
        n_archs, 3,
        figsize=figsize,
        gridspec_kw={"width_ratios": [0.35, 0.33, 0.32]},
        sharex="all",
        sharey="col",
    )
    if n_archs == 1:
        axes = axes.reshape(1, 3)

    fig.subplots_adjust(hspace=0.15, wspace=0.28)

    # Precompute colormap once
    cmap_colors = [mcolors.to_rgba(MODE_COLORS.get(m, "#cccccc"))
                   for m in MODE_ORDER]
    mode_cmap = mcolors.ListedColormap(cmap_colors)
    mode_bounds = np.arange(-0.5, n_modes + 0.5, 1)
    mode_norm = mcolors.BoundaryNorm(mode_bounds, mode_cmap.N)

    # --- Select a single representative episode, shared across all architectures ---
    # Using the same episode_id ensures identical seeds → identical anomaly
    # injection times, so differences in recovery are visually comparable.
    if episode_id is not None:
        shared_ep = episode_id
    else:
        # Pick the median-reward episode from the first architecture
        first_sdf = next(iter(step_dfs.values()))
        ep_rewards = (
            first_sdf.groupby("episode_id")["reward"].sum()
            if "reward" in first_sdf.columns
            else first_sdf.groupby("episode_id")["step"].count()
        )
        sorted_eps = ep_rewards.sort_values()
        shared_ep = sorted_eps.index[len(sorted_eps) // 2]  # median

    for row, (label, sdf) in enumerate(step_dfs.items()):
        ep = shared_ep
        edf = sdf[sdf["episode_id"] == ep].copy()
        if edf.empty:
            for c in range(3):
                axes[row, c].text(0.5, 0.5, "no data", ha="center",
                                  va="center", transform=axes[row, c].transAxes)
            continue

        x = edf["step"].values
        x_days = x / steps_per_day

        # --- Col 0: Mode colour strip ---
        ax_mode = axes[row, 0]
        if "resolved_mode" in edf.columns:
            modes = edf["resolved_mode"].values
            mode_idx = np.array([mode_to_idx.get(m, n_modes - 1) for m in modes])
            dx = x_days[1] - x_days[0] if len(x_days) > 1 else 1.0
            x_edges = np.append(x_days, x_days[-1] + dx)
            ax_mode.pcolormesh(
                x_edges, [0, 1],
                mode_idx.reshape(1, -1),
                cmap=mode_cmap, norm=mode_norm,
                shading="flat", rasterized=True,
            )
        ax_mode.set_yticks([])
        ax_mode.set_ylabel(label, fontsize=7.5, fontweight="bold")
        # White dashed day gridlines over the colour strip
        for d in range(1, int(x_days[-1]) + 1):
            ax_mode.axvline(d, color="white", linewidth=0.6,
                            linestyle="--", alpha=0.7, zorder=5)
        # Mark anomaly ONSET events (False→True transitions only)
        if "anomaly" in edf.columns:
            anom = edf["anomaly"].astype(bool).values
            onsets = np.where(anom[1:] & ~anom[:-1])[0] + 1
            if len(anom) > 0 and anom[0]:
                onsets = np.concatenate([[0], onsets])
            for idx in onsets:
                ax_mode.axvline(x_days[idx], color="red",
                                linewidth=0.8, alpha=0.7, zorder=6)

        # --- Col 1: Battery SoC ---
        ax_batt = axes[row, 1]
        if "battery_soc" in edf.columns:
            ax_batt.plot(x_days, edf["battery_soc"].values,
                         color=PALETTE[0], linewidth=0.9)
            ax_batt.axhline(0.2, color="red", linewidth=0.6,
                            linestyle="--", alpha=0.7)
            ax_batt.set_ylim(0, 1.05)
            ax_batt.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax_batt.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.4)

        # --- Col 2: Cumulative downlink ---
        ax_dl = axes[row, 2]
        dl_col = next(
            (c for c in ["data_downlinked_mb", "downlinked_mb"] if c in edf.columns),
            None,
        )
        if dl_col is not None:
            # data_downlinked_mb is already cumulative from the environment
            ax_dl.plot(x_days, edf[dl_col].values, color=PALETTE[1], linewidth=0.9)
        ax_dl.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.4)

    # Column titles (only on top row) — skip mode col, legend serves that role
    axes[0, 0].set_title(f"Episode {shared_ep}", fontsize=8, pad=3)
    axes[0, 1].set_title("Battery SoC", fontsize=8, pad=3)
    axes[0, 2].set_title("Cumul. Downlink (MB)", fontsize=8, pad=3)

    # x-label only on bottom row
    for c in range(3):
        axes[-1, c].set_xlabel("Mission day", fontsize=7)

    # Mode legend (shared)
    mode_handles = [
        Patch(facecolor=MODE_COLORS[m], alpha=0.6,
              label=m.replace("payload_", "").replace("_", " ").title())
        for m in MODE_ORDER
    ]
    # Place legend inside the top-left of the first mode panel
    ax_ref = axes[0, 0]
    ax_ref.legend(
        handles=mode_handles, loc="upper left",
        ncol=2, fontsize=5, frameon=True, fancybox=False,
        edgecolor="#cccccc", facecolor="white", framealpha=0.85,
        handlelength=1.0, handletextpad=0.3, columnspacing=0.5,
        borderpad=0.3, labelspacing=0.2,
    )
    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Pareto frontier scatter
# ---------------------------------------------------------------------------


def _pareto_staircase(
    xs: np.ndarray, ys: np.ndarray,
    x_maximize: bool = True, y_maximize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Pareto points to staircase coordinates.

    For the typical maximize-x / minimize-y case the staircase steps right
    then down (steps-post order, sorted by x ascending).
    """
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if len(xs) <= 1:
        return xs, ys
    sx: list = [xs[0]]
    sy: list = [ys[0]]
    for i in range(1, len(xs)):
        # horizontal segment, then vertical step
        sx.append(xs[i])
        sy.append(ys[i - 1])
        sx.append(xs[i])
        sy.append(ys[i])
    return np.array(sx), np.array(sy)


def plot_pareto(
    episode_df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    *,
    pareto_indices: Optional[List[int]] = None,
    group_col: str = "experiment_id",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    x_maximize: bool = True,
    y_maximize: bool = False,
    figsize: Tuple[float, float] = FIG_COL,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Axes:
    """Scatter plot with optional Pareto frontier overlay (staircase shape)."""
    apply_style()
    _, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        data=episode_df, x=x_metric, y=y_metric,
        hue=group_col, palette=PALETTE, ax=ax, s=25, edgecolor="black",
        linewidth=0.4,
    )

    if pareto_indices is not None:
        summary = episode_df.groupby(group_col)[[x_metric, y_metric]].mean()
        pts = summary.iloc[pareto_indices]
        sx, sy = _pareto_staircase(
            pts[x_metric].values, pts[y_metric].values,
            x_maximize=x_maximize, y_maximize=y_maximize,
        )
        ax.plot(
            sx, sy,
            "k--", linewidth=0.8, alpha=0.6, label="Pareto frontier",
        )

    ax.set_xlabel(xlabel or x_metric.replace("_", " ").title())
    ax.set_ylabel(ylabel or y_metric.replace("_", " ").title())
    ax.legend(
        frameon=True, edgecolor="black", fancybox=False,
        fontsize=7, markerscale=0.8,
        loc="best",
        borderpad=0.5, labelspacing=0.3,
    )
    _savefig(ax.figure, save_path)
    return ax


# ---------------------------------------------------------------------------
# Heatmap for morphological matrix interaction
# ---------------------------------------------------------------------------


def plot_metric_heatmap(
    episode_df: pd.DataFrame,
    metric: str,
    row_dim: str = "agent_organization",
    col_dim: str = "decision_procedure",
    *,
    figsize: Tuple[float, float] = FIG_COL,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Axes:
    """Heatmap of mean metric across two morphological dimensions."""
    apply_style()
    pivot = episode_df.pivot_table(
        index=row_dim, columns=col_dim, values=metric, aggfunc="mean",
    )
    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(metric.replace("_", " ").title())
    _savefig(ax.figure, save_path)
    return ax


# ---------------------------------------------------------------------------
# Multi-panel dashboard
# ---------------------------------------------------------------------------


def plot_experiment_dashboard(
    step_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    *,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """4-panel overview: battery, reward, mode distribution, episode bars."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(FIG_FULL[0], 5.0))
    fig.subplots_adjust(hspace=0.5, wspace=0.35)

    # (0,0) Battery SoC time-series
    plot_timeseries_mean(
        step_df, "battery_soc", ylabel="Battery SoC",
        title="Battery State of Charge", ax=axes[0, 0],
    )
    axes[0, 0].set_ylim(0, 1.05)

    # (0,1) Cumulative reward
    plot_timeseries_mean(
        step_df, "reward", ylabel="Reward",
        title="Step Reward", ax=axes[0, 1],
    )

    # (1,0) Mode distribution
    if "resolved_mode" in step_df.columns:
        mode_counts = step_df["resolved_mode"].value_counts()
        colors = PALETTE[:len(mode_counts)]
        axes[1, 0].barh(
            mode_counts.index, mode_counts.values,
            color=colors, edgecolor="black", linewidth=0.4,
        )
        axes[1, 0].set_xlabel("Count")
        axes[1, 0].set_title("Mode Distribution")
        axes[1, 0].tick_params(axis="y", labelsize=7)

    # (1,1) Episode total reward
    ep = episode_df.copy()
    sns.barplot(
        data=ep, x="episode_id", y="total_reward",
        hue="episode_id", palette=PALETTE[:len(ep)], legend=False,
        ax=axes[1, 1], edgecolor="black", linewidth=0.5,
    )
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Total Reward")
    axes[1, 1].set_title("Episode Rewards")

    _savefig(fig, save_path)
    return fig


def format_episode_summary(
    episode_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return a clean summary table for episode-level metrics.

    For a single episode returns a single-row table with readable column
    names. For multiple episodes returns mean ± std per metric.
    """
    default_metrics = [
        "total_reward", "mean_reward", "observation_hours",
        "downlinked_mb", "final_battery_soc", "final_data_stored_mb",
        "forced_mode_changes", "anomaly_events",
    ]
    cols = [m for m in (metrics or default_metrics) if m in episode_df.columns]

    pretty = {
        "total_reward": "Total Reward",
        "mean_reward": "Mean Reward",
        "observation_hours": "Observation (h)",
        "downlinked_mb": "Downlinked (MB)",
        "final_battery_soc": "Final Battery SoC",
        "final_data_stored_mb": "Final Data Stored (MB)",
        "forced_mode_changes": "Forced Mode Changes",
        "anomaly_events": "Anomaly Events",
    }

    if len(episode_df) == 1:
        row = episode_df[cols].iloc[0]
        summary = pd.DataFrame({
            "Metric": [pretty.get(c, c.replace("_", " ").title()) for c in cols],
            "Value": [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row],
        })
        return summary.set_index("Metric")

    # Multiple episodes: mean ± std
    means = episode_df[cols].mean()
    stds = episode_df[cols].std()
    summary = pd.DataFrame({
        "Metric": [pretty.get(c, c.replace("_", " ").title()) for c in cols],
        "Mean": [f"{m:.3f}" for m in means],
        "Std": [f"{s:.3f}" for s in stds],
        "Min": [f"{v:.3f}" for v in episode_df[cols].min()],
        "Max": [f"{v:.3f}" for v in episode_df[cols].max()],
    })
    return summary.set_index("Metric")

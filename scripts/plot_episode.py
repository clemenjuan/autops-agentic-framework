"""Generates Plots from a episode recording 

This file is used to generate all plots and figures that are needed for the thesis document.
This is done to ensure consistency in the thesis itself and have one central place to adjust 
anything.


"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Repo root is one level up: scripts -> root. On sys.path so `scripts.` and
# `src.` both resolve whether this file is run as a script or imported.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Artifacts belong in the git-ignored data/ tree, next to the recordings they
# were drawn from, same as adcs_record.
FIGURE_DIR = REPO_ROOT / "data" / "figures"

# Vector first: these go into the thesis through LaTeX, where a raster figure
# is visible at print size. PNG is for a quick look, SVG for anything that
# still has to be edited by hand.
FORMATS: Tuple[str, ...] = ("pdf", "png", "svg")
DEFAULT_FORMATS: Tuple[str, ...] = ("pdf",)

# --figure value meaning "every composed figure", as opposed to one of them.
ALL = "all"

# Height of one panel in a stacked figure [in]. Shorter than FIG_COL's, which
# is the height of a graph carrying its own x axis: stacked panels share one.
PANEL_HEIGHT = 1.6

from scripts.episode_io import Recording

# The thesis style lives in one place, and it is not here: these figures sit in
# the same document as the experiment-results ones, so a second style block
# would drift from the first. The import costs ~2.5 s, almost all of it
# seaborn, which is the price of that single definition.
from src.orchestration.plotting import FIG_COL, PALETTE, apply_style

# Reference lines -- thresholds, limits, event markers -- are context, not the
# subject: grey and thin so the trace reads first.
REFERENCE = {"color": "0.45", "linewidth": 0.7}

# Legend metrics. STYLE_DEFAULTS sets only the font size, leaving matplotlib's
# spacing defaults, which are generous enough that a legend took 16-47% of its
# panel. Tightening them costs no information and recovers about a third of
# that area; the rest comes from keeping labels short.
LEGEND = {
    "frameon": False,
    "handlelength": 1.1,
    "handletextpad": 0.4,
    "columnspacing": 1.0,
    "labelspacing": 0.25,
    "borderpad": 0.2,
}

# Fraction of the y range added above the data so a legend has somewhere to sit
# that is not on top of a trace.
HEADROOM = 0.30

def _axes(ax: Optional[plt.Axes] = None, figsize: Tuple[float, float] = FIG_COL) -> plt.Axes:
    """The Axes to draw on: the caller's, or a fresh single-panel figure.

    A panel handed an Axes must never make a figure of its own -- that is what
    lets `compose` stack it with others, and what will let an overlay put two
    recordings on one set of axes.

    `apply_style` runs only on the standalone path. rcParams are read when the
    figure is created, so styling has to precede `subplots`; on the composed
    path `compose` has already done it.
    """
    if ax is not None:
        return ax

    apply_style()
    _, ax = plt.subplots(figsize=figsize)
    return ax


def _headroom(ax: plt.Axes, fraction: float = HEADROOM) -> None:
    """Grow the y range so a legend has somewhere free to sit.

    An axis symmetric about zero grows at both ends rather than only at the
    top: the torque and dipole panels fix their limits to +-the actuator's,
    and lifting only the top would draw a symmetric quantity in a frame that
    looks asymmetric.
    """
    low, high = ax.get_ylim()
    span = high - low
    if span <= 0:
        return

    if abs(low + high) < 0.01 * span:
        ax.set_ylim(low - span * fraction / 2, high + span * fraction / 2)
    else:
        ax.set_ylim(low, high + span * fraction)


def _finish(
    ax: plt.Axes,
    rec,
    ylabel: str,
    cursor: Optional[int] = None,
    legend: bool = True,
    ncol: int = 1,
) -> plt.Axes:
    """The tail every panel shares: event markers, cursor, ylabel, legend.

    Factored out so eight panels cannot each decide slightly differently what
    a cleared target looks like, which is the inconsistency this module exists
    to prevent.

    The xlabel is deliberately not set here: on a stacked figure only the
    bottom panel carries one, and only `compose` knows which that is.

    Args:
        ax: The Axes to decorate.
        rec: The recording being drawn.
        ylabel: Axis label, units included.
        cursor: Row index to mark, or None. The one piece of view state a
            panel carries, for a viewer scrubbing a timeline or an animation.
        legend: Whether to draw a legend when there is anything to put in it.
        ncol: Legend columns. Two keeps a five-entry legend from running down
            the height of a column-width panel.

    Returns:
        The same Axes, so a panel can `return _finish(...)`.
    """
    # Only the first marker is labelled: one legend entry describes all of
    # them, and a legend with "target cleared" four times is noise.
    for n, row in enumerate(rec.clear_steps):
        ax.axvline(
            rec.t_rel[row],
            linestyle=":",
            label="target cleared" if n == 0 else None,
            # Behind the data: these mark when something happened, they are not
            # the thing being read off the axis.
            zorder=0,
            **REFERENCE,
        )

    if cursor is not None:
        ax.axvline(rec.t_rel[cursor], color="0.2", linewidth=1.0, zorder=3)

    ax.set_ylabel(ylabel)

    # `ax.legend()` warns when nothing carries a label, which is the normal
    # case for a single-series panel in an episode that cleared no targets.
    handles, _ = ax.get_legend_handles_labels()
    if legend and handles:
        # Headroom first: the legend is placed against the final limits, and
        # "best" can only find a free corner if one has been made.
        _headroom(ax)
        ax.legend(ncol=ncol, **LEGEND)

    return ax


def fig_pointing(rec, ax=None, cursor=None) -> plt.Axes:
    """Angle to the current target against time, with the mission tolerance.

    The headline graph: it is what "did the satellite point" means. Everything
    else in the mission figure explains this one.
    """
    ax = _axes(ax)

    ax.plot(
        rec.t_rel,
        rec["pointing_error_deg"],
        color=PALETTE[0],
        label="pointing error",
    )

    ax.set_ylim(bottom=0.0)

    ax.axhline(
        rec.tolerance_deg,
        linestyle="--",
        label=f"tolerance ({rec.tolerance_deg:g} deg)",
        **REFERENCE,
    )

    return _finish(ax, rec, ylabel="Pointing error [deg]", cursor=cursor)

def fig_hold_timer(rec, ax=None, cursor=None):
    ax = _axes(ax)
    
    ax.plot(
        rec.t_rel,
        rec["hold_timer"],
        color=PALETTE[0],
        label="tolerance time",
    )
    
    ax.set_ylim(bottom=0.0)
    
    ax.axhline(
        rec.hold_time,
        linestyle="--",
        label=f"hold time ({rec.hold_time:g} s)",
        **REFERENCE,
    )
    
    return _finish(ax, rec, ylabel="Track time [s]", cursor=cursor)

def fig_body_rate(rec, ax=None, cursor=None):
    ax = _axes(ax)
        
    ax.plot(
        rec.t_rel,
        np.rad2deg(rec["body_rate"]),
        color=PALETTE[0],
        label="|omega|",
    )
        
    ax.set_ylim(bottom=0.0)
        
    ax.axhline(
        np.rad2deg(rec.body_rate_thresh),
        linestyle="--",
        label=f"limit ({round(np.rad2deg(rec.body_rate_thresh),1):g} deg/s)",
        **REFERENCE,
    )
        
    return _finish(ax, rec, ylabel="Body rate [deg/s]", cursor=cursor)

def fig_wheels(rec, ax=None, cursor=None) -> plt.Axes:
    """Each wheel's speed against time, with the saturation limits.

    One line per wheel, not a norm: which wheel saturates first is the
    diagnostic. In rad/s to match the torque panel below it.
    """
    ax = _axes(ax)

    speeds = rec["wheel_speeds"]
    for wheel in range(rec.n_wheels):
        ax.plot(
            rec.t_rel,
            speeds[:, wheel],
            color=PALETTE[wheel % len(PALETTE)],
            label=f"wheel {wheel}",
        )

    # Speeds are signed; without this the spin direction is not readable.
    ax.axhline(0.0, color="0.8", linewidth=0.5, zorder=0)

    # Fix the view to the data before drawing the limits: axhline takes part in
    # autoscaling, so +-599 rad/s would flatten a trace that never leaves +-20.
    # The limits then show up only when the wheels approach them.
    ax.set_ylim(ax.get_ylim())

    limit = rec.wheel_max_speed
    for sign in (1.0, -1.0):
        ax.axhline(
            sign * limit,
            linestyle="--",
            # One entry describes the pair.
            label="saturation" if sign > 0 else None,
            **REFERENCE,
        )

    # Five entries down one column would run the height of the panel.
    return _finish(ax, rec, ylabel="Wheel speed [rad/s]", cursor=cursor, ncol=2)

def fig_wheel_saturation(rec, ax=None, cursor=None) -> plt.Axes:
    """How close the cluster is to saturation, on a fixed 0-1 axis.

    The max over wheels, not a mean: one saturated wheel already costs an
    axis. Which wheel it is, fig_wheels says.
    """
    ax = _axes(ax)

    frac = rec["wheel_speed_frac"]

    ax.plot(rec.t_rel, frac, color=PALETTE[0], label="max of 4 wheels")
    # A fraction of capacity, not a trajectory -- the fill reads as one.
    ax.fill_between(rec.t_rel, 0.0, frac, color=PALETTE[0], alpha=0.15, linewidth=0)

    # Fixed scale is this panel's whole job -- it is the absolute reference
    # fig_wheels gives up by autoscaling. The max() keeps it honest if the
    # actuator model ever lets the fraction past 1.
    ax.set_ylim(0.0, max(1.05, float(np.nanmax(frac)) * 1.05))

    ax.axhline(1.0, linestyle="--", **REFERENCE)

    return _finish(ax, rec, ylabel="Saturation [-]", cursor=cursor)

def fig_wheel_torque(rec, ax=None, cursor=None) -> plt.Axes:
    """Torque commanded to each wheel, against its limit.

    Commanded, not delivered: the archive stores action * max_action, while
    the wheels clip against their momentum envelope in the actuator model.
    The two diverge exactly at saturation.
    """
    ax = _axes(ax)

    # Milli, or matplotlib parks a 1e-3 offset in the corner of the panel.
    torque = rec.wheel_torque * 1e3
    for wheel in range(rec.n_wheels):
        ax.plot(
            rec.t_rel,
            torque[:, wheel],
            # Same colour as in fig_wheels, so a wheel's speed and its command
            # can be read against each other.
            color=PALETTE[wheel % len(PALETTE)],
            label=f"wheel {wheel}",
        )

    ax.axhline(0.0, color="0.8", linewidth=0.5, zorder=0)

    limits = rec.wheel_max_torque
    if limits is not None:
        limit = float(np.max(limits)) * 1e3

        # Unlike fig_wheels, the limits fix the axis: the action space is
        # [-1, 1] scaled by max_action, so the command is bounded by
        # construction and a policy sitting on the rails is the thing to see.
        ax.set_ylim(-1.05 * limit, 1.05 * limit)

        for sign in (1.0, -1.0):
            ax.axhline(
                sign * limit,
                linestyle="--",
                label="limit" if sign > 0 else None,
                **REFERENCE,
            )

    # "Cmd." kept, short as it is: the panel must not claim to show delivered
    # torque, which is a different quantity wherever the wheels saturate.
    return _finish(ax, rec, ylabel="Cmd. torque [mN m]", cursor=cursor, ncol=2)

def fig_mtq_dipole(rec, ax=None, cursor=None) -> plt.Axes:
    """Magnetic dipole commanded to each rod, against its limit.

    Dipole, not torque: a rod produces m x B, and the field is a local inside
    simulation.step that the archive does not carry. This panel shows magnetic
    effort, not what it bought.

    TODO: the rods are body-axis aligned in eventsat.py (axis_body = x, y, z),
    which would make "MTQ x/y/z" the better label -- but that is not recorded,
    so it cannot be asserted here. Record `mtq_axes` in build_meta alongside
    `wheel_axes`, check the alignment off it, and label by axis when it holds.
    """
    ax = _axes(ax)

    dipole = rec.mtq_dipole
    for rod in range(dipole.shape[1]):
        ax.plot(
            rec.t_rel,
            dipole[:, rod],
            # Offset from the wheels' colours: reusing them would make "rod 0"
            # and "wheel 0" the same colour in one figure.
            color=PALETTE[(rod + 4) % len(PALETTE)],
            label=f"rod {rod}",
        )

    ax.axhline(0.0, color="0.8", linewidth=0.5, zorder=0)

    limits = rec.mtq_max_dipole
    if limits is not None:
        limit = float(np.max(limits))

        # Bounded by construction, like the wheel torque: dipole is the action
        # scaled by max_action, so the rails are always the relevant range.
        ax.set_ylim(-1.05 * limit, 1.05 * limit)

        for sign in (1.0, -1.0):
            ax.axhline(
                sign * limit,
                linestyle="--",
                label="limit" if sign > 0 else None,
                **REFERENCE,
            )

    return _finish(ax, rec, ylabel="Cmd. dipole [A m^2]", cursor=cursor, ncol=2)

# Trailing words that say what kind of thing a reward term is. In a legend
# where every entry is a reward term they are noise, and they are what made
# this the widest legend in the module.
_TERM_NOISE = ("reward", "penalty", "boundary")


def _term_label(term: str) -> str:
    """A reward term's name, trimmed for a legend.

    >>> _term_label("slew_rate_boundary_reward")
    'slew rate'
    """
    words = term.split("_")
    while len(words) > 1 and words[-1] in _TERM_NOISE:
        words.pop()
    return " ".join(words)


def fig_reward_terms(rec, ax=None, cursor=None) -> plt.Axes:
    """What the policy was paid, step by step, term by term.

    Dense terms only. The event bonuses are worth 100 and 500 against a dense
    range of about +-1, so sharing an axis with them would flatten this panel
    into a line at zero. When they fired is marked by the cleared-target line;
    what they were worth is fig_cum_reward's job.

    Stacked rather than overlaid: the envelope is then the dense per-step
    reward, and the share of each term is readable off the band widths.
    """
    ax = _axes(ax)

    # Row 0 onward: the reset row's reward is NaN because nothing was earned
    # there, and stacking that as zero would draw a value that does not exist.
    t = rec.t_rel[1:]
    positive = np.zeros_like(t)
    negative = np.zeros_like(t)

    for n, (term, series) in enumerate(rec.dense_terms.items()):
        color = PALETTE[n % len(PALETTE)]
        values = series[1:]

        # Split by sign rather than assuming each term keeps one: costs
        # nothing and survives a future term that crosses zero.
        up = np.clip(values, 0.0, None)
        down = np.clip(values, None, 0.0)

        ax.fill_between(
            t, positive, positive + up,
            color=color, alpha=0.8, linewidth=0, label=_term_label(term),
        )
        ax.fill_between(
            t, negative, negative + down,
            color=color, alpha=0.8, linewidth=0,
        )

        positive = positive + up
        negative = negative + down

    ax.axhline(0.0, color="0.4", linewidth=0.6, zorder=3)

    return _finish(ax, rec, ylabel="Step reward", cursor=cursor, ncol=2)

def fig_cum_reward(rec, ax=None, cursor=None) -> plt.Axes:
    """Return to date, against the shaping terms alone.

    The gap between the two lines is what the event bonuses paid. It is the
    complement of fig_reward_terms, which shows the dense terms per step and
    leaves the events out because they would flatten it: here they are the
    point, and the cliff at a cleared target is the whole shape of the curve.

    Final values are in the legend rather than annotated on the curve -- they
    are the numbers worth comparing between policies.
    """
    ax = _axes(ax)

    total = rec.cum_reward
    # nancumsum, same as cum_reward: the reset row earned nothing rather than
    # an unknown amount, so it must contribute zero rather than poison the sum.
    shaping = np.nancumsum(sum(rec.dense_terms.values()))

    ax.plot(
        rec.t_rel, total,
        color=PALETTE[0], label=f"return ({total[-1]:.0f})",
    )
    ax.plot(
        rec.t_rel, shaping,
        color=PALETTE[1], linestyle="--",
        label=f"shaping only ({shaping[-1]:.0f})",
    )

    ax.axhline(0.0, color="0.8", linewidth=0.5, zorder=0)

    return _finish(ax, rec, ylabel="Return", cursor=cursor)

# Holds all Single graphs that are needed #
PANELS = {
# Mission performance 
"pointing":fig_pointing,
"hold_timer":fig_hold_timer,
"body_rate":fig_body_rate,

# Actuator usage 
"wheels":fig_wheels,
"wheels_saturation":fig_wheel_saturation,
"wheel_torque":fig_wheel_torque,
"mtq_dipole":fig_mtq_dipole,

# RL diagnostics
"reward_terms":fig_reward_terms,
"cum_reward":fig_cum_reward
}
# Holds all needed figures
FIGURES = {
"mission":(["pointing", "hold_timer", "body_rate"]),
# Split by hardware rather than kept as one four-panel actuator figure: at
# column width that stack ran 6.4 in tall with every legend over its data.
"reaction_wheels":(["wheels", "wheels_saturation", "wheel_torque"]),
"magnetorquers":(["mtq_dipole"]),
"rl_diagnostics":(["reward_terms", "cum_reward"])
}


def _check_specs() -> None:
    """Assert that FIGURES and PANELS agree, at import time.

    A FIGURES entry naming a panel that does not exist is a typo that would
    otherwise surface as a KeyError partway through a render -- after some of
    the figures had already been written to disk. Checking here turns it into
    an error on the first import, and it is what makes `--figure`'s choices
    trustworthy: every name the CLI offers resolves to something drawable.

    Raises:
        ValueError: If a figure names a missing panel, or if one name is both
            a figure and a panel -- `_resolve` would have to guess which was
            meant.
    """
    dangling = {
        name: [panel for panel in panels if panel not in PANELS]
        for name, panels in FIGURES.items()
    }
    dangling = {name: missing for name, missing in dangling.items() if missing}
    if dangling:
        raise ValueError(f"FIGURES names panels that do not exist: {dangling}")

    shadowed = sorted(set(FIGURES) & set(PANELS))
    if shadowed:
        raise ValueError(f"names used for both a figure and a panel: {shadowed}")


_check_specs()

def compose(
    rec,
    spec: Sequence[str],
    cursor: Optional[int] = None,
    name: Optional[str] = None,
) -> plt.Figure:
    """Draw several panels onto one figure, stacked on a shared time axis.

    Knows how many panels there are and in what order, and nothing about what
    any of them draws: a panel is looked up in PANELS and handed an Axes. That
    is what makes adding a graph one dict entry rather than an edit here.

    Returns the figure without saving or showing it, so the same call serves
    the CLI, a notebook, and a viewer re-rendering on every scrub.

    Args:
        rec: The recording to draw.
        spec: Panel names, top to bottom.
        cursor: Row index to mark on every panel, or None. Passed straight
            through -- a panel does not know whether it is being drawn once
            for a PDF or sixty times for an animation.
        name: The figure's name, used for the window title. None leaves
            matplotlib's default.

    Returns:
        The composed figure.

    Raises:
        ValueError: If `spec` is empty.
        KeyError: If a name in `spec` is not a panel.
    """
    if not spec:
        raise ValueError("a figure needs at least one panel")

    # Before subplots, not after: rcParams are read when the figure and its
    # axes are created, so styling afterwards misses figsize and font metrics.
    apply_style()

    # Width is the journal column throughout. Height grows with the stack: a
    # lone panel carries its own x axis and ticks and wants the full height,
    # while stacked panels share one and get a shorter strip each.
    height = FIG_COL[1] if len(spec) == 1 else PANEL_HEIGHT * len(spec)

    fig, axes = plt.subplots(
        len(spec),
        1,
        sharex=True,
        figsize=(FIG_COL[0], height),
        # Without this, subplots(1, 1) hands back a bare Axes rather than an
        # array and the zip below would iterate the Axes itself. It only bites
        # on single-panel figures, which is exactly what `--figure pointing`
        # asks for.
        squeeze=False,
        layout="constrained",
    )
    column = axes[:, 0]

    for panel_name, ax in zip(spec, column):
        PANELS[panel_name](rec, ax=ax, cursor=cursor)

    # Once, on the bottom panel: sharex means the others draw no tick labels,
    # so a label under each would be a caption for an axis that is not there.
    column[-1].set_xlabel("Time [s]")

    # The window title rather than a suptitle: on screen you need to know which
    # episode and which figure you are looking at, but in the thesis that is
    # the caption's job and a printed title is ink to crop. `manager` is None
    # under a headless backend, which is how the tests will draw.
    if name is not None and fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(f"{name} - {rec.label}")

    return fig

def _savefig(
    fig: plt.Figure,
    outdir: Path,
    name: str,
    formats: Sequence[str] = DEFAULT_FORMATS,
) -> List[Path]:
    """Write one figure, once per format.

    Does not close the figure: the caller decides, and `main` still needs it
    open to show it. No dpi or bbox arguments either -- `apply_style` has
    already set both, and passing them here would be a second place to change
    them.

    Args:
        fig: The figure to write.
        outdir: Directory to write into; created if it does not exist.
        name: Basename without a suffix -- the figure's name in FIGURES.
        formats: Extensions to write. matplotlib reads the format off the
            suffix.

    Returns:
        The paths written, in the order the formats were given.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written = []
    for fmt in formats:
        path = outdir / f"{name}.{fmt}"
        fig.savefig(path)
        written.append(path)

    return written


def save_all(
    rec,
    outdir: Path,
    formats: Sequence[str] = DEFAULT_FORMATS,
) -> List[Path]:
    """Render and write every composed figure.

    The programmatic counterpart of `--figure all --save`, for regenerating the
    thesis set from a notebook or a Makefile: same figures, no argv, no window.

    Composed figures only. Every panel already appears inside one of them, so
    writing the single-panel versions too would put the same axes on disk
    twice under two names.

    Args:
        rec: The recording to draw.
        outdir: Directory to write into.
        formats: Extensions to write.

    Returns:
        Every path written, figure by figure.
    """
    written = []

    for name, panels in _resolve([ALL]):
        fig = compose(rec, panels, name=name)
        written += _savefig(fig, outdir, name, formats)
        # Nothing will look at it again, and matplotlib warns once more than
        # 20 figures are open.
        plt.close(fig)

    return written

def _resolve(names: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """Turn `--figure` values into (figure name, panel names) pairs.

    Collapses the three kinds of name the CLI accepts into one shape, so the
    caller's loop does not have to branch: ALL expands to every composed
    figure, a FIGURES key to its panel list, and a PANELS key to a one-panel
    figure -- which is what makes `--figure pointing` a way to look at a single
    graph on its own.

    Duplicates are dropped, keeping the first occurrence, so `-f all pointing`
    draws the mission figure once rather than twice.

    Args:
        names: Figure names, panel names, or ALL, in any mix.

    Returns:
        (figure name, panel names) pairs, in the order first requested. The
        panel lists are copies: handing back the FIGURES entry itself would let
        a caller that mutates the result edit the module's spec.

    Raises:
        KeyError: If a name is neither a figure nor a panel. argparse rejects
            those from the command line, but `save_all` and any other
            programmatic caller arrive here unchecked.
    """
    # A dict, not a list, to double as an ordered set: insertion order is the
    # request order, and setdefault makes the first mention win.
    resolved: Dict[str, List[str]] = {}

    for name in names:
        if name == ALL:
            for figure_name, panels in FIGURES.items():
                resolved.setdefault(figure_name, list(panels))
        elif name in FIGURES:
            resolved.setdefault(name, list(FIGURES[name]))
        elif name in PANELS:
            resolved.setdefault(name, [name])
        else:
            raise KeyError(
                f"unknown figure or panel {name!r}; expected one of {_spec_names()}"
            )

    return list(resolved.items())

def _spec_names() -> List[str]:
    """Everything `--figure` accepts, in the order the help should list it.

    Composed figures first, then the individual panels: a panel name is there
    for debugging one graph in isolation, and a reader scanning `--help` wants
    the finished figures at the top.

    A name must not appear in both FIGURES and PANELS -- `_resolve` would have
    to guess which was meant.
    """
    return [ALL, *FIGURES, *PANELS]


def _default_out_dir(record: Path, mission: str) -> Path:
    """`data/figures/<mission>/<record stem>/`, absolute.

    One directory per recording, named after it, so the figures from two
    policies or two seeds do not overwrite each other and a directory listing
    says which episode it came from. Absolute, so output lands in the same
    place whatever the working directory.

    The mission leads, because the stem alone no longer identifies an episode:
    `adcs_record` puts the mission in the recording's directory rather than its
    filename, so a slew and a target_track run of the same policy and seed
    share a stem and would otherwise overwrite each other. Taken from the
    archive's own `mission_type` rather than from the path it was read from,
    so a recording moved or renamed still files itself correctly.

    Creates nothing: the writer makes the directory when it writes, so asking
    for the default path stays free of side effects.
    """
    return FIGURE_DIR / mission / record.stem


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI: which recording, which figures, and what to do with them.

    Args:
        argv: Argument list. None reads sys.argv, so the CLI behaves normally;
            passing a list lets a test drive the parser without a subprocess.

    Returns:
        The parsed arguments, with `out` resolved to a concrete directory and
        `show` resolved to a concrete bool.
    """
    parser = argparse.ArgumentParser(
        prog="plot_episode",
        description="Draw figures from one recorded ADCS episode.",
    )

    # Positional: there is no sensible default recording, while every other
    # argument has one. Single for now; overlaying several policies on one axes
    # widens this to nargs="+", which leaves today's command lines working.
    parser.add_argument(
        "record",
        type=Path,
        help="episode .npz, as written by adcs_record",
    )
    parser.add_argument(
        "-f",
        "--figure",
        nargs="+",
        choices=_spec_names(),
        default=[ALL],
        # Without a metavar argparse prints the whole choices list twice, once
        # in the usage line and once here.
        metavar="NAME",
        help="what to draw: " + ", ".join(_spec_names()) + " (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        # The default depends on the recording's name, which is not known
        # until after parsing; it is filled in below.
        help="output directory (default: data/figures/<record stem>/)",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write the figures to --out (default: %(default)s)",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        # Deliberately None rather than True: argparse cannot otherwise
        # distinguish "the user asked to show" from "nobody said", and the
        # sensible default differs between the two -- see below.
        default=None,
        help="open the figures in a window (default: on, unless --save is given)",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        choices=FORMATS,
        default=list(DEFAULT_FORMATS),
        metavar="FMT",
        help="file formats to write: " + ", ".join(FORMATS) + " (default: %(default)s)",
    )

    args = parser.parse_args(argv)

    # Fail on a missing file here rather than letting np.load raise: parser.error
    # gives the usage message and exit code 2 that a CLI should.
    if not args.record.is_file():
        parser.error(f"no such recording: {args.record}")

    # Naming an output directory is asking for output. Without this, `--out
    # somewhere/` writes nothing at all and says nothing about why. Left at
    # None otherwise: the default path needs the archive's mission, which only
    # `main` has once it has loaded the recording.
    if args.out is not None:
        args.save = True

    # Showing is the default when nothing is being written, and off when
    # something is -- a batch regenerating every thesis figure should not stop
    # on a window. `--show` alongside `--save` still gets both.
    if args.show is None:
        args.show = not args.save

    # Neither showing nor saving draws figures nobody will ever see.
    if not (args.show or args.save):
        parser.error("--no-show without --save would produce nothing")

    return args

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Draw the requested figures from one recording.

    Args:
        argv: Argument list, or None to read sys.argv.

    Returns:
        Process exit code. 0 on success, 1 if the file is not a recording,
        2 (via argparse) on a bad argument.
    """
    args = parse_args(argv)

    # Read once, here. Everything below works on the object, never the path --
    # which is what lets one recording feed several figures, and what an
    # overlay across several recordings will extend rather than fight.
    #
    # Only the anticipated failure is caught: a file that exists but was not
    # written by save_episode. parse_args has already ruled out a missing path,
    # and an unexpected exception should still surface as a traceback rather
    # than be flattened into an exit code.
    try:
        rec = Recording.load(args.record)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Resolved here rather than in parse_args: the default path is keyed by the
    # archive's mission, which is only known once it has been read.
    out_dir = args.out or _default_out_dir(args.record, rec.mission_type)

    for name, panels in _resolve(args.figure):
        fig = compose(rec, panels, name=name)

        if args.save:
            # Path first and on its own line, so it can be copied or piped,
            # same as adcs_record.
            for path in _savefig(fig, out_dir, name, args.format):
                print(path)

        # Nothing will look at it again, and matplotlib warns once more than
        # 20 figures are open -- which `--figure all` reaches quickly.
        if not args.show:
            plt.close(fig)

    # Once, after every figure is built: show() blocks until the windows are
    # closed, so calling it per figure would make them appear one at a time,
    # each waiting on the last.
    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
  

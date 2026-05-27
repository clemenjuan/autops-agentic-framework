"""Phase 1.2 hourly digest — appends a periodic progress block to a log file.

Watches ``data/results/*/decisions_ep*.jsonl`` and ``data/llm_cache/<model>/``
while a live LLM batch run is in flight, and appends a digest block to the
same tee'd logfile that captures the batch run's stdout. Independent of the
batch process: if the watcher crashes, the run continues.

Usage::

    uv run python scripts/phase12_hourly_digest.py \\
        --logfile data/phase12_pilot_20260527.log \\
        --interval 3600

Run ``--once`` for a single digest cycle (useful for testing).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


FAILOVER_PATTERNS = (
    re.compile(r"openai", re.IGNORECASE),
    re.compile(r"failover", re.IGNORECASE),
    re.compile(r"fallback", re.IGNORECASE),
)


def _read_trace_tail(path: Path, max_lines: int = 500) -> List[Dict[str, Any]]:
    """Return the last N parsed JSONL entries from a trace file."""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            # Read up to ~512 KB from the tail — enough for 500 trace lines.
            read_bytes = min(size, 512 * 1024)
            f.seek(size - read_bytes)
            chunk = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = chunk.splitlines()
    # Drop the first line if we sliced mid-line.
    if read_bytes < size and lines:
        lines = lines[1:]
    out: List[Dict[str, Any]] = []
    for line in lines[-max_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _scan_failover(log_path: Path) -> int:
    """Count failover/openai hits in experiment.log (rough heuristic)."""
    if not log_path.exists():
        return 0
    hits = 0
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if any(p.search(line) for p in FAILOVER_PATTERNS):
                    hits += 1
    except OSError:
        return 0
    return hits


def _digest_experiment(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Build a per-experiment digest dict, or None if not started."""
    trace_files = sorted(exp_dir.glob("decisions_ep*.jsonl"))
    if not trace_files:
        return None

    episodes_started = len(trace_files)
    current = trace_files[-1]
    current_steps = _count_lines(current)
    tail = _read_trace_tail(current, max_lines=500)

    if tail:
        latencies = [float(t.get("latency_s") or 0.0) for t in tail]
        latencies = [x for x in latencies if x > 0]
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        max_lat = max(latencies) if latencies else 0.0
        anomaly_steps = sum(
            1 for t in tail
            if (t.get("anomaly_forced_safe") or 0) > 0 or t.get("anomaly")
        )
        anomaly_rate = anomaly_steps / len(tail)
        last_battery = tail[-1].get("battery_soc")
        last_downlink = tail[-1].get("data_downlinked_mb")
    else:
        mean_lat = max_lat = anomaly_rate = 0.0
        last_battery = last_downlink = None

    failovers = _scan_failover(exp_dir / "experiment.log")
    mtime = current.stat().st_mtime if current.exists() else 0.0

    return {
        "exp": exp_dir.name,
        "episodes_started": episodes_started,
        "current_ep_steps": current_steps,
        "mean_latency_s_recent": round(mean_lat, 2),
        "max_latency_s_recent": round(max_lat, 2),
        "anomaly_rate_recent": round(anomaly_rate, 3),
        "last_battery_soc": last_battery,
        "last_downlink_mb": last_downlink,
        "failover_hits": failovers,
        "last_trace_mtime": mtime,
    }


def _format_block(
    digests: List[Dict[str, Any]],
    cache_files: int,
    cache_delta: int,
    started_at: float,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    elapsed_h = (time.time() - started_at) / 3600.0
    lines = [
        "",
        "=" * 72,
        f"[phase12-digest] {now}  (watcher elapsed {elapsed_h:.2f} h)",
        f"  cache_files_total={cache_files}  cache_delta_last_interval={cache_delta:+d}",
    ]
    if not digests:
        lines.append("  no experiment dirs with trace data yet")
    else:
        for d in digests:
            lines.append(
                f"  {d['exp']}: ep={d['episodes_started']} step={d['current_ep_steps']} "
                f"lat_mean={d['mean_latency_s_recent']}s lat_max={d['max_latency_s_recent']}s "
                f"anom_recent={d['anomaly_rate_recent']} "
                f"batt={d['last_battery_soc']} dl_mb={d['last_downlink_mb']} "
                f"failover_log_hits={d['failover_hits']}"
            )
    lines.append("=" * 72)
    lines.append("")
    return "\n".join(lines)


def _resolve_recent_experiments(
    results_dir: Path, since_epoch: float
) -> List[Path]:
    """Return experiment subdirs touched after ``since_epoch``."""
    if not results_dir.exists():
        return []
    out: List[Path] = []
    for child in results_dir.iterdir():
        if not child.is_dir():
            continue
        # Heuristic: any decisions_ep*.jsonl modified since the watcher started.
        traces = list(child.glob("decisions_ep*.jsonl"))
        if not traces:
            continue
        latest = max(t.stat().st_mtime for t in traces)
        if latest >= since_epoch:
            out.append(child)
    out.sort(key=lambda p: p.name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1.2 hourly digest.")
    parser.add_argument("--logfile", required=True, type=Path,
                        help="Path to append digest blocks to.")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("data/results"),
                        help="Root results dir to scan.")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("data/llm_cache/qwen3.5_122b"),
                        help="LLM cache dir to count files in.")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Seconds between digest cycles (default 3600).")
    parser.add_argument("--once", action="store_true",
                        help="Run a single digest cycle and exit.")
    args = parser.parse_args()

    started_at = time.time()
    args.logfile.parent.mkdir(parents=True, exist_ok=True)
    prev_cache_files = (
        sum(1 for _ in args.cache_dir.iterdir()) if args.cache_dir.exists() else 0
    )

    # Emit a header so it's obvious where the watcher started in the log.
    with args.logfile.open("a", encoding="utf-8") as f:
        f.write(
            f"\n[phase12-digest] watcher started "
            f"{datetime.now(timezone.utc).isoformat()} "
            f"interval={args.interval}s pid={__import__('os').getpid()}\n"
        )
        f.flush()

    while True:
        try:
            cache_files_now = (
                sum(1 for _ in args.cache_dir.iterdir())
                if args.cache_dir.exists()
                else 0
            )
            cache_delta = cache_files_now - prev_cache_files
            prev_cache_files = cache_files_now

            exp_dirs = _resolve_recent_experiments(args.results_dir, started_at - 60)
            digests = [d for d in (_digest_experiment(p) for p in exp_dirs) if d]
            block = _format_block(digests, cache_files_now, cache_delta, started_at)

            with args.logfile.open("a", encoding="utf-8") as f:
                f.write(block)
                f.flush()
        except Exception as e:  # noqa: BLE001
            with args.logfile.open("a", encoding="utf-8") as f:
                f.write(f"[phase12-digest] cycle error: {e!r}\n")
                f.flush()

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

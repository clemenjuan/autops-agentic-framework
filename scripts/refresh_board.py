"""Refresh extract.json with any results.json newer than it, then rebuild the board.
Numbers auto-refresh; validity STATUS is never auto-promoted — it flips only after
the substrate-integrity verification (deliberate)."""
import json, statistics, subprocess, sys
from pathlib import Path

EXTRACT = Path("data/figures/extract.json")
KEYS = ["utility","data_downlink_efficiency","observation_hours","downlinked_mb",
        "operator_load","explainability_score","mean_latency_s","final_battery_soc",
        "anomaly_events","safety_overrides","resource_efficiency"]

# Episodes excluded by substrate-integrity screening (run_id -> episode indices).
# Append-only, manual, evidence required — same policy as MEASURED status flips.
# Excluded episodes are nulled in per_ep (preserving index alignment for paired
# seeds) and the means are recomputed over the surviving episodes only.
EXCLUDED_EPISODES = {
    # run_id -> {episode indices excluded by substrate-integrity screening}.
    # Append-only, evidence required. (Cleared at the 2026-06-13 rename; the old
    # superseded runs cleared at the 2026-06-13 rename.)
}
ex_mtime = EXTRACT.stat().st_mtime
data = {d["id"]: d for d in json.loads(EXTRACT.read_text())}
changed = 0
for rj in Path("data/results").glob("*/results.json"):
    if rj.stat().st_mtime <= ex_mtime:
        continue
    try:
        r = json.loads(rj.read_text())
    except Exception:
        continue
    eps = r.get("episodes", [])
    mean = r.get("experiment_statistics", {}).get("mean", {})
    data[r.get("experiment_id", rj.parent.name)] = {
        "id": r.get("experiment_id", rj.parent.name), "n": len(eps),
        "mean": {k: mean.get(k) for k in KEYS},
        "per_ep": {k: [(e.get("episode_metrics", {}).get("aggregated", {}) or {}).get(k) for e in eps]
                   for k in KEYS},
        "flag": "", "desc": r.get("description", "")}
    changed += 1
for rid, bad in EXCLUDED_EPISODES.items():
    d = data.get(rid)
    if not d:
        continue
    before = json.dumps(d, sort_keys=True)
    per = d["per_ep"]
    for k, vals in per.items():
        per[k] = [None if i in bad else v for i, v in enumerate(vals)]
    n_total = max((len(v) for v in per.values()), default=0)
    d["n"] = sum(1 for i in range(n_total) if i not in bad)
    d["mean"] = {k: (statistics.mean(vs) if (vs := [v for v in per[k] if v is not None]) else None)
                 for k in per}
    d["flag"] = f"episodes {sorted(bad)} excluded by substrate screening"
    if json.dumps(d, sort_keys=True) != before:
        changed += 1
# Prune entries whose result directory no longer exists (removed / renamed runs)
# so the board never resolves stale ids.
for rid in [k for k in data if not (Path("data/results") / k / "results.json").exists()]:
    del data[rid]
    changed += 1
if changed:
    EXTRACT.write_text(json.dumps(list(data.values())))
subprocess.run([sys.executable, "scripts/extract_telemetry.py"], check=True)
subprocess.run([sys.executable, "scripts/build_results_board.py"], check=True)
print(f"refreshed {changed} experiment(s)")

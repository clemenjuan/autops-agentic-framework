"""Refresh extract.json with any results.json newer than it, then rebuild the board.
Numbers auto-refresh; validity STATUS is never auto-promoted — it flips only after
the substrate-integrity verification (deliberate)."""
import json, subprocess, sys
from pathlib import Path

EXTRACT = Path("data/figures/extract.json")
KEYS = ["utility","data_downlink_efficiency","observation_hours","downlinked_mb",
        "operator_load","explainability_score","mean_latency_s","final_battery_soc",
        "anomaly_events","safety_overrides","resource_efficiency"]
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
if changed:
    EXTRACT.write_text(json.dumps(list(data.values())))
subprocess.run([sys.executable, "scripts/extract_telemetry.py"], check=True)
subprocess.run([sys.executable, "scripts/build_results_board.py"], check=True)
print(f"refreshed {changed} experiment(s)")

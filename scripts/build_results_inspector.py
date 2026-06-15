"""Build a self-contained HTML inspector over data/results/*/results.json.

Purpose: let a human verify every reported number from the raw per-episode data
— the overview table and the per-episode utility distributions — without trusting
anyone's summary. Output: data/figures/results_inspector.html (plotly via CDN;
data embedded as JSON).

This inspects *all* run directories as-is (including legacy-named campaigns), so
it doubles as an audit of what is actually on disk. The board
(`build_results_board.py`) is the curated O-framework view; this is the raw one.

Usage:  uv run python scripts/build_results_inspector.py
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("data/results")
OUT = Path("data/figures/results_inspector.html")

# Honesty flags shown next to experiments (kept in code so they are versioned).
FLAGS = {
    "eventsat_sas_symbolic_ag": "review: ground-schedule playback at full episode length (passes vs ADCS settling)",
    "eventsat_sas_symbolic_cg": "review: same ground-schedule playback caveat as AG",
}

KEYS = ["utility", "data_downlink_efficiency", "observation_hours", "downlinked_mb",
        "operator_load", "explainability_score", "mean_latency_s", "final_battery_soc",
        "anomaly_events", "safety_overrides"]

# Pre-compaction runs embedded raw per-step state and can reach multiple GB.
# Loading those fully would OOM the inspector, so skip them with a flag (the
# compact post-compaction results.json sit well under this).
MAX_PARSE_BYTES = 300_000_000


def load() -> list[dict]:
    out = []
    for rj in sorted(RESULTS.glob("*/results.json")):
        size = rj.stat().st_size
        if size > MAX_PARSE_BYTES:
            out.append({
                "id": rj.parent.name, "n": 0,
                "mean": {k: None for k in KEYS}, "per_ep": {k: [] for k in KEYS},
                "flag": f"skipped: {size // 1_000_000} MB pre-compaction results.json (not parsed)",
                "desc": "",
            })
            continue
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        eps = r.get("episodes", [])
        per_ep = {}
        for k in KEYS:
            per_ep[k] = [
                (e.get("episode_metrics", {}).get("aggregated", {}) or {}).get(k)
                for e in eps
            ]
        mean = r.get("experiment_statistics", {}).get("mean", {})
        rid = r.get("experiment_id", rj.parent.name)
        out.append({
            "id": rid,
            "n": len(eps),
            "mean": {k: mean.get(k) for k in KEYS},
            "per_ep": per_ep,
            "flag": FLAGS.get(rid, ""),
            "desc": r.get("description", ""),
        })
    return out


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AUTOPS results inspector</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
 body {{ font-family: Arial, sans-serif; margin: 24px; color: #1a1a1a; }}
 h1 {{ color: #0065BD; }} h2 {{ color: #005293; margin-top: 36px; }}
 table {{ border-collapse: collapse; font-size: 13px; }}
 th {{ background: #005293; color: white; padding: 5px 9px; text-align: left; }}
 td {{ border-bottom: 1px solid #ddd; padding: 4px 9px; }}
 .flag {{ color: #E37222; font-size: 12px; max-width: 480px; }}
 .note {{ background: #EEF4FA; padding: 10px 14px; margin: 10px 0; font-size: 13px; }}
 .plot {{ width: 100%; max-width: 1100px; height: 440px; }}
</style></head><body>
<h1>AUTOPS — results inspector</h1>
<div class="note">Every value is computed from the raw per-episode records in
<code>data/results/*/results.json</code> — hover any point for episode index and value.
All run directories are shown as-is (including legacy-named campaigns) so the contents of disk can be
audited directly. The curated O-framework view is the board. Flags are honesty annotations.</div>
<h2>Overview (all runs on disk)</h2><div id="tbl"></div>
<h2>Per-episode utility distributions</h2><div id="strip" class="plot"></div>
<script>
const DATA = {data_json};
// ---- overview table
const cols = ["id","n","utility","data_downlink_efficiency","observation_hours","operator_load","mean_latency_s","flag"];
let html = "<table><tr>" + cols.map(c=>`<th>${{c}}</th>`).join("") + "</tr>";
for (const d of DATA) {{
  html += `<tr><td>${{d.id}}</td><td>${{d.n}}</td>` +
    ["utility","data_downlink_efficiency","observation_hours","operator_load","mean_latency_s"]
      .map(k=>`<td>${{d.mean[k]==null?"—":d.mean[k].toFixed(3)}}</td>`).join("") +
    `<td class="flag">${{d.flag}}</td></tr>`;
}}
document.getElementById("tbl").innerHTML = html + "</table>";
// ---- strip plot
const strips = DATA.filter(d=>d.per_ep.utility.some(v=>v!=null)).map(d=>({{
  y: d.per_ep.utility, x: Array(d.n).fill(d.id), type:"box", boxpoints:"all",
  jitter:0.5, pointpos:0, name:d.id, hovertext:d.per_ep.utility.map((v,i)=>`ep${{i}}: ${{v==null?"-":v.toFixed(3)}}`)
}}));
Plotly.newPlot("strip", strips, {{showlegend:false, yaxis:{{title:"utility (raw)"}}, margin:{{b:200}}, xaxis:{{tickangle:35}}}});
</script></body></html>
"""


def main() -> None:
    data = load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HTML.format(data_json=json.dumps(data)))
    print(f"wrote {OUT} with {len(data)} experiments")


if __name__ == "__main__":
    main()

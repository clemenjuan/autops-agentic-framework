"""Build a self-contained HTML inspector over data/results/*/results.json.

Purpose: let a human verify every reported number from the raw per-episode data
— distributions, paired LF-HF episodes, the B-axis walk, the sensitivity sweep —
without trusting anyone's summary. Output: data/figures/results_inspector.html
(plotly via CDN; data embedded as JSON).

Usage:  uv run python scripts/build_results_inspector.py
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("data/results")
OUT = Path("data/figures/results_inspector.html")

# Honesty flags shown next to experiments (kept in code so they are versioned).
FLAGS = {
    "eventsat_sas_sda_symb_hd_ag": "EXCLUDED — ground-schedule playback bug at full episode length (probe: passes eaten by ADCS settling before telemetry refresh)",
    "eventsat_sas_sda_symb_hd_cg": "EXCLUDED — same playback bug as AG",
    "eventsat_sas_sda_hyre_hd_ah": "HF pilot, n=3 ×1440 steps — deterministic cache replay of real qwen3.5:122b outputs (temp 0)",
    "eventsat_sas_sda_hyag_hd_ah": "HF pilot, n=2 ×720 steps — agentic, partially cached",
    "lf_hyre_4b_ah": "LF rung L1, n=3 ×1440 — qwen3.5:4b live, paired seeds with HF hyre",
    "nbr_b2_hyre_ah": "B2 neighbour, onboard LLM gate-legal (R-COMPUTE1), live 122B",
}

KEYS = ["utility", "data_downlink_efficiency", "observation_hours", "downlinked_mb",
        "operator_load", "explainability_score", "mean_latency_s", "final_battery_soc",
        "anomaly_events", "safety_overrides"]


def load() -> list[dict]:
    out = []
    for rj in sorted(RESULTS.glob("*/results.json")):
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
        out.append({
            "id": r.get("experiment_id", rj.parent.name),
            "n": len(eps),
            "mean": {k: mean.get(k) for k in KEYS},
            "per_ep": per_ep,
            "flag": FLAGS.get(r.get("experiment_id", rj.parent.name), ""),
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
<code>data/results/*/results.json</code> — hover any point for episode id and value.
Utility is the raw A1 instance (unclamped). Flags are honesty annotations; flagged-EXCLUDED
experiments are shown so the exclusion itself can be verified.</div>
<h2>Overview</h2><div id="tbl"></div>
<h2>Per-episode utility distributions</h2><div id="strip" class="plot"></div>
<h2>B-axis walk (categorical neighbours: B1 &rarr; B2 &rarr; B3)</h2><div id="bwalk" class="plot"></div>
<h2>Within-profile sensitivity sweep (B_dl &times; anomaly rate) — not neighbours</h2><div id="sweep" class="plot"></div>
<h2>LF&ndash;HF paired episodes (qwen3.5: 4b vs 122b, same seeds)</h2><div id="lfhf" class="plot"></div>
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
Plotly.newPlot("strip", strips, {{showlegend:false, yaxis:{{title:"utility (raw)"}}, margin:{{b:160}}, xaxis:{{tickangle:30}}}});
// ---- B-walk
function meanOf(id) {{ const d = DATA.find(x=>x.id===id); return d? d.mean.utility : null; }}
const bw = [
 {{x:["B1","B2","B3"], y:[meanOf("eventsat_sas_sda_symb_hd_ao"), meanOf("nbr_b2_symb_ao"), meanOf("nbr_b3_symb_ao")], name:"symbolic AO", mode:"lines+markers"}},
 {{x:["B1","B2","B3"], y:[meanOf("eventsat_sas_sda_symb_hd_ah"), meanOf("nbr_b2_symb_ah"), meanOf("nbr_b3_symb_ah")], name:"symbolic AH", mode:"lines+markers"}},
 {{x:["B1","B2"], y:[meanOf("eventsat_sas_sda_hyre_hd_ah"), meanOf("nbr_b2_hyre_ah")], name:"LLM (hyre) AH — B1 informs B2+ per R-COMPUTE1", mode:"lines+markers", line:{{dash:"dot"}}}},
];
Plotly.newPlot("bwalk", bw, {{yaxis:{{title:"mean utility"}}, xaxis:{{title:"SS-B engineering tier"}}}});
// ---- sweep
const sweepPts = DATA.filter(d=>d.id.startsWith("sweep_symb_ah_dl"));
for (const atag of ["a1","a4"]) {{
  // grouped below
}}
const series = {{}};
for (const d of sweepPts) {{
  const m = d.id.match(/dl(\\d+)_(a\\d)/); if (!m) continue;
  (series[m[2]] = series[m[2]]||[]).push({{x:+m[1], y:d.mean.utility, n:d.n}});
}}
const sweepTraces = Object.entries(series).map(([k,pts])=>({{
  x: pts.sort((a,b)=>a.x-b.x).map(p=>p.x), y: pts.map(p=>p.y),
  name: k==="a1" ? "anomaly 0.001/step" : "anomaly 0.004/step", mode:"lines+markers"
}}));
Plotly.newPlot("sweep", sweepTraces, {{xaxis:{{title:"daily downlink budget B_dl (MB)"}}, yaxis:{{title:"mean utility (30 eps)"}}}});
// ---- LF-HF pairs
const hf = DATA.find(d=>d.id==="eventsat_sas_sda_hyre_hd_ah"), lf = DATA.find(d=>d.id==="lf_hyre_4b_ah");
if (hf && lf) {{
  const n = Math.min(hf.per_ep.utility.length, lf.per_ep.utility.length);
  const xs = lf.per_ep.utility.slice(0,n), ys = hf.per_ep.utility.slice(0,n);
  const lim = [0, Math.max(...xs, ...ys)*1.15];
  Plotly.newPlot("lfhf", [
    {{x:xs, y:ys, mode:"markers", marker:{{size:12}}, name:"paired episodes",
      text:xs.map((_,i)=>`episode ${{i}}`)}},
    {{x:lim, y:lim, mode:"lines", line:{{dash:"dash", color:"#999"}}, name:"y = x"}}
  ], {{xaxis:{{title:"LF utility (qwen3.5:4b)", range:lim}}, yaxis:{{title:"HF utility (qwen3.5:122b)", range:lim}}}});
}}
</script></body></html>
"""


def main() -> None:
    data = load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HTML.format(data_json=json.dumps(data)))
    print(f"wrote {OUT} with {len(data)} experiments")


if __name__ == "__main__":
    main()

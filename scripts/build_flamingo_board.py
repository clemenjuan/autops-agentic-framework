"""AUTOPS Flamingo-lite organisation board.

Standalone board for the first multi-satellite increment. It intentionally does
not modify the EventSat results board: Flamingo opens the organisation axis and
gets its own result surface.

Usage:  uv run python scripts/build_flamingo_board.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

EXTRACT = Path("data/figures/extract.json")
OUT = Path("data/figures/flamingo_board.html")

FLAMINGO_ORGS = [
    {
        "id": "flamingo_sas_ag_symb",
        "label": "SAS",
        "organization": "sas",
        "role": "one global agent controls the constellation",
        "status_if_missing": "notrun",
    },
    {
        "id": "flamingo_cmas_ag_symb",
        "label": "CMAS",
        "organization": "centralized_mas",
        "role": "mission manager plus local satellite agents",
        "status_if_missing": "notrun",
    },
    {
        "id": "flamingo_imas_ag_symb",
        "label": "IMAS",
        "organization": "independent_mas",
        "role": "local satellite agents, no inter-agent communication",
        "status_if_missing": "deferred",
    },
    {
        "id": "flamingo_dmas_ag_symb",
        "label": "DMAS",
        "organization": "decentralized_mas",
        "role": "peer-to-peer coordination with consensus",
        "status_if_missing": "deferred",
    },
    {
        "id": "flamingo_hmas_ag_symb",
        "label": "HMAS",
        "organization": "hybrid_mas",
        "role": "clustered or heterogeneous hierarchy plus peer/local behavior",
        "status_if_missing": "deferred",
    },
]

VALUE_KEYS = [
    "utility", "coverage_rate", "successful_observations",
    "duplicate_observation_rate", "constraint_violation_rate",
    "mean_revisit_steps", "mean_latency_s", "resource_efficiency",
    "operator_load", "explainability_score",
]


def _run_text(rid: str, limit: int = 200_000) -> str:
    path = Path("data/results") / rid / "results.json"
    if not path.exists():
        return ""
    return path.open(encoding="utf-8").read(limit)


def _int_field(rid: str, field: str) -> int | None:
    text = _run_text(rid)
    if not text:
        return None
    m = re.search(rf'"{field}":\s*(\d+)', text)
    return int(m.group(1)) if m else None


def _rows(data: dict[str, dict]) -> list[dict]:
    planned_ids = {spec["id"] for spec in FLAMINGO_ORGS}
    rows: list[dict] = []
    for spec in FLAMINGO_ORGS:
        rec = data.get(spec["id"], {})
        has = bool(rec.get("n"))
        rid = rec.get("id", "")
        rows.append({
            **spec,
            "status": "measured" if has else spec["status_if_missing"],
            "n": rec.get("n", 0),
            "steps": _int_field(rid, "max_steps") if has else None,
            "constellation_size": _int_field(rid, "constellation_size") if has else 3,
            "mean": {k: rec.get("mean", {}).get(k) for k in VALUE_KEYS},
            "per_ep_utility": rec.get("per_ep", {}).get("utility", []),
            "source": "",
        })

    for rid in sorted(k for k in data if k.startswith("flamingo_") and k not in planned_ids):
        rec = data[rid]
        if not rec.get("n"):
            continue
        rows.append({
            "id": rid,
            "label": rid.replace("flamingo_", ""),
            "organization": "ad hoc",
            "role": rec.get("desc") or "measured Flamingo run outside the planned five-config sweep",
            "status_if_missing": "notrun",
            "status": "measured",
            "n": rec.get("n", 0),
            "steps": _int_field(rec.get("id", ""), "max_steps"),
            "constellation_size": _int_field(rec.get("id", ""), "constellation_size"),
            "mean": {k: rec.get("mean", {}).get(k) for k in VALUE_KEYS},
            "per_ep_utility": rec.get("per_ep", {}).get("utility", []),
            "source": "ad hoc",
        })
    return rows


def main() -> None:
    data = {d["id"]: d for d in json.loads(EXTRACT.read_text())}
    rows = _rows(data)
    payload = {
        "rows": rows,
        "planned": FLAMINGO_ORGS,
        "metric_keys": VALUE_KEYS,
    }
    OUT.write_text(TEMPLATE.replace("__PAYLOAD__", json.dumps(payload)))
    measured = sum(1 for row in rows if row["status"] == "measured")
    print(f"wrote {OUT}: {measured}/{len(FLAMINGO_ORGS)} planned Flamingo configs measured")


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AUTOPS — Flamingo-lite board</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/aaaakshat/cm-web-fonts@latest/fonts.css">
<style>
 body { background:#fff; color:#111; margin:0; font-family:'Computer Modern Serif',Georgia,'Times New Roman',serif; }
 header { padding:34px 56px 16px; border-bottom:2px solid #0065BD; }
 h1 { margin:0; font-size:26px; font-weight:600; color:#0065BD; }
 .sub { color:#444; margin-top:8px; font-size:13.5px; font-style:normal; font-family:'Computer Modern Sans',Arial,sans-serif; max-width:1040px; line-height:1.5; }
 section { padding:10px 56px 24px; max-width:1280px; }
 h2 { font-size:17.5px; color:#005293; margin:26px 0 4px; font-weight:700; }
 .caption { color:#555; font-size:13px; margin:2px 0 10px; }
 table { border-collapse:collapse; width:100%; font-size:13.5px; font-family:'Computer Modern Serif',Georgia,serif; }
 th { text-align:left; padding:6px 10px; border-top:2px solid #111; border-bottom:1px solid #111; font-weight:600; }
 td { padding:5px 10px; border-bottom:1px solid #e2e2e2; vertical-align:top; }
 tr:last-child td { border-bottom:2px solid #111; }
 .num { text-align:right; font-variant-numeric:tabular-nums; }
 .st { font-size:11px; padding:1px 7px; border:1px solid; border-radius:2px; white-space:nowrap; font-family:'Computer Modern Sans',Arial,sans-serif; }
 .st.measured { color:#1e6b34; border-color:#1e6b34; }
 .st.notrun { color:#999; border-color:#ccc; }
 .st.deferred { color:#9a6200; border-color:#9a6200; }
 .kpis { display:flex; gap:26px; margin:10px 0 4px; flex-wrap:wrap; }
 .kpi b { font-size:21px; display:block; color:#0065BD; }
 .kpi { font-size:12px; color:#555; }
 .guide { border-left:3px solid #0065BD; background:#f5f8fb; padding:10px 16px; font-size:13.5px; margin:8px 0; }
 .plot { width:100%; height:350px; }
 .twocol { display:grid; grid-template-columns:1fr 1fr; gap:30px; }
 @media (max-width:1100px){ .twocol{grid-template-columns:1fr;} }
</style></head><body>
<header>
 <h1>AUTOPS — Flamingo-lite organisation board</h1>
 <div class="sub">Standalone board for the multi-satellite SSA scheduling increment. Representation and paradigm are held fixed
 at <b>symbolic AG</b>; the comparison varies the five literature organisation configs at N = 3 before scaling to N = 6 and N = 12.
 Values appear only for measured runs. Source plan: <i>docs/flamingo_mvp.md</i>.</div>
</header>

<section>
 <h2>1&emsp;Organisation sweep</h2>
 <div class="caption">Five planned Flamingo configs. Pending rows stay visible so the decentralized/independent/hybrid MAS work is not lost.</div>
 <div class="kpis" id="kpis"></div>
 <table id="orgTable"></table>
 <div class="guide" id="note"></div>
</section>

<section>
 <h2>2&emsp;Mission metrics</h2>
 <div class="caption">Utility and coverage for measured Flamingo runs. The plot intentionally remains blank until real runs exist.</div>
 <div class="twocol">
  <div id="utilityPlot" class="plot"></div>
  <div id="wastePlot" class="plot"></div>
 </div>
</section>

<section>
 <h2>3&emsp;Scale-efficiency staging</h2>
 <div class="caption">M-10 will become measurable after an N = 1 Flamingo anchor plus N = 3, 6, and 12 paired runs.</div>
 <table id="scaleTable"></table>
</section>

<script>
const P = __PAYLOAD__;
const rows = P.rows || [];
const fmt = v => v==null ? "—" : (+v).toFixed(3);
const PERCENT = new Set(["coverage_rate","duplicate_observation_rate",
  "constraint_violation_rate","operator_load","explainability_score"]);
const UNITS = {successful_observations:"obs/ep", mean_revisit_steps:"steps",
  mean_latency_s:"s/decision", resource_efficiency:"U/sat"};
function fmtMetric(k, v){
  if (v==null) return "—";
  const x = +v;
  if (!Number.isFinite(x)) return "—";
  if (PERCENT.has(k)){
    const pct = x * 100;
    return pct < 1 && pct > 0 ? pct.toFixed(2)+"%" : pct.toFixed(1)+"%";
  }
  let body;
  if (Math.abs(x) >= 1000) body = x.toLocaleString(undefined,{maximumFractionDigits:0});
  else if (Math.abs(x) >= 100) body = x.toFixed(1);
  else if (Math.abs(x) >= 10) body = x.toFixed(2);
  else body = x.toFixed(3);
  return UNITS[k] ? body + " " + UNITS[k] : body;
}
const stc = (s,txt) => `<span class="st ${s}">${txt||s}</span>`;
const statusText = {measured:"measured", notrun:"not run", deferred:"pending impl"};
const measured = rows.filter(r=>r.status==="measured");
const planned = rows.filter(r=>!r.source);
const pending = rows.filter(r=>r.status==="deferred");
document.getElementById("kpis").innerHTML =
  `<div class="kpi"><b>${measured.length}</b>measured</div>`+
  `<div class="kpi"><b>${planned.length}</b>planned configs</div>`+
  `<div class="kpi"><b>${pending.length}</b>pending org loops</div>`+
  `<div class="kpi"><b>${new Set(rows.map(r=>r.constellation_size).filter(Boolean)).size}</b>N values present</div>`;

document.getElementById("orgTable").innerHTML =
  "<tr><th>config</th><th>organisation</th><th>status</th><th class=\"num\">N</th>"+
  "<th class=\"num\">steps</th><th class=\"num\">episodes</th><th class=\"num\">utility</th>"+
  "<th class=\"num\">coverage</th><th class=\"num\">successes</th><th class=\"num\">duplicates</th>"+
  "<th class=\"num\">violations</th><th class=\"num\">mean revisit</th><th class=\"num\">latency</th><th>role</th></tr>"+
  rows.map(r=>{
    const m = r.mean || {};
    return `<tr><td><code>${r.id}</code>${r.source?`<br><span style="color:#777;font-size:12px">${r.source}</span>`:""}</td>`+
      `<td><b>${r.label}</b><br><span style="color:#666;font-size:12px">${r.organization}</span></td>`+
      `<td>${stc(r.status, statusText[r.status] || r.status)}</td>`+
      `<td class="num">${r.constellation_size || "&mdash;"}</td>`+
      `<td class="num">${r.steps || "&mdash;"}</td>`+
      `<td class="num">${r.n || "&mdash;"}</td>`+
      `<td class="num">${fmtMetric("utility", m.utility)}</td>`+
      `<td class="num">${fmtMetric("coverage_rate", m.coverage_rate)}</td>`+
      `<td class="num">${fmtMetric("successful_observations", m.successful_observations)}</td>`+
      `<td class="num">${fmtMetric("duplicate_observation_rate", m.duplicate_observation_rate)}</td>`+
      `<td class="num">${fmtMetric("constraint_violation_rate", m.constraint_violation_rate)}</td>`+
      `<td class="num">${fmtMetric("mean_revisit_steps", m.mean_revisit_steps)}</td>`+
      `<td class="num">${fmtMetric("mean_latency_s", m.mean_latency_s)}</td>`+
      `<td style="color:#666;font-size:12px">${r.role}</td></tr>`;
  }).join("");
document.getElementById("note").innerHTML =
  `<b>Scope lock:</b> this board is separate from EventSat. The first sweep keeps representation = symbolic and paradigm = AG
  so the measured difference is organisation. DMAS is part of the baseline five-config plan, not a later optional add-on.`;

const FONT = {family:"Helvetica Neue, Helvetica, Arial, sans-serif", size:13, color:"#111"};
const baseLayout = (xTitle, yTitle, extra) => Object.assign({
  paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:FONT,
  xaxis:{title:xTitle, gridcolor:"#eee"},
  yaxis:{title:yTitle, gridcolor:"#eee"},
  margin:{t:16,b:60,l:62,r:24}
}, extra || {});
if (measured.length){
  const x = measured.map(r=>`${r.label}<br><span style="font-size:10px;color:#888">N=${r.constellation_size || "?"}, n=${r.n}</span>`);
  Plotly.newPlot("utilityPlot", [
    {type:"bar", name:"utility", x, y:measured.map(r=>r.mean.utility), marker:{color:"#4878a6"}, width:0.58},
    {type:"scatter", mode:"lines+markers", name:"coverage", x, y:measured.map(r=>r.mean.coverage_rate), yaxis:"y2",
     line:{color:"#1e8449"}, marker:{size:8}}
  ], baseLayout("organisation", "utility", {
    yaxis2:{title:"coverage", overlaying:"y", side:"right", range:[0,1]},
    legend:{orientation:"h", y:1.14}, margin:{t:16,b:60,l:62,r:62}
  }), {displayModeBar:false});
  Plotly.newPlot("wastePlot", [
    {type:"bar", name:"duplicate rate", x, y:measured.map(r=>r.mean.duplicate_observation_rate), marker:{color:"#9a6200"}, width:0.42},
    {type:"bar", name:"violation rate", x, y:measured.map(r=>r.mean.constraint_violation_rate), marker:{color:"#a13026"}, width:0.42}
  ], baseLayout("organisation", "rate", {barmode:"group", yaxis:{title:"rate", tickformat:".0%", gridcolor:"#eee"},
    legend:{orientation:"h", y:1.14}}), {displayModeBar:false});
} else {
  document.getElementById("utilityPlot").innerHTML =
    `<div class="guide"><b>No Flamingo runs measured yet.</b> Run a Flamingo experiment, then refresh this board.</div>`;
  document.getElementById("wastePlot").innerHTML =
    `<div class="guide">The duplicate/violation plot will populate from <code>data/results/flamingo_*/results.json</code>.</div>`;
}

const scaleNs = [1,3,6,12];
const byN = {};
for (const r of measured){
  const n = r.constellation_size;
  if (n && (!byN[n] || (r.mean.utility || 0) > (byN[n].mean.utility || 0))) byN[n] = r;
}
const anchor = byN[1] && byN[1].mean.utility;
document.getElementById("scaleTable").innerHTML =
  "<tr><th>N</th><th>best measured run</th><th class=\"num\">utility</th><th class=\"num\">scale efficiency</th><th>status</th></tr>"+
  scaleNs.map(n=>{
    const r = byN[n];
    const scaleEff = (r && anchor) ? (r.mean.utility / n) / anchor : null;
    return `<tr><td><b>${n}</b></td><td>${r?`<code>${r.id}</code>`:"—"}</td>`+
      `<td class="num">${r?fmtMetric("utility", r.mean.utility):"—"}</td>`+
      `<td class="num">${fmtMetric("scale_efficiency", scaleEff)}</td>`+
      `<td>${r?stc("measured","measured"):stc("notrun","not run")}</td></tr>`;
  }).join("");
</script></body></html>
"""


if __name__ == "__main__":
    main()

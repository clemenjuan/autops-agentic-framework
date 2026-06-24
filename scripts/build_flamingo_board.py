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
        "status_if_missing": "notrun",
    },
    {
        "id": "flamingo_dmas_ag_symb",
        "label": "DMAS",
        "organization": "decentralized_mas",
        "role": "peer-to-peer all-to-all consensus",
        "status_if_missing": "notrun",
    },
    {
        "id": "flamingo_hmas_ag_symb",
        "label": "HMAS",
        "organization": "hybrid_mas",
        "role": "clustered: coordinate within, independent across",
        "status_if_missing": "notrun",
    },
]

VALUE_KEYS = [
    "utility", "coverage_rate", "successful_observations",
    "duplicate_observation_rate", "constraint_violation_rate",
    "mean_revisit_steps", "mean_latency_s", "resource_efficiency",
    "operator_load", "explainability_score", "coordination_messages",
]

# token -> display label for the organisation scale sweep.
ORG_LABELS = {"sas": "SAS", "cmas": "CMAS", "imas": "IMAS", "dmas": "DMAS", "hmas": "HMAS"}
SCALE_RE = re.compile(r"^flamingo_(sas|cmas|imas|dmas|hmas)_ag_symb_n(\d+)$")


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
        # The N=1/3/6/12 scale-sweep runs belong to the scale-efficiency section,
        # not the (fixed-N) organisation sweep table.
        if SCALE_RE.match(rid):
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


def _stdev(values: list) -> float:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


def _scale_series(data: dict[str, dict]) -> dict:
    """Per-organisation M-10 = (U(N)/N)/U(1) across the scale sweep.

    Reads the ``flamingo_<org>_ag_symb_n<N>`` runs, normalises every point by the
    shared N=1 anchor, and carries per-episode spread through to error bars.
    """
    anchor_rec = data.get("flamingo_sas_ag_symb_n1", {})
    anchor = (anchor_rec.get("mean") or {}).get("utility")

    series: dict[str, list] = {}
    for rid, rec in data.items():
        m = SCALE_RE.match(rid)
        if not m or not rec.get("n"):
            continue
        org, n = m.group(1), int(m.group(2))
        utility = (rec.get("mean") or {}).get("utility")
        if utility is None:
            continue
        per_ep = (rec.get("per_ep") or {}).get("utility", []) or []
        point = {
            "N": n,
            "utility": utility,
            "utility_std": _stdev(per_ep),
            "episodes": rec.get("n", 0),
        }
        if anchor:
            point["m10"] = (utility / n) / anchor
            effs = [(v / n) / anchor for v in per_ep if v is not None]
            point["m10_std"] = _stdev(effs)
        series.setdefault(org, []).append(point)

    # Shared N=1 anchor point (M-10 = 1 by definition) so every line starts there.
    for org, points in series.items():
        if anchor and not any(p["N"] == 1 for p in points):
            points.append({"N": 1, "utility": anchor, "utility_std": _stdev(
                (anchor_rec.get("per_ep") or {}).get("utility", []) or []),
                "m10": 1.0, "m10_std": 0.0, "episodes": anchor_rec.get("n", 0)})
        points.sort(key=lambda p: p["N"])

    return {"anchor": anchor, "labels": ORG_LABELS,
            "series": {ORG_LABELS[o]: pts for o, pts in series.items()}}


def main() -> None:
    data = {d["id"]: d for d in json.loads(EXTRACT.read_text())}
    rows = _rows(data)
    payload = {
        "rows": rows,
        "planned": FLAMINGO_ORGS,
        "metric_keys": VALUE_KEYS,
        "scale": _scale_series(data),
    }
    OUT.write_text(TEMPLATE.replace("__PAYLOAD__", json.dumps(payload)))
    measured = sum(1 for row in rows if row["status"] == "measured")
    scale_pts = sum(len(v) for v in payload["scale"]["series"].values())
    print(f"wrote {OUT}: {measured}/{len(FLAMINGO_ORGS)} planned Flamingo configs "
          f"measured, {scale_pts} scale points")


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
 Values appear only for measured runs. Historical planning note: <i>archive/docs/flamingo_mvp.md</i>.</div>
</header>

<section>
 <h2>1&emsp;Organisation sweep</h2>
 <div class="caption">All five literature organisations are implemented and measured at N = 3 (paired seeds, stochastic catalog). Coordination cost (coord. msgs) is the all-to-all/intra-cluster message count per step.</div>
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
 <h2>3&emsp;Scale efficiency (M-10)</h2>
 <div class="caption">M-10 = (U(N)/N) / U(1) — does per-satellite productivity hold as the constellation grows?
 The RSO catalog scales with N (count = 2N) and seeds are paired, so this isolates coordination cost rather than
 target scarcity. The dashed line is ideal linear scaling. Coordinated organisations (SAS/CMAS/DMAS) stay flat;
 an uncoordinated one (IMAS) collapses as duplicate observations grow with N. Source runs:
 <code>flamingo_&lt;org&gt;_ag_symb_n&lt;N&gt;</code>.</div>
 <div class="twocol">
  <div id="m10Plot" class="plot"></div>
  <div id="scaleUtilPlot" class="plot"></div>
 </div>
 <table id="scaleTable"></table>
</section>

<script>
const P = __PAYLOAD__;
const rows = P.rows || [];
const fmt = v => v==null ? "—" : (+v).toFixed(3);
const PERCENT = new Set(["coverage_rate","duplicate_observation_rate",
  "constraint_violation_rate","operator_load","explainability_score"]);
const UNITS = {successful_observations:"obs/ep", mean_revisit_steps:"steps",
  mean_latency_s:"s/decision", resource_efficiency:"U/sat", coordination_messages:"msgs/step"};
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
  "<th class=\"num\">violations</th><th class=\"num\">mean revisit</th><th class=\"num\">latency</th>"+
  "<th class=\"num\">coord. msgs</th><th>role</th></tr>"+
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
      `<td class="num">${fmtMetric("coordination_messages", m.coordination_messages)}</td>`+
      `<td style="color:#666;font-size:12px">${r.role}</td></tr>`;
  }).join("");
document.getElementById("note").innerHTML =
  `<b>Reading the axis:</b> representation = symbolic and paradigm = AG are held fixed, so the only difference is organisation.
  Under contention the axis separates on <b>outcome</b> (SAS = CMAS = DMAS &gt; HMAS &gt; IMAS) and on <b>coordination cost</b>
  (DMAS &gt; HMAS &gt; SAS). SAS = CMAS = DMAS because the symbolic core reaches the same deconflicted plan from full information;
  HMAS coordinates only within clusters; IMAS not at all. Separate from the EventSat board.`;

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

const SCALE = P.scale || {series:{}, anchor:null};
const scaleOrgs = Object.keys(SCALE.series);
const ORG_COLORS = {SAS:"#0065BD", CMAS:"#1e8449", DMAS:"#7d3c98", HMAS:"#9a6200", IMAS:"#a13026"};
const scaleLine = (key, stdKey) => scaleOrgs.map(org=>{
  const pts = SCALE.series[org];
  return {type:"scatter", mode:"lines+markers", name:org,
    x:pts.map(p=>p.N), y:pts.map(p=>p[key]),
    error_y:{type:"data", array:pts.map(p=>p[stdKey]||0), visible:true, thickness:1, width:3},
    line:{color:ORG_COLORS[org]||"#444", width:2}, marker:{size:7}};
});
if (scaleOrgs.length && SCALE.anchor){
  const Ns = [1,3,6,12];
  const m10Traces = scaleLine("m10","m10_std");
  m10Traces.push({type:"scatter", mode:"lines", name:"ideal (linear)", x:Ns, y:Ns.map(()=>1),
    line:{color:"#bbb", dash:"dash", width:1.5}, hoverinfo:"skip"});
  Plotly.newPlot("m10Plot", m10Traces, baseLayout("constellation size N", "M-10 = (U(N)/N)/U(1)", {
    legend:{orientation:"h", y:1.16}, xaxis:{title:"constellation size N", tickvals:Ns, gridcolor:"#eee"}
  }), {displayModeBar:false});
  Plotly.newPlot("scaleUtilPlot", scaleLine("utility","utility_std"),
    baseLayout("constellation size N", "total utility U(N)", {
    legend:{orientation:"h", y:1.16}, xaxis:{title:"constellation size N", tickvals:Ns, gridcolor:"#eee"}
  }), {displayModeBar:false});
  const allN = [...new Set([].concat(...scaleOrgs.map(o=>SCALE.series[o].map(p=>p.N))))].sort((a,b)=>a-b);
  document.getElementById("scaleTable").innerHTML =
    "<tr><th>organisation</th>"+allN.map(n=>`<th class=\"num\">M-10 @ N=${n}</th>`).join("")+"</tr>"+
    scaleOrgs.map(org=>{
      const byNloc = Object.fromEntries(SCALE.series[org].map(p=>[p.N,p]));
      return `<tr><td><b>${org}</b></td>`+allN.map(n=>{
        const p = byNloc[n];
        return `<td class="num">${p?fmtMetric("scale_efficiency", p.m10):"—"}</td>`;
      }).join("")+"</tr>";
    }).join("");
} else {
  document.getElementById("m10Plot").innerHTML =
    `<div class="guide"><b>Scale sweep not run yet.</b> Run <code>scripts/run_flamingo_scale.py</code>, then refresh this board.</div>`;
  document.getElementById("scaleUtilPlot").innerHTML =
    `<div class="guide">M-10 needs the N = 1 anchor (<code>flamingo_sas_ag_symb_n1</code>) plus the N = 3/6/12 paired runs.</div>`;
}
</script></body></html>
"""


if __name__ == "__main__":
    main()

"""AUTOPS results board — single-file interactive explorer over the campaign results.

Reads the cached extraction (data/figures/extract.json — built once from the heavy
results.json files) and renders data/figures/results_board.html: dark single-file
explorer with human-readable names, KPI header, linked sections, and an interactive
sample-size calculator (the §5.6 power analysis, live).

Usage:  uv run python scripts/build_results_board.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

EXTRACT = Path("data/figures/extract.json")
OUT = Path("data/figures/results_board.html")

# id -> (display name, group, note)   — everything not listed goes to "legacy"
NAMES = {
    "eventsat_sas_sda_symb_hd_ao":  ("Symbolic · Onboard (AO)", "paradigm", ""),
    "eventsat_sas_sda_symb_hd_ah":  ("Symbolic · Hybrid ops (AH)", "paradigm", ""),
    "eventsat_sas_sda_symb_hd_ag":  ("Symbolic · Ground (AG)", "excluded", "playback bug — passes eaten by ADCS settling before telemetry refresh; satellite idles"),
    "eventsat_sas_sda_symb_hd_cg":  ("Conventional Ground (CG)", "excluded", "same playback bug as AG"),
    "eventsat_sas_sda_hyre_hd_ah":  ("LLM 122B single-shot · AH", "paradigm", "pilot n=3 ×1440 — deterministic cache replay of real 122B outputs"),
    "eventsat_sas_sda_hyag_hd_ah":  ("LLM 122B agentic · AH", "paradigm", "pilot n=2 ×720 — partially cached"),
    "lf_hyre_4b_ah":                ("LLM 4B single-shot · AH  (LF rung)", "fidelity", "live, paired seeds with the 122B run"),
    "nbr_b2_symb_ao":               ("Symbolic · AO @ B2", "bwalk", "B-axis neighbour (power ×5)"),
    "nbr_b2_symb_ah":               ("Symbolic · AH @ B2", "bwalk", ""),
    "nbr_b3_symb_ao":               ("Symbolic · AO @ B3", "bwalk", "B-axis neighbour (power ×20)"),
    "nbr_b3_symb_ah":               ("Symbolic · AH @ B3", "bwalk", ""),
    "nbr_b2_hyre_ah":               ("LLM 122B single-shot · AH @ B2", "bwalk", "onboard LLM at its gate-legal tier — pilot n=2 ×720, live"),
}


def main() -> None:
    data = json.loads(EXTRACT.read_text())
    by_id = {d["id"]: d for d in data}

    # paired power inputs from the 100-episode symbolic runs
    ao = [v for v in by_id["eventsat_sas_sda_symb_hd_ao"]["per_ep"]["utility"] if v is not None]
    ah = [v for v in by_id["eventsat_sas_sda_symb_hd_ah"]["per_ep"]["utility"] if v is not None]
    n = min(len(ao), len(ah))
    rho = st.correlation(ao[:n], ah[:n])
    sigma = st.stdev(ao[:n] + ah[:n])

    main_rows, legacy_rows = [], []
    for d in data:
        if d["id"].startswith("sweep_symb_ah_dl"):
            continue
        (main_rows if d["id"] in NAMES else legacy_rows).append(d)

    sweep = [d for d in data if d["id"].startswith("sweep_symb_ah_dl")]

    payload = {
        "names": NAMES, "main": main_rows, "legacy_ids": sorted(d["id"] for d in legacy_rows),
        "sweep": sweep, "rho": round(rho, 3), "sigma": round(sigma, 3),
        "total_eps": sum(d["n"] for d in data),
    }
    OUT.write_text(TEMPLATE.replace("__PAYLOAD__", json.dumps(payload)))
    print(f"wrote {OUT}  (main={len(main_rows)}, sweep={len(sweep)}, legacy={len(legacy_rows)}, rho={rho:.3f})")


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AUTOPS — M×O×T results board</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
 :root { --bg:#0b1220; --panel:#121b2e; --ink:#e8eef7; --dim:#8fa3bd; --blue:#4da3ff; --orange:#ff8c42; --red:#ff5c5c; --line:#23304a; }
 body { background:var(--bg); color:var(--ink); font-family:'Segoe UI',Arial,sans-serif; margin:0; }
 header { padding:28px 40px 18px; border-bottom:1px solid var(--line); }
 h1 { margin:0; font-size:26px; font-weight:600; } h1 span { color:var(--blue); }
 .sub { color:var(--dim); margin-top:6px; font-size:14px; }
 .kpis { display:flex; gap:14px; padding:18px 40px; flex-wrap:wrap; }
 .kpi { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 22px; min-width:150px; }
 .kpi .v { font-size:30px; font-weight:700; color:var(--blue); } .kpi .l { color:var(--dim); font-size:12px; margin-top:2px; }
 .kpi.warn .v { color:var(--orange); }
 section { padding:10px 40px 26px; } h2 { font-size:17px; font-weight:600; color:var(--ink); border-left:4px solid var(--blue); padding-left:10px; }
 .grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; } @media (max-width:1100px){ .grid{grid-template-columns:1fr;} }
 .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; }
 .note { color:var(--dim); font-size:12.5px; margin:6px 2px; }
 table { border-collapse:collapse; font-size:13px; width:100%; }
 th { text-align:left; color:var(--dim); font-weight:600; padding:6px 10px; border-bottom:1px solid var(--line); }
 td { padding:6px 10px; border-bottom:1px solid var(--line); }
 .flag { color:var(--orange); font-size:12px; } .bad { color:var(--red); }
 .ctrl { display:flex; gap:26px; align-items:center; flex-wrap:wrap; margin:8px 0 4px; }
 .ctrl label { color:var(--dim); font-size:13px; } .ctrl output { color:var(--blue); font-weight:700; }
 input[type=range] { accent-color:var(--blue); width:180px; vertical-align:middle; }
 .big { font-size:34px; font-weight:700; color:var(--orange); }
 details summary { cursor:pointer; color:var(--dim); }
</style></head><body>
<header>
 <h1>AUTOPS · <span>M × O × T</span> — first measurements</h1>
 <div class="sub">EventSat anchor (SSP-04, A1/B1/C1/D1/E0) + B-axis neighbours · raw per-episode data, hover anything · utility = raw A1 instance (unclamped)</div>
</header>
<div class="kpis" id="kpis"></div>

<section><h2>Operations paradigms at the anchor — per-episode utility</h2>
 <div class="card"><div id="paradigms" style="height:420px"></div>
 <div class="note">AG and CG are shown <span class="bad">excluded</span>: a ground-schedule playback bug (passes consumed by ADCS settling before telemetry refresh) idles the satellite — the bug is visible here precisely so the exclusion can be verified.</div></div>
</section>

<section><div class="grid">
 <div class="card"><h2>B-axis walk — categorical neighbours B1 → B2 → B3</h2><div id="bwalk" style="height:380px"></div>
  <div class="note">Symbolic is flat across tiers (platform power is not its binding constraint). The onboard LLM rises at its gate-legal tier B2 — pilot n=2, treat as a teaser, not a finding.</div></div>
 <div class="card"><h2>Within-profile sensitivity (B_dl × anomaly rate)</h2><div id="sweep" style="height:380px"></div>
  <div class="note">SSP-variable sweep inside the anchor profile (30 episodes per point) — not neighbours; demonstrates SSP-parametric response for the surrogate.</div></div>
</div></section>

<section><div class="grid">
 <div class="card"><h2>Fidelity ladder — LF (4B) vs HF (122B), paired seeds</h2><div id="lfhf" style="height:380px"></div>
  <div class="note">Same episodes, same seeds, two rungs of the LLM ladder (§4.2). n=3 — the first data point of Q3, not an estimate of ρ.</div></div>
 <div class="card"><h2>How many episodes are enough? <span style="color:var(--dim);font-size:13px">(§5.6, live)</span></h2>
  <div class="ctrl">
   <label>min. detectable effect d <input type="range" id="d" min="0.1" max="1.0" step="0.05" value="0.5"> <output id="dv">0.50</output></label>
   <label>pairing ρ <input type="range" id="rho" min="0" max="0.95" step="0.01"> <output id="rhov"></output></label>
   <label>tests in family (Bonferroni) <input type="range" id="m" min="1" max="14" step="1" value="1"> <output id="mv">1</output></label>
  </div>
  <div style="display:flex;gap:40px;margin:10px 2px;">
   <div><div class="big" id="npair"></div><div class="note">paired episodes needed<br>(shared launch-lottery seeds)</div></div>
   <div><div class="big" id="nunp" style="color:var(--dim)"></div><div class="note">per group if unpaired</div></div>
  </div>
  <div id="powcurve" style="height:230px"></div>
  <div class="note" id="pownote"></div></div>
</div></section>

<section><h2>All campaign experiments</h2><div class="card" id="tbl"></div>
 <details style="margin:10px 40px 30px;"><summary>legacy result directories (older code, not part of this campaign)</summary><div class="note" id="legacy"></div></details>
</section>

<script>
const P = __PAYLOAD__;
const C = { blue:"#4da3ff", orange:"#ff8c42", red:"#ff5c5c", dim:"#8fa3bd", grid:"#23304a" };
const LAYOUT = { paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
  font:{color:"#e8eef7", size:12}, xaxis:{gridcolor:C.grid}, yaxis:{gridcolor:C.grid},
  margin:{t:10,r:20,b:60,l:55}, legend:{orientation:"h", y:1.12} };
const name = id => (P.names[id]||[id])[0];
const note = id => (P.names[id]||["","",""])[2];
const grp  = id => (P.names[id]||["","legacy",""])[1];
const get  = id => P.main.find(d=>d.id===id);

// ---- KPIs
const kpis = [
 [P.main.length + P.sweep.length, "experiments in campaign"],
 [P.total_eps, "episodes (raw data points)"],
 ["26", "valid O-cells at the anchor"],
 ["ρ = "+P.rho, "seed-pairing correlation (measured)"],
 ["2", "cells excluded for a found bug", "warn"],
];
document.getElementById("kpis").innerHTML = kpis.map(k=>`<div class="kpi ${k[2]||""}"><div class="v">${k[0]}</div><div class="l">${k[1]}</div></div>`).join("");

// ---- paradigms
const order = ["eventsat_sas_sda_symb_hd_ao","eventsat_sas_sda_symb_hd_ah","eventsat_sas_sda_hyre_hd_ah","eventsat_sas_sda_hyag_hd_ah","lf_hyre_4b_ah","eventsat_sas_sda_symb_hd_ag","eventsat_sas_sda_symb_hd_cg"];
Plotly.newPlot("paradigms", order.filter(get).map(id=>{
  const d = get(id), ex = grp(id)==="excluded";
  return { y:d.per_ep.utility, x:Array(d.n).fill(name(id)), type:"box", boxpoints:"all", jitter:0.55, pointpos:0,
    marker:{color: ex?C.red: id.includes("hyre")||id.includes("hyag")||id.includes("lf_")?C.orange:C.blue, opacity: ex?0.45:0.85, size:5},
    line:{color: ex?C.red:C.blue}, name:name(id),
    hovertext:d.per_ep.utility.map((v,i)=>`${name(id)}<br>episode ${i}: U=${v==null?"—":v.toFixed(3)}${ex?"<br>EXCLUDED — "+note(id):""}`), hoverinfo:"text" };
}), {...LAYOUT, showlegend:false, yaxis:{...LAYOUT.yaxis, title:"utility"}});

// ---- B-walk
const bw = (ids,label,col,dash)=>({ x:["B1","B2","B3"].slice(0,ids.length),
  y:ids.map(i=>get(i)?get(i).mean.utility:null), name:label, mode:"lines+markers",
  marker:{size:10,color:col}, line:{color:col,dash:dash||"solid"} });
Plotly.newPlot("bwalk", [
  bw(["eventsat_sas_sda_symb_hd_ao","nbr_b2_symb_ao","nbr_b3_symb_ao"],"Symbolic AO",C.blue,"dot"),
  bw(["eventsat_sas_sda_symb_hd_ah","nbr_b2_symb_ah","nbr_b3_symb_ah"],"Symbolic AH",C.blue),
  bw(["eventsat_sas_sda_hyre_hd_ah","nbr_b2_hyre_ah"],"LLM 122B AH (pilot)",C.orange),
], {...LAYOUT, xaxis:{...LAYOUT.xaxis, title:"SS-B engineering tier"}, yaxis:{...LAYOUT.yaxis, title:"mean utility"}});

// ---- sweep
const series = {};
for (const d of P.sweep) { const m = d.id.match(/dl(\d+)_(a\d)/); if(!m) continue;
  (series[m[2]]=series[m[2]]||[]).push({x:+m[1], y:d.mean.utility}); }
Plotly.newPlot("sweep", Object.entries(series).map(([k,pts],i)=>({
  x:pts.sort((a,b)=>a.x-b.x).map(p=>p.x), y:pts.map(p=>p.y),
  name:k==="a1"?"anomaly 0.001 / step":"anomaly 0.004 / step", mode:"lines+markers",
  marker:{size:9}, line:{color:i?C.orange:C.blue} })),
 {...LAYOUT, xaxis:{...LAYOUT.xaxis, title:"daily downlink budget (MB)"}, yaxis:{...LAYOUT.yaxis, title:"mean utility (30 eps)"}});

// ---- LF-HF
const hf=get("eventsat_sas_sda_hyre_hd_ah"), lf=get("lf_hyre_4b_ah");
if (hf&&lf){ const k=Math.min(hf.n,lf.n), xs=lf.per_ep.utility.slice(0,k), ys=hf.per_ep.utility.slice(0,k);
 const lim=[0, Math.max(...xs,...ys)*1.2];
 Plotly.newPlot("lfhf", [
  {x:lim,y:lim,mode:"lines",line:{dash:"dash",color:C.dim},name:"y = x", hoverinfo:"skip"},
  {x:xs,y:ys,mode:"markers+text",marker:{size:14,color:C.orange},text:xs.map((_,i)=>"ep"+i),
   textposition:"top center", name:"paired episodes"}],
  {...LAYOUT, xaxis:{...LAYOUT.xaxis,title:"LF utility — qwen3.5:4b",range:lim}, yaxis:{...LAYOUT.yaxis,title:"HF utility — qwen3.5:122b",range:lim}}); }

// ---- power panel
const Z = p => { // inverse normal CDF (Acklam)
  const a=[-39.6968302866538,220.946098424521,-275.928510446969,138.357751867269,-30.6647980661472,2.50662827745924],
  b=[-54.4760987982241,161.585836858041,-155.698979859887,66.8013118877197,-13.2806815528857],
  c=[-0.00778489400243029,-0.322396458041136,-2.40075827716184,-2.54973253934373,4.37466414146497,2.93816398269878],
  d=[0.00778469570904146,0.32246712907004,2.445134137143,3.75440866190742], pl=0.02425;
  let q,r; if(p<pl){q=Math.sqrt(-2*Math.log(p));return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);}
  if(p<=1-pl){q=p-0.5;r=q*q;return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);}
  q=Math.sqrt(-2*Math.log(1-p));return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); };
const $=id=>document.getElementById(id);
$("rho").value = P.rho;
function power(){ const d=+$("d").value, rho=+$("rho").value, m=+$("m").value;
  const za=Z(1-0.05/(2*m)), zb=Z(0.8);
  const npair=Math.ceil((za+zb)**2*2*(1-rho)/d**2/0.955);
  const nunp =Math.ceil(2*(za+zb)**2/d**2/0.955);
  $("dv").textContent=d.toFixed(2); $("rhov").textContent=rho.toFixed(2); $("mv").textContent=m;
  $("npair").textContent=npair; $("nunp").textContent=nunp;
  $("pownote").textContent=`n = (z₁₋ₐ/₂ᵐ + z₀.₈)² · 2(1−ρ) / d² ÷ 0.955 (Wilcoxon ARE) · measured at the anchor: ρ = ${P.rho}, σ_U = ${P.sigma} (symbolic, 100 episodes, shared launch-lottery seeds)`;
  const ds=[...Array(19)].map((_,i)=>0.1+i*0.05);
  Plotly.react("powcurve",[
   {x:ds,y:ds.map(x=>Math.ceil((za+zb)**2*2*(1-rho)/x**2/0.955)),name:"paired",line:{color:C.orange}},
   {x:ds,y:ds.map(x=>Math.ceil(2*(za+zb)**2/x**2/0.955)),name:"unpaired /group",line:{color:C.dim,dash:"dot"}},
   {x:[d],y:[npair],mode:"markers",marker:{size:12,color:C.orange},showlegend:false}],
   {...LAYOUT, xaxis:{...LAYOUT.xaxis,title:"effect size d"}, yaxis:{...LAYOUT.yaxis,title:"episodes",type:"log"}, margin:{t:6,r:10,b:40,l:55}});
}
["d","rho","m"].forEach(id=>$(id).addEventListener("input",power)); power();

// ---- table
let rows = "<table><tr><th>experiment</th><th>n</th><th>utility</th><th>downlink eff.</th><th>obs h</th><th>override rate</th><th>note</th></tr>";
for (const d of P.main) { const f = note(d.id) || d.flag || "";
 rows += `<tr title="${d.id}"><td>${name(d.id)}</td><td>${d.n}</td>` +
  ["utility","data_downlink_efficiency","observation_hours","operator_load"].map(k=>`<td>${d.mean[k]==null?"—":d.mean[k].toFixed(3)}</td>`).join("") +
  `<td class="flag">${grp(d.id)==="excluded"?'<span class="bad">EXCLUDED</span> — ':""}${f}</td></tr>`; }
document.getElementById("tbl").innerHTML = rows + "</table><div class='note'>hover a row for the raw experiment id · sweep points are plotted above and omitted here</div>";
document.getElementById("legacy").textContent = P.legacy_ids.join("  ·  ");
</script></body></html>
"""

if __name__ == "__main__":
    main()

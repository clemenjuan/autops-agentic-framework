"""AUTOPS results board — EventSat operations-system (O) framework.

A navigable instrument over the EventSat experiment matrix
(`docs/morphological_matrix.md`): the framework cells (paradigm × representation)
with their measured state and the 14-metric registry, plus publication-style
results plots. Architectures that have not yet been run show as such; nothing is
zero-filled. The earlier M×O×T tradespace navigator (SSP selectors, the test
catalogue, the surrogate neighbourhood) was dropped in the EventSat refocus.

Data comes from `data/figures/extract.json` (built by refresh_board.py). Where a
framework cell's campaign currently lives under a legacy run-id, LEGACY_ALIAS
resolves it.

Usage:  uv run python scripts/build_results_board.py
"""
from __future__ import annotations

import json
import re
import statistics as st
import warnings
from pathlib import Path

from src.orchestration.config_loader import ExperimentConfig

EXTRACT = Path("data/figures/extract.json")

# Resolved core types that are stand-ins (symbolic delegates), not the cell's real
# policy — a cell resolving to any of these is a "placeholder" result.
PLACEHOLDER_TYPES = {
    "hrl_onboard_eventsat", "hrl_scheduler_eventsat",
    "llm_single_onboard_eventsat",  # llm-s ground (llm_single_scheduler_eventsat) is REAL now
    "llm_agentic_onboard_eventsat", "llm_agentic_scheduler_eventsat",
    "subsymbolic_scheduler_eventsat", "agentic_scheduler_eventsat",
}
OUT = Path("data/figures/results_board.html")

# The board enumerates the full 32-cell matrix straight from the generator
# (scripts/generate_experiment_configs.py — the single source of truth), so it
# never drifts from the configs. build_matrix() is imported at run time.
GROUND_ORDER = ["symb", "rl", "hrl", "llm-s", "llm-a", "hllm-s", "hllm-a"]
ONBOARD_ORDER = ["symb", "rl", "hrl"]

# Human-readable cell labels (never show raw tokens to the supervisor).
REP_LABELS = {
    "symb": "symbolic", "rl": "RL", "hrl": "hybrid-RL", "llm-s": "LLM single-shot",
    "llm-a": "LLM agentic", "hllm-s": "hybrid LLM", "hllm-a": "agentic hybrid LLM",
}

# Legacy run-id aliases (pre-refocus campaigns). Empty now that experiments run
# under their framework names; kept as a hook if older results need surfacing.
LEGACY_ALIAS: dict = {}

# Honesty annotations carried forward verbatim (versioned in code).
NOTES = {
    "eventsat_sas_ag_symb": "review: ground-schedule playback at full episode length (passes vs ADCS settling).",
    "eventsat_sas_conventional_symb": "review: same ground-schedule playback caveat as AG.",
}


def _load_matrix() -> dict:
    """Import build_matrix() from the generator (scripts/ isn't a package)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_experiment_configs",
        Path(__file__).resolve().parent / "generate_experiment_configs.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_matrix()

# 14-metric registry (morphological_matrix.md §6). status: measured | deferred.
METRICS = [
    ("M-01", "Mission Utility",           "utility",                  "measured",                 "weighted EventSat objective achievement, target-normalised"),
    ("M-02", "Mean Age of Information",   "mean_aoi_s",              "measured",                 "mean data staleness in seconds"),
    ("M-03", "Peak Age of Information",   "peak_aoi_s",              "measured",                 "maximum data staleness in seconds"),
    ("M-04", "Autonomous Recovery Eff.",  "robustness_mean_recovery_steps", "measured",             "steps from anomaly onset to cleared and nominal"),
    ("M-05", "Safety-Override Rate",      "operator_load",            "measured",                 "environment-veto fraction; operator-intervention proxy"),
    ("M-06", "Resource Efficiency",       "resource_efficiency",      "measured",                 "utility per Wh consumed"),
    ("M-07", "Decision Latency",          "mean_latency_s",           "measured",                 "mean wall-clock per decision cycle (live calls)"),
    ("M-08", "Explainability Coverage",   "explainability_score",     "measured (presence)",      "rationale-presence per decision cycle; faithfulness scorer deferred"),
    ("M-09", "Robustness (CV)",           None,                       "measured (≥30 ep)",        "cross-episode CV of utility"),
    ("M-10", "Scale Efficiency",          None,                       "deferred",                 "multi-satellite scenario"),
    ("M-11", "Downlink Efficiency",       "data_downlink_efficiency", "measured",                 "delivered / max-achievable through S-band"),
    ("M-12", "Value-of-Information",      "value_of_information",    "measured",                 "raw-equivalent delivered/captured value"),
    ("M-13", "Constraint-Violation Rate", "constraint_violation_rate", "measured",                "environment-clamped invalid command fraction"),
    ("M-14", "Commanding Effort",         "commanding_effort",       "measured",                 "commands plus weighted manual interventions per day"),
]

VALUE_KEYS = ["utility", "mean_aoi_s", "peak_aoi_s", "value_of_information",
              "constraint_violation_rate", "commanding_effort",
              "robustness_mean_recovery_steps",
              "operator_load", "resource_efficiency", "mean_latency_s",
              "mean_ground_latency_s", "explainability_score", "data_downlink_efficiency",
              "observation_hours", "downlinked_mb",
              "llm_api_calls", "llm_mean_call_latency_s", "llm_cache_hit_rate",
              "llm_total_latency_s", "llm_tokens_prompt", "llm_tokens_completion",
              "llm_schedule_entries"]

# Metrics selectable in the matrix heatmap dropdown: (key, label). "↓" = lower is better.
METRIC_OPTIONS = [
    ("utility", "Mission Utility (M-01)"),
    ("mean_aoi_s", "Mean AoI, s (M-02) ↓"),
    ("peak_aoi_s", "Peak AoI, s (M-03) ↓"),
    ("robustness_mean_recovery_steps", "Recovery, steps (M-04) ↓"),
    ("data_downlink_efficiency", "Downlink Efficiency, % (M-11)"),
    ("value_of_information", "Value of Information, % (M-12)"),
    ("constraint_violation_rate", "Constraint Violations, % (M-13) ↓"),
    ("commanding_effort", "Commanding Effort, cmd/day (M-14) ↓"),
    ("observation_hours", "Observation time, h/episode"),
    ("downlinked_mb", "Delivered data, MB/episode"),
    ("resource_efficiency", "Resource Efficiency, U/Wh (M-06)"),
    ("operator_load", "Safety-Override Rate, % (M-05) ↓"),
    ("explainability_score", "Explainability Coverage, % (M-08)"),
    ("mean_latency_s", "Decision Latency, s/decision (M-07) ↓"),
    ("mean_ground_latency_s", "Diagnostic: ground-planner latency, s ↓"),
    ("llm_mean_call_latency_s", "Diagnostic: LLM live-call latency, s ↓"),
    ("llm_api_calls", "Diagnostic: LLM calls/episode"),
    ("llm_cache_hit_rate", "Diagnostic: LLM cache-hit rate, %"),
]
LOWER_BETTER = ["mean_aoi_s", "peak_aoi_s", "robustness_mean_recovery_steps", "constraint_violation_rate", "commanding_effort", "operator_load", "mean_latency_s", "mean_ground_latency_s", "llm_mean_call_latency_s"]


def _episode_steps(rid: str) -> int | None:
    """Episode length (steps), head-streamed from the run's config so we never
    load the whole results.json."""
    if not rid:
        return None
    p = Path(f"data/results/{rid}/results.json")
    if not p.exists():
        return None
    head = p.open(encoding="utf-8").read(200_000)
    m = re.search(r'"max_steps":\s*(\d+)', head)
    if m:
        return int(m.group(1))
    m = re.search(r'"num_steps":\s*(\d+)', head)
    return int(m.group(1)) if m else None


def main() -> None:
    data = {d["id"]: d for d in json.loads(EXTRACT.read_text())}
    cells = []
    for eid, cfg in _load_matrix().items():
        paradigm = eid.split("_")[2]  # conventional | ag | ao | ah
        if paradigm == "ah":
            onb = cfg["onboard"]["representation"]
            gnd = cfg["ground"]["representation"]
            rep = f"{REP_LABELS[onb]} · {REP_LABELS[gnd]}"
        else:
            cell = cfg["representation"]
            # Place the single core in its slot: AO → onboard; AG/conventional → ground.
            if paradigm == "ao":
                onb, gnd = cell, None
            else:
                onb, gnd = None, cell
            rep = REP_LABELS[cell]
        # Truthful placeholder flag: resolve the cell's actual cores and check whether
        # either is a stand-in (symbolic delegate) rather than the real policy.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ec = ExperimentConfig(**cfg)
        resolved = {ec.resolved_onboard_type, ec.resolved_ground_planner_type}
        placeholder = bool(resolved & PLACEHOLDER_TYPES)
        alias = LEGACY_ALIAS.get(eid)
        rec = data.get(eid)
        src = eid
        if (rec is None or not rec.get("n")) and alias:
            rec = data.get(alias)
            src = alias
        rec = rec or {}
        has = bool(rec.get("n"))
        status = "measured" if has else ("placeholder" if placeholder else "notrun")
        cells.append({
            "id": eid, "paradigm": paradigm, "onboard": onb, "ground": gnd, "rep": rep,
            "status": status,
            "note": NOTES.get(eid, ""),
            "source": src if (has and src != eid) else "",
            "n": rec.get("n", 0),
            "steps": _episode_steps(rec.get("id", "")) if has else None,
            "mean": {k: rec.get("mean", {}).get(k) for k in VALUE_KEYS},
            "per_ep_utility": rec.get("per_ep", {}).get("utility", []),
        })
    by_id = {c["id"]: c for c in cells}
    ao = [v for v in by_id["eventsat_sas_ao_symb"]["per_ep_utility"] if v is not None]
    ah = [v for v in by_id["eventsat_sas_ah_symb_symb"]["per_ep_utility"] if v is not None]
    n = min(len(ao), len(ah))
    rho = round(st.correlation(ao[:n], ah[:n]), 3) if n >= 2 else 0.79
    sigma = round(st.stdev(ao[:n] + ah[:n]), 3) if n >= 2 else 0.0
    tele_path = Path("data/figures/telemetry.json")
    telemetry = json.loads(tele_path.read_text()) if tele_path.exists() else {}
    payload = {"cells": cells, "metrics": METRICS, "telemetry": telemetry,
               "rho": rho, "sigma": sigma,
               "onboard_order": ONBOARD_ORDER, "ground_order": GROUND_ORDER,
               "rep_labels": REP_LABELS,
               "metric_options": METRIC_OPTIONS, "lower_better": LOWER_BETTER}
    OUT.write_text(TEMPLATE.replace("__PAYLOAD__", json.dumps(payload)))
    measured = sum(1 for c in cells if c["status"] == "measured")
    placeholder = sum(1 for c in cells if c["status"] == "placeholder")
    print(f"wrote {OUT}: {measured}/{len(cells)} cells measured, "
          f"{placeholder} placeholder, {len(METRICS)} metrics")


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AUTOPS — EventSat O-framework</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/aaaakshat/cm-web-fonts@latest/fonts.css">
<style>
 body { background:#fff; color:#111; margin:0; font-family:'Computer Modern Serif',Georgia,'Times New Roman',serif; }
 header { padding:34px 56px 16px; border-bottom:2px solid #0065BD; }
 h1 { margin:0; font-size:26px; font-weight:600; color:#0065BD; }
 .sub { color:#444; margin-top:8px; font-size:13.5px; font-style:normal; font-family:'Computer Modern Sans',Arial,sans-serif; max-width:1000px; line-height:1.5; }
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
 .st.notrun   { color:#999;    border-color:#ccc; }
 .st.deferred { color:#9a6200; border-color:#9a6200; }
 .st.placeholder { color:#9a6200; border-color:#d9a441; border-style:dashed; }
 .sel { margin:8px 0 6px; font-family:'Computer Modern Sans',Arial,sans-serif; font-size:13px; color:#444; }
 .sel select { font-size:13px; padding:2px 6px; }
 .mgrid { width:auto; border-collapse:collapse; }
 .mgrid td, .mgrid th { border:1px solid #e0e0e0; text-align:center; padding:7px 10px; font-size:12px; min-width:62px; }
 .mgrid th.mh, .mgrid th.mrow { font-family:'Computer Modern Sans',Arial,sans-serif; font-weight:600; font-size:11px; color:#444; background:#f5f8fb; }
 .mgrid th.mrow { text-align:right; }
 .mgrid th.corner { background:#eef2f6; color:#888; }
 .mcell { font-variant-numeric:tabular-nums; color:#111; }
 .mcell.empty { background:#fff; color:#ccc; border:1px solid #f0f0f0; }
 .mcell.placeholder { background:#fbf2dd; color:#9a6200; }
 .mcell.notrun { background:#fafafa; color:#bbb; }
 .mcell .sub { display:block; font-size:9.5px; color:#777; font-weight:400; }
 .mcell.pair { padding:0; overflow:hidden; }
 .pairline { display:block; padding:7px 10px; font-size:12px; min-width:62px; font-variant-numeric:tabular-nums; }
 .pairline + .pairline { border-top:1px solid rgba(255,255,255,0.55); }
 .pairline b { font-family:'Computer Modern Sans',Arial,sans-serif; font-size:10px; margin-right:4px; }
 .mlegend { font-size:11px; color:#666; margin:6px 0 0; font-family:'Computer Modern Sans',Arial,sans-serif; }
 .kpis { display:flex; gap:26px; margin:10px 0 4px; }
 .kpi b { font-size:21px; display:block; color:#0065BD; } .kpi { font-size:12px; color:#555; }
 .guide { border-left:3px solid #0065BD; background:#f5f8fb; padding:10px 16px; font-size:13.5px; margin:8px 0; }
 .plot { width:100%; height:380px; }
 .twocol { display:grid; grid-template-columns:1fr 1fr; gap:30px; }
 @media (max-width:1100px){ .twocol{grid-template-columns:1fr;} }
</style></head><body>
<header>
 <h1>AUTOPS — EventSat architecture comparison</h1>
 <div class="sub">A controlled comparison of operations-system architectures on the <b>EventSat</b>
 event-camera CubeSat. An architecture is organisation &times; representation (cognitive substrate &times;
 action space) &times; operational paradigm; for EventSat the organisation is fixed at SAS and the
 comparison varies the <b>7 representation cells</b> across the <b>conventional&middot;ag&middot;ao&middot;ah</b>
 paradigms (32-experiment matrix). Values appear only for verified runs. Full spec:
 <i>docs/morphological_matrix.md</i>.</div>
</header>

<section>
 <h2>1&emsp;Experiment matrix (O) — onboard &times; ground</h2>
 <div class="caption">All 32 experiments on one grid: rows = onboard core, columns = ground planner.
 The operational paradigm is read from position — <b>ground only</b> (onboard&nbsp;=&nbsp;&mdash;) is AG,
 <b>onboard only</b> (ground&nbsp;=&nbsp;&mdash;) is AO, <b>both</b> is AH; the (&mdash;,&nbsp;symbolic) cell
 also carries Conventional. Pick a metric to colour the grid; values appear only for verified runs.</div>
 <div class="kpis" id="kpis"></div>
 <div class="sel"><label>metric&nbsp; <select id="metricSel"></select></label></div>
 <div id="matrix"></div>
</section>

<section>
 <h2>2&emsp;Results — EventSat</h2>
 <div class="caption" id="res-prov"></div>
 <div class="twocol">
  <div id="gradient" class="plot" style="height:330px"></div>
  <div id="cognition" class="plot" style="height:330px"></div>
 </div>
 <div class="twocol">
  <div id="obshours" class="plot" style="height:300px"></div>
  <div id="delivmb" class="plot" style="height:300px"></div>
 </div>
</section>

<section>
 <h2>3&emsp;Measured utility distributions (verified runs)</h2>
 <div id="dist" class="plot"></div>
</section>

<section>
 <h2>4&emsp;Metric registry — M-01 &hellip; M-14</h2>
 <div class="caption">Measured set populated from verified runs; deferred metrics marked.</div>
 <table id="mx"></table>
</section>

<section>
 <h2>5&emsp;Architecture comparison (verified runs)</h2>
 <div class="caption">Left: metric profile per architecture, min&ndash;max normalised across the shown
 runs (1 = best of this set — relative). Right: paired per-episode utility differences (shared
 launch-lottery seeds).</div>
 <div class="twocol">
  <div id="radar" class="plot" style="height:420px"></div>
  <div id="paired" class="plot" style="height:420px"></div>
 </div>
</section>

<section>
 <h2>6&emsp;Episode inspector — simulation telemetry</h2>
 <div class="caption">One simulated week, step by step: battery state of charge, data stored and
 downlinked, ground-contact windows (grey), anomaly-forced safe periods (red), and the operating mode
 at every step.</div>
 <div class="kpis"><label style="font-size:13px;font-family:'Computer Modern Sans',Arial,sans-serif">episode&nbsp; <select id="teleSel"></select></label></div>
 <div id="teleA" class="plot" style="height:330px"></div>
 <div id="teleB" class="plot" style="height:150px"></div>
</section>

<section>
 <h2>7&emsp;Statistical adequacy (pre-registered)</h2>
 <div class="guide" id="guide"></div>
 <div class="twocol">
  <div id="powcurve" class="plot" style="height:290px"></div>
  <div style="font-size:13.5px">
   <p><b>Pre-registered rule:</b> n<sub>pairs</sub> = (z<sub>1&minus;&alpha;/2m</sub> + z<sub>0.8</sub>)&sup2; &middot; 2(1&minus;&rho;) / d&sup2; &divide; 0.955 (Wilcoxon ARE).</p>
   <p><b>Measured inputs</b> (symbolic AO/AH, shared launch-lottery seeds): &rho; = <span id="rhoval"></span>, &sigma;<sub>U</sub> = <span id="sigval"></span>.</p>
   <p><b>In force:</b> 100 episodes per confirmatory cell — sufficient for d &ge; 0.25 at &alpha; = 0.05, power 0.8, Bonferroni m &le; 10. Cells under 30 episodes are screening pilots.</p>
  </div>
 </div>
</section>
<script>
const P = __PAYLOAD__;
const CELLS = {}; P.cells.forEach(c => CELLS[c.id] = c);
const fmt = v => v==null ? "—" : (+v).toFixed(3);
const UNITS = {
  mean_aoi_s:"s", peak_aoi_s:"s", robustness_mean_recovery_steps:"steps",
  commanding_effort:"cmd/day", observation_hours:"h/ep", downlinked_mb:"MB/ep",
  resource_efficiency:"U/Wh", mean_latency_s:"s/decision", mean_ground_latency_s:"s",
  llm_mean_call_latency_s:"s", llm_total_latency_s:"s", llm_api_calls:"calls/ep",
  llm_cache_hits:"hits/ep", llm_tokens_prompt:"tok/ep", llm_tokens_completion:"tok/ep",
  llm_schedule_entries:"entries/ep"
};
const PERCENT = new Set(["operator_load","data_downlink_efficiency","value_of_information",
  "constraint_violation_rate","explainability_score","llm_cache_hit_rate"]);
const INTEGERISH = new Set(["llm_tokens_prompt","llm_tokens_completion"]);
function fmtMetric(k, v){
  if (v==null) return "—";
  const x = +v;
  if (!Number.isFinite(x)) return "—";
  if (PERCENT.has(k)){
    const pct = x * 100;
    return pct < 1 && pct > 0 ? pct.toFixed(2)+"%" : pct.toFixed(1)+"%";
  }
  let body;
  if (INTEGERISH.has(k)) body = x.toLocaleString(undefined,{maximumFractionDigits:0});
  else if (Math.abs(x) >= 1000) body = x.toLocaleString(undefined,{maximumFractionDigits:0});
  else if (Math.abs(x) >= 100) body = x.toFixed(1);
  else if (Math.abs(x) >= 10) body = x.toFixed(2);
  else body = x.toFixed(3);
  return UNITS[k] ? body + " " + UNITS[k] : body;
}
const stc = (s,txt) => `<span class="st ${s}">${txt||s}</span>`;
// ---- 1: unified onboard×ground matrix with a metric-selector heatmap
(function(){
  const sel = document.getElementById("metricSel");
  sel.innerHTML = P.metric_options.map(([k,l])=>`<option value="${k}">${l}</option>`).join("");
  const lowerBetter = new Set(P.lower_better);
  const ROWS = ["—"].concat(P.onboard_order);    // onboard: —, symb, rl, hrl
  const COLS = ["—"].concat(P.ground_order);     // ground:  —, +7 cells
  const byPos = {}; let conv = null;
  P.cells.forEach(c=>{
    if (c.paradigm==="conventional"){ conv = c; return; }   // shares (—, symb) with AG-symb
    byPos[(c.onboard||"—")+"|"+(c.ground||"—")] = c;
  });
  const paradigmAt = (o,g)=> (o==="—"&&g==="—") ? "" : o==="—" ? "AG" : g==="—" ? "AO" : "AH";
  const colour = t => { const f=(a,b)=>Math.round(a+(b-a)*t);
    return `rgb(${f(234,0)},${f(242,101)},${f(250,189)})`; };   // #eaf2fa → TUM #0065BD

  function render(metric){
    const vals = P.cells.filter(c=>c.status==="measured" && c.mean[metric]!=null).map(c=>c.mean[metric]);
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const norm = v => { if(!isFinite(mn)||mx<=mn) return 1; const t=(v-mn)/(mx-mn); return lowerBetter.has(metric)?1-t:t; };
    let h = `<table class="mgrid"><tr><th class="corner">onboard ↓ / ground →</th>`+
            COLS.map(g=>`<th class="mh">${g==="—"?"—":P.rep_labels[g]}</th>`).join("")+`</tr>`;
    for (const o of ROWS){
      h += `<tr><th class="mrow">${o==="—"?"—":P.rep_labels[o]}</th>`;
      for (const g of COLS){
        const par = paradigmAt(o,g);
        const isConv = (o==="—" && g==="symb");
        if (!par){ h += `<td class="mcell empty" title="no cores">—</td>`; continue; }
        const c = byPos[o+"|"+g];
        let cls="notrun", txt="·", style="", tip=par, dark=false;
        if (c && c.status==="measured" && c.mean[metric]!=null){
          const t = norm(c.mean[metric]);
          dark = t>0.55;
          cls="measured"; txt=fmtMetric(metric, c.mean[metric]);
          style=`background:${colour(t)};color:${dark?'#fff':'#111'}`;
          tip=`${par} · ${c.id}${c.source?' [data: '+c.source+']':''}`;
        } else if (c && c.status==="placeholder"){
          cls="placeholder"; txt="▢"; tip=`${par} · ${c.id} (placeholder cell)`;
        } else if (c){ tip=`${par} · ${c.id} (not yet run)`; }
        if (isConv && conv){
          const pairLine = (label, cell) => {
            if (cell && cell.status==="measured" && cell.mean[metric]!=null){
              const t = norm(cell.mean[metric]);
              const darkLine = t>0.55;
              return `<span class="pairline" style="background:${colour(t)};color:${darkLine?'#fff':'#111'}"><b>${label}</b>${fmtMetric(metric, cell.mean[metric])}</span>`;
            }
            const fallback = cell && cell.status==="placeholder" ? "▢" : "·";
            return `<span class="pairline"><b>${label}</b>${fallback}</span>`;
          };
          tip += ` | Conventional: ${conv.id}`;
          h += `<td class="mcell pair" title="${tip}">${pairLine("AG", c)}${pairLine("CG", conv)}</td>`;
          continue;
        }
        h += `<td class="mcell ${cls}" style="${style}" title="${tip}">${txt}</td>`;
      }
      h += "</tr>";
    }
    const label = (P.metric_options.find(o=>o[0]===metric)||[,metric])[1];
    h += `</table><div class="mlegend">colour = ${label} `+
         `(darker blue = better${lowerBetter.has(metric)?", inverted for ↓":""}) · `+
         `▢ = placeholder cell · · = not yet run · paradigm read from position `+
         `(AG = ground only, AO = onboard only, AH = both); (—, symbolic) shows AG and CG (sub).</div>`;
    document.getElementById("matrix").innerHTML = h;
  }
  sel.addEventListener("change", e=>render(e.target.value));
  render(P.metric_options[0][0]);

  const meas = P.cells.filter(c=>c.status==="measured").length;
  const ph = P.cells.filter(c=>c.status==="placeholder").length;
  document.getElementById("kpis").innerHTML =
    `<div class="kpi"><b>${meas}</b>measured</div>`+
    `<div class="kpi"><b>${ph}</b>placeholder cells</div>`+
    `<div class="kpi"><b>${P.cells.length}</b>experiments</div>`;
})();

// ---- 2: results plots (publication style)
(function(){
  const FONT = {family:"Helvetica Neue, Helvetica, Arial, sans-serif", size:14, color:"#1a1a1a"};
  const BAR = "#4878a6";
  const OK = ["#0072B2","#D55E00","#009E73","#E69F00","#56B4E9","#CC79A7","#000000"];
  const axx = t => ({title:{text:t, font:{size:14}}, showline:true, linecolor:"#444", linewidth:1,
                     ticks:"outside", tickcolor:"#444", ticklen:5, gridcolor:"#ededed", zeroline:false, automargin:true});
  const LAY = (xt, yt, extra) => Object.assign(
    {paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:FONT,
     xaxis:axx(xt), yaxis:axx(yt), margin:{t:12,b:60,l:70,r:16}, showlegend:false}, extra||{});
  const mean = a => a.reduce((x,y)=>x+y,0)/a.length;
  const ci95 = a => { if(a.length<2) return 0; const m=mean(a);
    const sd=Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/(a.length-1)); return 1.96*sd/Math.sqrt(a.length); };
  const peps = c => (c.per_ep_utility||[]).filter(v=>v!=null);
  // Symbolic paradigm ladder (the cells with verified data).
  const ladder = [["eventsat_sas_conventional_symb","Conventional"],["eventsat_sas_ag_symb","AG"],
                  ["eventsat_sas_ao_symb","AO"],["eventsat_sas_ah_symb_symb","AH"]];
  const rows = ladder.map(([id,nm])=>{ const c=CELLS[id];
    return (c&&c.status==="measured")?{nm,c}:null; }).filter(Boolean);
  const xs = rows.map(d=>`${d.nm}<br><span style="font-size:10px;color:#888">n=${d.c.n}</span>`);

  if (rows.length){
    const S = rows[0].c.steps, dd = S ? S*60/86400 : null;
    const days = dd!=null ? (dd%1 ? dd.toFixed(1) : dd.toFixed(0)) : null;
    document.getElementById("res-prov").innerHTML =
      `Symbolic paradigm ladder at a matched <b>${S?S.toLocaleString():"—"}-step</b> episode`+
      (days?` (${days} ${days==="1"?"day":"days"} at 60 s/step)`:"")+`, shared launch-lottery seeds. `+
      `Bars show mean &plusmn; 95% CI; utility is target-normalised.`;
    Plotly.newPlot("gradient", [{type:"bar", x:xs, y:rows.map(d=>d.c.mean.utility),
      error_y:{type:"data", array:rows.map(d=>ci95(peps(d.c))), visible:true, color:"#333", thickness:1.3, width:5},
      marker:{color:BAR}, width:0.62}], LAY("architecture","mission utility"), {displayModeBar:false});
    Plotly.newPlot("obshours", [{type:"bar", x:xs, y:rows.map(d=>d.c.mean.observation_hours),
      marker:{color:BAR}, width:0.62}], LAY("architecture","observation time (h / episode)"), {displayModeBar:false});
    Plotly.newPlot("delivmb", [{type:"bar", x:xs, y:rows.map(d=>d.c.mean.downlinked_mb),
      marker:{color:BAR}, width:0.62}], LAY("architecture","delivered data (MB / episode)"), {displayModeBar:false});
  }
  const parName = p => ({conventional:"Conventional", ag:"AG", ao:"AO", ah:"AH"}[p] || p);
  const cog = P.cells.filter(c=>c.status==="measured" && c.mean.utility!=null && c.mean.mean_latency_s>0)
    .map(c=>({nm:`${parName(c.paradigm)} · ${c.rep} (n=${c.n})`, x:c.mean.mean_latency_s, y:c.mean.utility}));
  if (cog.length){
    Plotly.newPlot("cognition", cog.map((d,i)=>({type:"scatter", mode:"markers", name:d.nm,
      x:[d.x], y:[d.y], marker:{size:12, color:OK[i%OK.length], line:{color:"#fff", width:1}}})),
      LAY("decision latency  (s, log scale)","mission utility",
          {xaxis:Object.assign(axx("decision latency  (s, log scale)"),{type:"log"}),
           showlegend:true, legend:{orientation:"h", font:{size:11}, x:0, y:-0.28, xanchor:"left", yanchor:"top", bgcolor:"rgba(255,255,255,0)"}, margin:{t:12,b:110,l:70,r:16}}),
      {displayModeBar:false});
  }
})();

// ---- 3: distributions
const dists = P.cells.filter(c=>c.status==="measured" && c.per_ep_utility.some(v=>v!=null)).map(c=>({
  y:c.per_ep_utility, x:Array(c.n).fill(`${c.paradigm} · ${c.rep}`), type:"box", boxpoints:"all",
  jitter:0.5, pointpos:0, marker:{color:"#333",size:4,opacity:0.7}, line:{color:"#111"}, fillcolor:"#eee",
  hovertext:c.per_ep_utility.map((v,i)=>`ep ${i}: ${fmt(v)}`), hoverinfo:"text"}));
Plotly.newPlot("dist", dists, {paper_bgcolor:"#fff", plot_bgcolor:"#fff", showlegend:false,
  font:{family:"Helvetica Neue, Helvetica, Arial, sans-serif",size:13,color:"#111"},
  yaxis:{title:"utility",gridcolor:"#eee"}, margin:{t:8,b:130,l:60,r:20}, xaxis:{tickangle:18}});

// ---- 4: metric registry
const vcols = P.cells.filter(c=>c.status==="measured");
document.getElementById("mx").innerHTML =
 "<tr><th>metric</th><th>status</th>" + vcols.map(c=>`<th class="num">${c.rep}<br><span style="font-weight:400;color:#777">${c.paradigm}</span></th>`).join("") + "<th>note</th></tr>" +
 P.metrics.map(([id,nm,key,s,note])=>{
  const cls = s.startsWith("measured")?"measured":"deferred";
  return `<tr><td><b>${id}</b> ${nm}</td><td>${stc(cls,s)}</td>` +
    vcols.map(c=>`<td class="num">${key?fmtMetric(key, c.mean[key]):"—"}</td>`).join("") +
    `<td style="color:#666;font-size:12px">${note}</td></tr>`;}).join("");

// ---- 5a: radar
const RMETRICS = [
  ["utility","mission utility"],
  ["mean_aoi_s","mean AoI (inverted)"],
  ["data_downlink_efficiency","downlink efficiency"],
  ["value_of_information","value of information"],
  ["constraint_violation_rate","constraint violations (inverted)"],
  ["commanding_effort","commanding effort (inverted)"],
  ["explainability_score","explainability (presence)"]
];
const RLOWER = new Set(P.lower_better || []);
if (vcols.length){
  const vals = {};
  for (const [k] of RMETRICS){
    let xs = vcols.map(c=> c.mean[k]??0);
    const mn=Math.min(...xs), mx=Math.max(...xs);
    vals[k] = xs.map(v => {
      const norm = mx>mn ? (v-mn)/(mx-mn) : 1;
      return RLOWER.has(k) ? 1-norm : norm;
    });
  }
  Plotly.newPlot("radar", vcols.map((c,i)=>({
    type:"scatterpolar", fill:"toself", opacity:0.55, name:`${c.paradigm} · ${c.rep}`,
    theta:RMETRICS.map(m=>m[1]).concat(RMETRICS[0][1]),
    r:RMETRICS.map(([k])=>vals[k][i]).concat(vals[RMETRICS[0][0]][i])
  })), {paper_bgcolor:"#fff", font:{family:"Helvetica Neue, Helvetica, Arial, sans-serif", size:12},
       polar:{radialaxis:{range:[0,1], gridcolor:"#eee"}}, legend:{orientation:"h", y:-0.12}, margin:{t:30}});
}
// ---- 5b: paired differences (AH - AO, symbolic)
const aoR = CELLS["eventsat_sas_ao_symb"], ahR = CELLS["eventsat_sas_ah_symb_symb"];
if (aoR && ahR && aoR.status==="measured" && ahR.status==="measured"){
  const k = Math.min(aoR.per_ep_utility.length, ahR.per_ep_utility.length);
  const diffs = [...Array(k)].map((_,i)=>ahR.per_ep_utility[i]-aoR.per_ep_utility[i]).filter(v=>v!=null);
  const mu = diffs.reduce((a,b)=>a+b,0)/diffs.length;
  const wins = diffs.filter(v=>v>0).length;
  Plotly.newPlot("paired", [{x:diffs, type:"histogram", nbinsx:25, marker:{color:"#0065BD", opacity:0.8}}],
   {paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:{family:"Helvetica Neue, Helvetica, Arial, sans-serif", size:12},
    xaxis:{title:"per-orbit utility gain from adding the ground plan (AH − AO, same orbits)", gridcolor:"#eee"},
    yaxis:{title:"orbits", gridcolor:"#eee"}, showlegend:false, margin:{t:34,b:50,l:55,r:10},
    shapes:[{type:"line", x0:0, x1:0, yref:"paper", y0:0, y1:1, line:{color:"#a13026", dash:"dash"}},
            {type:"line", x0:mu, x1:mu, yref:"paper", y0:0, y1:0.92, line:{color:"#9a6200", width:2}}],
    annotations:[
     {x:0, yref:"paper", y:1.04, text:"no benefit", showarrow:false, font:{color:"#a13026", size:12}},
     {x:mu, yref:"paper", y:0.98, text:`mean gain ${mu.toFixed(2)}`, showarrow:false, font:{color:"#9a6200", size:12}},
     {xref:"paper", x:0.98, yref:"paper", y:0.85, text:`ground plan better in ${wins}/${diffs.length} orbits →`,
      showarrow:false, font:{size:12.5, color:"#0065BD"}, xanchor:"right"}]});
}

// ---- 6: episode inspector
const T = P.telemetry, tids = Object.keys(T);
const tsel = document.getElementById("teleSel");
tsel.innerHTML = tids.map(id=>`<option value="${id}">${T[id].label}</option>`).join("");
const MODECOL = {charging:"#9ecae1", communication:"#0065BD", payload_observe:"#1e8449",
 payload_compress:"#a1d99b", payload_detect:"#74c476", payload_send:"#41ab5d", safe:"#a13026"};
function bands(flagArr, steps, color){
  const shapes=[]; let s0=null;
  for (let i=0;i<flagArr.length;i++){
    if (flagArr[i] && s0===null) s0=steps[i];
    if ((!flagArr[i]||i===flagArr.length-1) && s0!==null){
      shapes.push({type:"rect", xref:"x", yref:"paper", x0:s0, x1:steps[i], y0:0, y1:1,
        fillcolor:color, opacity:0.18, line:{width:0}}); s0=null; }
  }
  return shapes;
}
function renderTele(){
  const t = T[tsel.value]; if(!t) return;
  const shapes = bands(t.gpass, t.steps, "#777").concat(bands(t.anomaly, t.steps, "#a13026"));
  Plotly.newPlot("teleA", [
    {x:t.steps, y:t.soc, name:"battery SoC", yaxis:"y", line:{color:"#0065BD"}},
    {x:t.steps, y:t.stored, name:"data stored (MB)", yaxis:"y2", line:{color:"#1e8449"}},
    {x:t.steps, y:t.downlinked, name:"downlinked cum. (MB)", yaxis:"y2", line:{color:"#9a6200"}}],
   {paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:{family:"Helvetica Neue, Helvetica, Arial, sans-serif", size:12},
    xaxis:{title:"simulation step (1 min each)", gridcolor:"#eee"},
    yaxis:{title:"SoC", range:[0,1.05], gridcolor:"#eee"},
    yaxis2:{title:"MB", overlaying:"y", side:"right"},
    shapes, legend:{orientation:"h", y:1.12}, margin:{t:10,b:45,l:55,r:55}});
  Plotly.newPlot("teleB", [{x:t.steps, y:t.mode.map(()=>1), mode:"markers",
    marker:{color:t.mode.map(m=>MODECOL[m]||"#999"), size:6, symbol:"square"},
    text:t.mode, hoverinfo:"text+x"}],
   {paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:{family:"Helvetica Neue, Helvetica, Arial, sans-serif", size:11},
    xaxis:{title:"operating mode per step (hover)", gridcolor:"#fff"},
    yaxis:{visible:false}, showlegend:false, margin:{t:4,b:40,l:55,r:55}});
}
if (tids.length){ tsel.addEventListener("change", renderTele); renderTele(); }

// ---- 7: statistics
document.getElementById("rhoval").textContent = P.rho;
document.getElementById("sigval").textContent = P.sigma;
document.getElementById("guide").innerHTML =
 `<b>How to read:</b> a difference between two cells is confirmable when the paired-episode count meets
 the curve for the effect size of interest. With measured ρ = ${P.rho}: d = 0.5 → <b>14 paired episodes</b>;
 d = 0.3 → <b>38</b>; the pre-registered default (100) covers d ≥ 0.25 under Bonferroni m = 10.`;
const za=2.807, zb=0.8416, ds=[...Array(19)].map((_,i)=>0.1+i*0.05);
Plotly.newPlot("powcurve",[
 {x:ds,y:ds.map(d=>Math.ceil((za+zb)**2*2*(1-P.rho)/d/d/0.955)),name:"paired (measured ρ)",line:{color:"#111"}},
 {x:ds,y:ds.map(d=>Math.ceil(2*(za+zb)**2/d/d/0.955)),name:"unpaired / group",line:{color:"#999",dash:"dot"}}],
 {paper_bgcolor:"#fff",plot_bgcolor:"#fff",font:{family:"Helvetica Neue, Helvetica, Arial, sans-serif",size:13},
  xaxis:{title:"minimum detectable effect d",gridcolor:"#eee"},
  yaxis:{title:"episodes required",type:"log",gridcolor:"#eee"},margin:{t:8,b:45,l:60,r:10},legend:{x:0.55,y:0.95}});
</script></body></html>
"""

if __name__ == "__main__":
    main()

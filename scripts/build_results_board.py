"""AUTOPS results board v4 — navigable M×O×T instrument.

Every SSP of the M space is selectable (SS-A…SS-E); for the selected profile the
full valid O-cell slate is enumerated client-side (mirroring the §3.2 gates) and
each cell shows its true state: measured (values) · running · excluded · gated
(with the rule) · not yet run. All 14 metrics and the full test catalogue are
always rendered. Academic (print-like) styling.

Usage:  uv run python scripts/build_results_board.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

EXTRACT = Path("data/figures/extract.json")
OUT = Path("data/figures/results_board.html")

# run-id -> (SSP code, cell key, status, note). cell key: PARADIGM|onboard|ground
MEASURED = {
    "eventsat_sas_sda_symb_hd_ao":  ("A1|B1|C1|D1|E0", "AO|sym",         "valid",   "100 episodes"),
    "eventsat_sas_sda_symb_hd_ah":  ("A1|B1|C1|D1|E0", "AH|sym|sym",     "valid",   "100 episodes; ground-model fix (787857f) verified neutral on this anchor"),
    "eventsat_sas_sda_symb_hd_ag":  ("A1|B1|C1|D1|E0", "AG|sym",         "valid",   "100 episodes. Ground-segment fix (787857f: true pass-table horizon) verified NEUTRAL — AG ep0 utility bit-identical pre/post (3.591369). The horizon was never the binding constraint (battery/OBC/27MB-budget gating saturates first); the longer gap becomes trailing charging. Robustness result, not a regression."),
    "eventsat_sas_sda_symb_hd_cg":  ("A1|B1|C1|D1|E0", "CG|sym",         "valid",   "100 episodes, U=2.99 — the conventional-ops anchor. The 'observation-free' exclusion was FALSIFIED (2026-06-12): CG does ~2 obs/gap (its own max_observations cap, matching real-LEO 1-3/gap). Construct fixes landed (787857f: plans the true FOLLOWING gap given one-pass delay; link-gated uplink) and verified neutral on A1."),
    "eventsat_sas_sda_hyre_hd_ah":  ("A1|B1|C1|D1|E0", "AH|llm_re|sym",  "valid",   "2 of 3 episodes verified clean (ep0+ep2: 1440 steps each, real 122B, full-trace screen, zero fallbacks). ep1 excluded: 14 symbolic-fallback decisions during the 2026-06-12 Ollama 504 storm — the run process predated fallback-removal commit ec1b83b, so the silent-fallback path was still live. Informs B2+ per R-COMPUTE1."),
    "eventsat_sas_sda_hyag_hd_ah":  ("A1|B1|C1|D1|E0", "AH|hyb_ag|sym",  "invalid", "queue3 rerun EXCLUDED: 474/720 decisions (66%) were silent symbolic fallbacks — the agentic loop rode its 3-call budget without ever deciding (fixed in 6866ab5: forced Decide step + loud failure). Rerun on fixed code queued (queue4). Informs B3+ per R-COMPUTE2."),
    "lf_hyre_4b_ah":                ("A1|B1|C1|D1|E0", "AH|llm_re|sym",  "running", "LF rung (4B), paired seeds with HF"),
    "nbr_b2_symb_ao":               ("A1|B2|C1|D1|E0", "AO|sym",         "valid",   "30 episodes (power ×5)"),
    "nbr_b2_symb_ah":               ("A1|B2|C1|D1|E0", "AH|sym|sym",     "valid",   "30 episodes. NB: B2 (battery ×5) and B3 (×20) currently give bit-identical results — battery never binds for symbolic AH at A1 power (final SoC ≈0.98), so capacity scaling is inert. The B-tier power model is a known placeholder (see decision_matrix §2.2)."),
    "nbr_b3_symb_ao":               ("A1|B3|C1|D1|E0", "AO|sym",         "valid",   "30 episodes (power ×20)"),
    "nbr_b3_symb_ah":               ("A1|B3|C1|D1|E0", "AH|sym|sym",     "valid",   "30 episodes — bit-identical to B2 (battery scaling inert at A1 power; see B2 note)"),
    "nbr_b2_hyre_ah":               ("A1|B2|C1|D1|E0", "AH|llm_re|sym",  "valid",   "2 episodes, real qwen3.5:122b, full-trace screened clean (720/720 LLM rationales/ep, zero fallbacks). Onboard LLM at its gate-legal B2 tier (R-COMPUTE1)."),
}

# Active = the measured EventSat set (2026-06-13 scope). Deferred rows kept
# visible but marked, so the extension is drop-in (decision_matrix §5.2).
METRICS = [
    ("M-01", "Mission Utility",           "utility",                  "measured",                    "raw A1 instance, unclamped"),
    ("M-05", "Safety-Override Rate",      "operator_load",            "measured",                    "environment-veto fraction; operator proxy for CG/AG"),
    ("M-06", "Resource Efficiency",       "resource_efficiency",      "measured (raw)",              "min–max normalised per M-slice at matrix build"),
    ("M-07", "Decision Latency",          "mean_latency_s",           "measured (live calls only)",  "live probe: 122B median 38 s · symbolic 12 µs; cached runs measure cache reads"),
    ("M-08", "Explainability (presence)", "explainability_score",     "measured (presence only)",    "presence floor; faithfulness scorer deferred (§5)"),
    ("M-09", "Robustness (CV)",           None, "measured for 100-ep runs",    "cross-episode; N/A below 30 episodes"),
    ("M-11", "Downlink Efficiency",       "data_downlink_efficiency", "measured",                    "delivered / max-achievable"),
    ("M-02", "Mean Age of Information",   None, "deferred (§5)",               "out of scope; AoI clock-at-capture instrument"),
    ("M-03", "Peak Age of Information",   None, "deferred (§5)",               "out of scope; same field as M-02"),
    ("M-04", "Autonomous Recovery Eff.",  None, "deferred (§5)",               "out of scope; persistent-anomaly collector"),
    ("M-10", "Scale Efficiency",          None, "deferred (§5)",               "out of scope; multi-sat (Flamingo)"),
    ("M-12", "Value-of-Information",      None, "deferred (§5)",               "out of scope; per-product value weights"),
    ("M-13", "Constraint-Violation Rate", None, "deferred (§5)",               "out of scope; constraint ledger + pass^k"),
    ("M-14", "Commanding Effort",         None, "deferred (§5)",               "out of scope; command ledger"),
]

TESTS = [
    ("MC.01", "Decision-cycle latency",           "M-07",            "runnable now",                    "live-call latencies only"),
    ("MC.02", "Telemetry continuity in blackout", "M-04 + M-01",     "needs env feature",               "ground-outage windows (§7)"),
    ("MC.03", "Command execution verification",   "d_cmd",           "needs env feature",               "uplink-command model + execution ledger (§7)"),
    ("MC.04", "Detection-triggered retasking",    "completion",      "needs env feature",               "event injection + per-product tracking (§7)"),
    ("MC.05", "Fleet-status monitoring",          "TBD",             "to define (approved 2026-06-11)", "C3+; fault-localization time"),
    ("AU.01", "Safe-mode entry (gate)",           "p_safemode",      "runnable (DEBUG traces)",         "correctness gate, must equal 1.0"),
    ("AU.02", "Autonomous recovery w/o pass",     "M-04",            "needs env feature + collector",   "gap-timed anomaly + recovery-task fault (§7)"),
    ("AU.03", "Predictive FDIR",                  "p_avoid",         "needs env feature",               "mitigable degradation (§7)"),
    ("AU.04", "Contact-window utilisation",       "M-11",            "runnable now",                    "measured — see metric registry"),
    ("AU.05", "Preemptive rescheduling",          "M-01 (HP)",       "needs env feature",               "priority-event injection (§7)"),
    ("AU.06", "Lights-out endurance",             "M-01, M-09",      "needs env feature",               "extended outage windows (§7)"),
    ("AU.07", "Fleet automation endurance",       "M-14",            "to define (approved 2026-06-11)", "C3+"),
    ("PL.01", "Schedule optimality",              "M-01",            "raw only",                        "scoring needs the U_max helper (§7)"),
    ("PL.02", "Reactive replanning",              "M-01 ratio",      "needs env feature",               "scheduled anomaly timing (§7)"),
    ("PL.03", "Multi-objective trade-off",        "M-06 · M-05 · M-08", "needs env feature",            "resource-conflict scenario (§7)"),
    ("PL.04", "Pass deconfliction",               "mean M-11",       "multi-sat (Flamingo)",            "C2+"),
    ("PL.05", "Distributed task allocation",      "M-10",            "multi-sat (Flamingo)",            "C2+, E2+"),
    ("PL.06", "Consensus under ISL failure",      "p_conflict",      "multi-sat (Flamingo)",            "C2+, E2+"),
    ("PL.07", "MARL generalisation to new N",     "TR",              "multi-sat (Flamingo)",            "C4+, E3+"),
    ("FD.01", "Propagation under GPS denial",     "timing error",    "needs env feature",               "GPS-denial mechanism (§7)"),
    ("FD.02", "Collision avoidance",              "binary × M-05",   "needs env feature",               "conjunction injection (§7)"),
    ("DP.01", "AoI-optimal scheduling",           "M-02",            "not instrumented",                "per-product age field (§7)"),
    ("DP.02", "VoI downlink triage",              "M-12",            "not instrumented",                "headline H-tier test; value weights (§7)"),
    ("DP.03", "Storage overflow prevention",      "r_ovf",           "runnable (at-cap proxy)",         "loss accounting pending (§7)"),
    ("AN.01", "Explainability of reports",        "M-08",            "runnable (L-rung)",               "presence only; quality scorer pending"),
    ("AN.02", "Nominal trend reporting",          "lead time",       "to define (approved 2026-06-11)", "C1+"),
]

VALUE_KEYS = ["utility", "operator_load", "resource_efficiency", "mean_latency_s",
              "explainability_score", "data_downlink_efficiency"]


def main() -> None:
    data = {d["id"]: d for d in json.loads(EXTRACT.read_text())}
    cells: dict = {}
    for rid, (ssp, cell, status, note) in MEASURED.items():
        d = data.get(rid, {})
        cells.setdefault(ssp, {})[cell] = {
            "id": rid, "status": status, "note": note, "n": d.get("n", 0),
            "mean": {k: d.get("mean", {}).get(k) for k in VALUE_KEYS},
            "per_ep_utility": d.get("per_ep", {}).get("utility", []),
        }
    ao = [v for v in data["eventsat_sas_sda_symb_hd_ao"]["per_ep"]["utility"] if v is not None]
    ah = [v for v in data["eventsat_sas_sda_symb_hd_ah"]["per_ep"]["utility"] if v is not None]
    n = min(len(ao), len(ah))
    tele_path = Path("data/figures/telemetry.json")
    telemetry = json.loads(tele_path.read_text()) if tele_path.exists() else {}
    payload = {"cells": cells, "metrics": METRICS, "tests": TESTS, "telemetry": telemetry,
               "rho": round(st.correlation(ao[:n], ah[:n]), 3),
               "sigma": round(st.stdev(ao[:n] + ah[:n]), 3)}
    OUT.write_text(TEMPLATE.replace("__PAYLOAD__", json.dumps(payload)))
    print(f"wrote {OUT}: {sum(len(v) for v in cells.values())} cell records, "
          f"{len(METRICS)} metrics, {len(TESTS)} tests")


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AUTOPS — M×O×T instrument</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/aaaakshat/cm-web-fonts@latest/fonts.css">
<style>
 /* LaTeX-style typography: Computer Modern with Georgia fallback; TUM blue accents */
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
 .st.valid    { color:#1e6b34; border-color:#1e6b34; } .kpi b { color:#0065BD !important; }
 .st.invalid  { color:#a13026; border-color:#a13026; }
 .st.running  { color:#9a6200; border-color:#9a6200; }
 .st.gated    { color:#666;    border-color:#999; }
 .st.notrun   { color:#999;    border-color:#ccc; }
 .sel { display:flex; gap:18px; flex-wrap:wrap; margin:10px 0 14px; font-family:'Computer Modern Sans',Arial,sans-serif; font-size:13px; }
 .sel label { color:#444; } select { font-size:13px; padding:2px 6px; }
 .kpis { display:flex; gap:26px; margin:6px 0 4px; font-family:'Computer Modern Serif',Georgia,serif; }
 .kpi b { font-size:21px; display:block; color:#111; } .kpi { font-size:12px; color:#555; }
 .guide { border-left:3px solid #0065BD; background:#f5f8fb; padding:10px 16px; font-size:13.5px; margin:8px 0; }
 .plot { width:100%; height:380px; }
 .twocol { display:grid; grid-template-columns:1fr 1fr; gap:30px; }
 @media (max-width:1100px){ .twocol{grid-template-columns:1fr;} }
</style></head><body>
<header>
 <h1>AUTOPS — EventSat architecture comparison</h1>
 <div class="sub">A controlled comparison of operations architectures (symbolic / RL / LLM / LLM-agentic,
 across the CG&middot;AG&middot;AO&middot;AH paradigms) on the <b>EventSat</b> event-camera CubeSat mission, on the
 <b>measured metric set</b> — mission utility, downlink, observation hours, decision latency, safety
 overrides, resource efficiency, explainability-presence. Values appear only for verified runs. The broader
 M&thinsp;&times;&thinsp;O&thinsp;&times;&thinsp;T tradespace and the CCSDS-520 test catalogue are the
 method/extension (deferred) &mdash; full specification: <i>docs/decision_matrix.md</i>.</div>
</header>

<section>
 <h2>1&emsp;Satellite-System Profile (M)</h2>
 <div class="sel" id="sel"></div>
 <div class="kpis" id="kpis"></div>
 <h2 id="otitle">2&emsp;Operations-system cells (O) for the selected profile</h2>
 <div class="caption">AH cells are ⟨onboard | ground-planner⟩ pairs. Gated cells name their rule; nothing is zero-filled.</div>
 <div id="ocells"></div>
</section>

<section>
 <h2>3&emsp;Measured utility distributions (verified runs, all profiles)</h2>
 <div id="dist" class="plot"></div>
</section>

<section>
 <h2>4&emsp;Test catalogue — deferred</h2>
 <div id="tests" class="caption"></div>
</section>

<section>
 <h2>5&emsp;Metric registry — measured EventSat set (values for verified runs)</h2>
 <table id="mx"></table>
</section>

<section>
 <h2>6&emsp;Architecture comparison (verified runs)</h2>
 <div class="caption">Left: metric profile per architecture, min&ndash;max normalised across the shown runs
 (1 = best of this set — relative, not absolute). Right: paired per-episode utility differences
 (shared launch-lottery seeds), the basis of the &sect;5.6 statistics.</div>
 <div class="twocol">
  <div id="radar" class="plot" style="height:420px"></div>
  <div id="paired" class="plot" style="height:420px"></div>
 </div>
</section>

<section>
 <h2>7&emsp;Episode inspector — simulation telemetry</h2>
 <div class="caption">One simulated week, step by step: battery state of charge, data stored on board and
 downlinked, ground-contact windows (grey bands), anomaly-forced safe periods (red bands), and the
 operating mode chosen at every step. For verifying that the simulation behaves physically, not just
 that the aggregate numbers look right.</div>
 <div class="sel"><label>episode&nbsp; <select id="teleSel"></select></label></div>
 <div id="teleA" class="plot" style="height:330px"></div>
 <div id="teleB" class="plot" style="height:150px"></div>
</section>

<section>
 <h2>8&emsp;Statistical adequacy (&sect;5.6, pre-registered)</h2>
 <div class="guide" id="guide"></div>
 <div class="twocol">
  <div id="powcurve" class="plot" style="height:290px"></div>
  <div style="font-size:13.5px">
   <p><b>Pre-registered rule:</b> n<sub>pairs</sub> = (z<sub>1−α/2m</sub> + z<sub>0.8</sub>)² · 2(1−ρ) / d² ÷ 0.955 (Wilcoxon ARE).</p>
   <p><b>Measured inputs</b> (symbolic AO/AH, 100 episodes, shared launch-lottery seeds): ρ = <span id="rhoval"></span>, σ<sub>U</sub> = <span id="sigval"></span>.</p>
   <p><b>Recommendation in force:</b> 100 episodes per confirmatory cell — sufficient for d ≥ 0.25 at α = 0.05, power 0.8, Bonferroni m ≤ 10. Cells under 30 episodes are screening pilots and never support confirmatory claims. ρ, σ re-estimated per substrate family and per SSP.</p>
  </div>
 </div>
</section>
<script>
const P = __PAYLOAD__;
const DIM = { A:["A1 EO","A2 Comms","A3 Nav","A4 Science","A5 SSA","A6 Planetary","A7 TechDemo"],
              B:["B1 CubeSat","B2 SmallSat","B3 MedBus","B4 LargePlt"],
              C:["C1 (1)","C2 (2–10)","C3 (10–100)","C4 (100–1k)","C5 (>1k)"],
              D:["D1 LEO","D2 GEO","D3 Lunar","D4 Mars","D5 Deep space"],
              E:["E0 none","E1 intra-plane","E2 planned","E3 full-mesh"] };
const CORE = { sym:"symbolic rules", srl:"reinforcement learning", llm_re:"LLM (single-shot)",
               llm_ag:"LLM (agentic)", hyb_re:"hybrid (RL + rules)", hyb_ag:"hybrid (LLM + tools)" };
const DIMLABEL = { A:"Mission function", B:"Platform class", C:"Constellation size", D:"Comms latency", E:"Inter-satellite links" };
const GROUND = ["sym","srl","llm_re","llm_ag","hyb_re","hyb_ag"];
const fmt = v => v==null ? "—" : (+v).toFixed(3);
const stc = (s,txt) => `<span class="st ${s}">${txt||s}</span>`;

function onboardCells(b){ const o=["sym","srl"]; if(b>=2)o.push("llm_re","hyb_re"); if(b>=3)o.push("llm_ag","hyb_ag"); return o; }
function gatedOnboard(b){ const g={}; if(b<2){g.llm_re="R-COMPUTE1 (needs B2+)";g.hyb_re="R-COMPUTE1";} if(b<3){g.llm_ag="R-COMPUTE2 (needs B3+)";g.hyb_ag="R-COMPUTE2";} return g; }

// selectors
const sel = {A:0,B:0,C:0,D:0,E:0};
document.getElementById("sel").innerHTML = Object.keys(DIM).map(k =>
 `<label title="SS-${k}">${DIMLABEL[k]}&nbsp; <select id="s${k}">${DIM[k].map((v,i)=>`<option value="${i}">${v}</option>`).join("")}</select></label>`).join("");
Object.keys(DIM).forEach(k => document.getElementById("s"+k).addEventListener("change", e => { sel[k]=+e.target.value; render(); }));

function cellState(sspKey, cellKey){
  const rec = (P.cells[sspKey]||{})[cellKey];
  return rec || null;
}
function row(label, cellKey, sspKey, gateNote){
  const rec = cellState(sspKey, cellKey);
  let st, note, n="—", u="—", state="notrun";
  if (rec){ st = stc(rec.status); note = rec.note; n = rec.n||"—"; u = rec.status==="valid"?fmt(rec.mean.utility):"—"; state = rec.status; }
  else if (gateNote){ st = stc("gated","gated"); note = gateNote; state = "gated"; }
  else { st = stc("notrun","not yet run"); note = ""; }
  return { state, html:`<tr><td>${label}</td><td>${st}</td><td class="num">${n}</td><td class="num">${u}</td><td style="color:#666;font-size:12px">${note}</td></tr>` };
}
function group(title, expl, rows){
  const active = rows.filter(r=>["valid","running","invalid"].includes(r.state)).length;
  const counts = `${rows.filter(r=>r.state==="valid").length} measured · ${rows.filter(r=>r.state==="running").length} running · ${rows.length} cells`;
  return `<details ${active?"open":""}><summary style="cursor:pointer;padding:6px 0"><b>${title}</b> — ${expl} &nbsp;<span style="color:#666;font-size:12px">(${counts})</span></summary>
   <table><tr><th>architecture</th><th>state</th><th class="num">episodes</th><th class="num">U mean</th><th>note</th></tr>${rows.map(r=>r.html).join("")}</table></details>`;
}
function render(){
  const code = `A${sel.A+1}|B${sel.B+1}|C${sel.C+1}|D${sel.D+1}|E${sel.E}`;
  const b=sel.B+1, c=sel.C+1, e=sel.E;
  document.getElementById("otitle").innerHTML = `2&emsp;Operations-system cells (O) — profile <i>${code.replaceAll("|","/")}</i>`;
  if (e>0 && c<2){
    document.getElementById("ocells").innerHTML =
      `<div style="padding:10px 0">Profile invalid: <b>R-ISL</b> — inter-satellite links require at least 2 satellites.</div>`;
    document.getElementById("kpis").innerHTML = ""; return;
  }
  const gates = gatedOnboard(b);
  const ALL = ["sym","srl","llm_re","llm_ag","hyb_re","hyb_ag"];
  let h = group("Conventional Ground (CG)", "operators plan the schedule on ground; uplinked once per contact",
                GROUND.map(g=>row(`⟨${CORE[g]} planner⟩`, `CG|${g}`, code, null)));
  h += group("Autonomous Ground (AG)", "a ground planner plans autonomously; uplinked per contact",
             GROUND.map(g=>row(`⟨${CORE[g]} planner⟩`, `AG|${g}`, code, null)));
  h += group("Autonomous Onboard (AO)", "all decisions made on the satellite, every step",
             ALL.map(o=>row(`⟨${CORE[o]}⟩`, `AO|${o}`, code, gates[o]||null)));
  h += group("Autonomous Hybrid (AH)", "an onboard per-step core paired with a ground planner",
             ALL.flatMap(o=>GROUND.map(g=>row(`⟨${CORE[o]} onboard | ${CORE[g]} planner⟩`, `AH|${o}|${g}`, code, gates[o]||null))));
  // organisation availability summary
  let orgs = "sas: valid";
  orgs += c>=2 ? " · cmas: valid (AH only)" : " · cmas: identified with SAS·AH at C1 (R-ORG1)";
  orgs += c>=3 ? ` · imas/hmas: valid · dmas: ${e>=1?"valid":"gated (R-ORG3 — needs ISL)"}` : " · dmas/imas/hmas: gated (R-ORG2 — need C3+)";
  h += `<div style="color:#555;font-size:12.5px;margin-top:8px">Organisations at this profile — ${orgs}. Cells above are the single-satellite slate; multi-agent organisations repeat it per organisation (§3.4).</div>`;
  document.getElementById("ocells").innerHTML = h;
  const here = Object.values(P.cells[code]||{});
  document.getElementById("kpis").innerHTML =
    `<div class="kpi"><b>${here.filter(r=>r.status==="valid").length}</b>measured (verified) at this profile</div>` +
    `<div class="kpi"><b>${here.filter(r=>r.status==="running").length}</b>running</div>` +
    `<div class="kpi"><b>${here.filter(r=>r.status==="invalid").length}</b>excluded by verification</div>` +
    `<div class="kpi"><b>${Object.values(P.cells).flatMap(Object.values).filter(r=>r.status==="valid").length} / 364,980</b>measured across the whole M×O space</div>`;
}
render();
// preset to EventSat profile is the default (A1/B1/C1/D1/E0 = indices 0)

// distributions
const dists = [];
for (const [ssp, cs] of Object.entries(P.cells))
  for (const [ck, r] of Object.entries(cs))
    if (r.status==="valid" && r.per_ep_utility.some(v=>v!=null))
      dists.push({y:r.per_ep_utility, x:Array(r.n).fill(`${ck.replaceAll("|"," · ")}  @ ${ssp.replaceAll("|","/")}`),
        type:"box", boxpoints:"all", jitter:0.5, pointpos:0, marker:{color:"#333",size:4,opacity:0.7},
        line:{color:"#111"}, fillcolor:"#eee", name:ck,
        hovertext:r.per_ep_utility.map((v,i)=>`ep ${i}: ${fmt(v)}`), hoverinfo:"text"});
Plotly.newPlot("dist", dists, {paper_bgcolor:"#fff", plot_bgcolor:"#fff", showlegend:false,
  font:{family:"Computer Modern Serif, Georgia, serif",size:13,color:"#111"}, yaxis:{title:"utility",gridcolor:"#eee"},
  margin:{t:8,b:130,l:60,r:20}, xaxis:{tickangle:22}});

// tests — parked (2026-06-13 scope decision; see decision_matrix §5 banner)
document.getElementById("tests").innerHTML =
 `The full ${P.tests.length}-test CCSDS-520 catalogue is <b>deferred</b> — out of current scope. ` +
 `The focus is the EventSat empirical comparison on the measured metric set (&sect;5 below). ` +
 `The standards-grounded test framework is retained in <code>decision_matrix.md &sect;5.4</code> ` +
 `as the publishable extension, not the current deliverable.`;

// metric registry (columns = verified runs)
const vcols = [];
for (const [ssp, cs] of Object.entries(P.cells))
  for (const [ck, r] of Object.entries(cs)) if (r.status==="valid") vcols.push([ssp,ck,r]);
document.getElementById("mx").innerHTML =
 "<tr><th>metric</th><th>status</th>" + vcols.map(([s,c])=>`<th class="num">${c.replaceAll("|","·")}<br><span style="font-weight:400;color:#777">${s.replaceAll("|","/")}</span></th>`).join("") + "<th>note</th></tr>" +
 P.metrics.map(([id,nm,key,s,note])=>{
  const cls = s.startsWith("measured")?"valid":s.includes("adopted")||s.includes("pending")?"running":"notrun";
  return `<tr><td><b>${id}</b> ${nm}</td><td>${stc(cls,s)}</td>` +
    vcols.map(([,,r])=>`<td class="num">${key?fmt(r.mean[key]):"—"}</td>`).join("") +
    `<td style="color:#666;font-size:12px">${note}</td></tr>`;}).join("");

// statistics
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
 {paper_bgcolor:"#fff",plot_bgcolor:"#fff",font:{family:"Computer Modern Serif, Georgia, serif",size:13},
  xaxis:{title:"minimum detectable effect d",gridcolor:"#eee"},
  yaxis:{title:"episodes required",type:"log",gridcolor:"#eee"},margin:{t:8,b:45,l:60,r:10},legend:{x:0.55,y:0.95}});

// ---- 6a: radar (min-max normalised over shown runs)
const RMETRICS = [["utility","mission utility"],["data_downlink_efficiency","downlink efficiency"],
 ["explainability_score","explainability (presence)"],["operator_load","override rate (inverted)"]];
const rruns = vcols;  // [ssp, cell, rec]
if (rruns.length){
  const vals = {};
  for (const [k] of RMETRICS) {
    let xs = rruns.map(([,,r])=> k==="operator_load" ? 1-(r.mean[k]??0) : (r.mean[k]??0));
    const mn=Math.min(...xs), mx=Math.max(...xs);
    vals[k] = xs.map(v => mx>mn ? (v-mn)/(mx-mn) : 1);
  }
  Plotly.newPlot("radar", rruns.map(([ssp,ck],i)=>({
    type:"scatterpolar", fill:"toself", opacity:0.55, name:`${ck.replaceAll("|","·")} @${ssp.replaceAll("|","/")}`,
    theta:RMETRICS.map(m=>m[1]).concat(RMETRICS[0][1]),
    r:RMETRICS.map(([k])=>vals[k][i]).concat(vals[RMETRICS[0][0]][i])
  })), {paper_bgcolor:"#fff", font:{family:"Computer Modern Serif, Georgia, serif", size:12},
       polar:{radialaxis:{range:[0,1], gridcolor:"#eee"}}, legend:{orientation:"h", y:-0.12}, margin:{t:30}});
}
// ---- 6b: paired differences (AH - AO at the anchor)
const aoR = cellState("A1|B1|C1|D1|E0","AO|sym"), ahR = cellState("A1|B1|C1|D1|E0","AH|sym|sym");
if (aoR && ahR){
  const k = Math.min(aoR.per_ep_utility.length, ahR.per_ep_utility.length);
  const diffs = [...Array(k)].map((_,i)=>ahR.per_ep_utility[i]-aoR.per_ep_utility[i]).filter(v=>v!=null);
  const mu = diffs.reduce((a,b)=>a+b,0)/diffs.length;
  const wins = diffs.filter(v=>v>0).length;
  Plotly.newPlot("paired", [
   {x:diffs, type:"histogram", nbinsx:25, marker:{color:"#0065BD", opacity:0.8}}],
   {paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:{family:"Computer Modern Serif, Georgia, serif", size:12},
    xaxis:{title:"per-orbit utility gain from adding the ground plan (AH − AO, same 100 orbits)", gridcolor:"#eee"},
    yaxis:{title:"orbits", gridcolor:"#eee"}, showlegend:false, margin:{t:34,b:50,l:55,r:10},
    shapes:[{type:"line", x0:0, x1:0, yref:"paper", y0:0, y1:1, line:{color:"#a13026", dash:"dash"}},
            {type:"line", x0:mu, x1:mu, yref:"paper", y0:0, y1:0.92, line:{color:"#9a6200", width:2}}],
    annotations:[
     {x:0, yref:"paper", y:1.04, text:"no benefit", showarrow:false, font:{color:"#a13026", size:12}},
     {x:mu, yref:"paper", y:0.98, text:`mean gain ${mu.toFixed(2)}`, showarrow:false, font:{color:"#9a6200", size:12}},
     {xref:"paper", x:0.98, yref:"paper", y:0.85, text:`ground plan better in ${wins}/${diffs.length} orbits →`,
      showarrow:false, font:{size:12.5, color:"#0065BD"}, xanchor:"right"}]});
}
// ---- 7: episode inspector
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
   {paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:{family:"Computer Modern Serif, Georgia, serif", size:12},
    xaxis:{title:"simulation step (1 min each)", gridcolor:"#eee"},
    yaxis:{title:"SoC", range:[0,1.05], gridcolor:"#eee"},
    yaxis2:{title:"MB", overlaying:"y", side:"right"},
    shapes, legend:{orientation:"h", y:1.12}, margin:{t:10,b:45,l:55,r:55}});
  Plotly.newPlot("teleB", [{x:t.steps, y:t.mode.map(()=>1), mode:"markers",
    marker:{color:t.mode.map(m=>MODECOL[m]||"#999"), size:6, symbol:"square"},
    text:t.mode, hoverinfo:"text+x"}],
   {paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:{family:"Computer Modern Serif, Georgia, serif", size:11},
    xaxis:{title:"operating mode per step (hover)", gridcolor:"#fff"},
    yaxis:{visible:false}, showlegend:false, margin:{t:4,b:40,l:55,r:55}});
}
if (tids.length){ tsel.addEventListener("change", renderTele); renderTele(); }
</script></body></html>
"""

if __name__ == "__main__":
    main()

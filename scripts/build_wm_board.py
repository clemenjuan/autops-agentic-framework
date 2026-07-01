"""Build the EventSat World-Model planner board (served separately on :8801).

A focused comparison surface for onboard world-model planners, so variants can be
compared side by side: action selector (CEM-MPC vs policy-guided beam over the LeWM
predictor vs model-free RL), plan-hold / Jetson duty cycle, horizon, CEM width, etc.

Auto-discovers every WM-track run under ``data/results`` (by its representation type),
reads ``results.json`` for metrics and ``config.json`` for the planner knobs, and draws
a per-run SOC curve from ``decisions_ep0.jsonl`` when present. Static output →
``data/figures/wm_board/index.html``. Rebuilt by ``scripts/refresh_board.py``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path("data/results")
OUT = Path("data/figures/wm_board/index.html")

WM_TYPES = {"lewm_cem_eventsat", "dreamerv3_eventsat"}
# Reference (non-WM) baselines shown greyed for context, keyed by result id.
REFERENCE_IDS = {
    "eventsat_sas_ao_symb": "Symbolic · rules on OBC (no Jetson power tax) — AO reference",
}

# metric key -> (label, "higher"|"lower" better, format)
METRICS = [
    ("utility", "Utility", "higher", "{:.3f}"),
    ("downlinked_mb", "Downlink MB", "higher", "{:.2f}"),
    ("final_battery_soc", "Final SOC", "higher", "{:.3f}"),
    ("safety_overrides", "Safety ovr.", "lower", "{:.0f}"),
    ("observation_hours", "Obs. hrs", "neutral", "{:.2f}"),
    ("planner_latency_s", "Plan lat. s", "lower", "{:.3f}"),
]


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _selector(rc: dict) -> tuple[str, str]:
    """Return (selector label, css class) for a representation config."""
    t = rc.get("type", "")
    if t == "dreamerv3_eventsat":
        return "DreamerV3 · model-free RL", "sel-rl"
    if t == "lewm_cem_eventsat":
        sel = str(rc.get("selector", "cem")).lower()
        if sel in ("beam", "rl", "policy"):
            return "LeWM · Beam (RL top-k)", "sel-beam"
        return "LeWM · CEM (MPC)", "sel-cem"
    return t or "unknown", "sel-cem"


def _config_summary(rc: dict) -> str:
    t = rc.get("type", "")
    horizon = rc.get("horizon", "?")
    if t == "lewm_cem_eventsat":
        sel = str(rc.get("selector", "cem")).lower()
        hold = int(rc.get("plan_hold", 1) or 1)
        width = (
            f"beam={rc.get('beam_width', '?')}"
            if sel in ("beam", "rl", "policy")
            else f"{rc.get('samples', '?')}×{rc.get('cem_iterations', '?')}"
        )
        return f"H={horizon} · hold={hold} · {width} · {rc.get('mission_mode', '?')}"
    return f"H={horizon} · heuristic-fallback"


def _jetson_duty(rc: dict) -> float | None:
    if rc.get("type") == "lewm_cem_eventsat":
        return 1.0 / max(1, int(rc.get("plan_hold", 1) or 1))
    if rc.get("type") == "dreamerv3_eventsat":
        return 1.0
    return None


def _soc_curve(rid: str, n_pts: int = 160) -> list[float]:
    f = RESULTS / rid / "decisions_ep0.jsonl"
    if not f.exists():
        return []
    socs: list[float] = []
    with f.open() as fh:
        for line in fh:
            try:
                socs.append(float(json.loads(line).get("battery_soc", 0.0)))
            except Exception:
                continue
    if len(socs) <= n_pts:
        return socs
    step = len(socs) / n_pts
    return [socs[min(len(socs) - 1, int(i * step))] for i in range(n_pts)]


def _sparkline(socs: list[float], w: int = 220, h: int = 40) -> str:
    if not socs:
        return '<span class="nodata">no trace</span>'
    n = len(socs)
    pts = " ".join(
        f"{(i/(n-1))*w:.1f},{h - max(0.0, min(1.0, s))*h:.1f}" for i, s in enumerate(socs)
    )
    # reserve line at 0.5 and safe floor at 0.2
    y_res = h - 0.5 * h
    y_safe = h - 0.2 * h
    end = socs[-1]
    stroke = "#1a7f37" if end > 0.4 else ("#d1242f" if end < 0.15 else "#bf8700")
    return (
        f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<line x1="0" y1="{y_res:.1f}" x2="{w}" y2="{y_res:.1f}" class="ref"/>'
        f'<line x1="0" y1="{y_safe:.1f}" x2="{w}" y2="{y_safe:.1f}" class="ref safe"/>'
        f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="1.6"/>'
        f"</svg>"
    )


def collect() -> tuple[list[dict], list[dict]]:
    wm, ref = [], []
    for cfgp in sorted(RESULTS.glob("*/config.json")):
        rid = cfgp.parent.name
        cfg = _load(cfgp) or {}
        res = _load(cfgp.parent / "results.json") or {}
        rc = cfg.get("representation_config", {}) or {}
        mean = res.get("experiment_statistics", {}).get("mean", {}) or {}
        row = {
            "id": rid,
            "desc": (cfg.get("description") or "").strip(),
            "rc": rc,
            "mean": mean,
            "n": len(res.get("episodes", [])),
            "steps": cfg.get("max_steps"),
            "socs": _soc_curve(rid),
        }
        if rid in REFERENCE_IDS:
            row["ref_label"] = REFERENCE_IDS[rid]
            ref.append(row)
        elif rc.get("type") in WM_TYPES:
            wm.append(row)
    # order WM rows: LeWM-CEM, LeWM-Beam, DreamerV3, then by utility desc
    def key(r):
        _, cls = _selector(r["rc"])
        order = {"sel-cem": 0, "sel-beam": 1, "sel-rl": 2}.get(cls, 3)
        u = r["mean"].get("utility")
        return (order, -(u if isinstance(u, (int, float)) else -1e9))

    wm.sort(key=key)
    return wm, ref


def _best(wm: list[dict]):
    best = {}
    for key, _, better, _ in METRICS:
        if better == "neutral":
            continue
        vals = [(r["id"], r["mean"].get(key)) for r in wm if isinstance(r["mean"].get(key), (int, float))]
        if not vals:
            continue
        best[key] = (max if better == "higher" else min)(vals, key=lambda kv: kv[1])[0]
    return best


def _cell(r, key, fmt):
    v = r["mean"].get(key)
    return fmt.format(v) if isinstance(v, (int, float)) else "—"


def render(wm: list[dict], ref: list[dict]) -> str:
    best = _best(wm)
    head_cells = "".join(f"<th>{lbl}</th>" for _, lbl, _, _ in METRICS)

    def row_html(r, is_ref=False):
        sel_lbl, sel_cls = _selector(r["rc"])
        duty = _jetson_duty(r["rc"])
        duty_s = f"1/{int(round(1/duty))}" if duty and duty < 1 else ("every step" if duty else "—")
        artifact = r["mean"].get("artifact_loaded")
        badge = ""
        if r["rc"].get("type") == "lewm_cem_eventsat":
            badge = (
                '<span class="ok">artifact</span>' if artifact == 1.0 else '<span class="warn">fallback</span>'
            )
        elif r["rc"].get("type") == "dreamerv3_eventsat":
            badge = (
                '<span class="ok">policy</span>'
                if r["mean"].get("policy_loaded") == 1.0
                else '<span class="warn">untrained · heuristic stub</span>'
            )
        name = r.get("ref_label") if is_ref else sel_lbl
        cls = "ref-row" if is_ref else sel_cls
        cells = ""
        for key, _, _, fmt in METRICS:
            hit = (not is_ref) and best.get(key) == r["id"]
            cells += f'<td class="{"best" if hit else ""}">{_cell(r, key, fmt)}</td>'
        n = f'n={r["n"]}' if r["n"] else ""
        steps = r["steps"]
        dur = f'{steps//1440}d' if isinstance(steps, int) and steps % 1440 == 0 else (f"{steps} st" if steps else "")
        cfg = f"{dur} · {n} · rules on OBC" if is_ref else f"{_config_summary(r['rc'])} · Jetson {duty_s} · {dur} {n}"
        return f"""<tr class="{cls}">
      <td class="exp"><div class="sel">{name} {badge}</div>
        <div class="eid">{r['id']}</div>
        <div class="cfg">{cfg}</div></td>
      {cells}
      <td class="spark-cell">{_sparkline(r['socs'])}</td>
    </tr>"""

    wm_rows = "\n".join(row_html(r) for r in wm) or '<tr><td colspan="99" class="nodata">no WM runs yet — run a config under configs/experiments/world_model/</td></tr>'
    ref_rows = "\n".join(row_html(r, is_ref=True) for r in ref)
    ref_block = (
        f'<h2>Reference baseline</h2><table class="board"><thead><tr><th class="exp">Baseline</th>{head_cells}<th>SOC over run</th></tr></thead><tbody>{ref_rows}</tbody></table>'
        if ref
        else ""
    )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EventSat · World-Model Planner Board</title>
<style>
 :root {{ --line:#e3e6ea; --muted:#616b76; --blue:#0969da; --bg:#f6f8fa; }}
 * {{ box-sizing:border-box; }}
 body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
   margin:0; color:#1f2328; background:var(--bg); }}
 header {{ padding:26px 40px 14px; }}
 h1 {{ margin:0 0 6px; font-size:22px; }}
 .sub {{ color:var(--muted); font-size:13.5px; max-width:1000px; line-height:1.5; }}
 main {{ padding:8px 40px 40px; }}
 h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:26px 0 8px; }}
 table.board {{ border-collapse:collapse; width:100%; background:#fff; border:1px solid var(--line);
   border-radius:12px; overflow:hidden; font-size:13px; }}
 .board th, .board td {{ padding:10px 12px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }}
 .board th {{ background:#fbfcfd; color:var(--muted); font-weight:600; font-size:12px; text-align:right; }}
 .board th.exp, .board td.exp {{ text-align:left; white-space:normal; min-width:300px; }}
 .board tr:last-child td {{ border-bottom:none; }}
 td.best {{ background:#eaf7ee; font-weight:700; color:#0a5c2b; }}
 .sel {{ font-weight:650; font-size:13.5px; }}
 .eid {{ font-family:ui-monospace,monospace; font-size:11.5px; color:var(--blue); margin-top:1px; }}
 .cfg {{ color:var(--muted); font-size:11.5px; margin-top:3px; }}
 .sel-cem .sel {{ color:#0969da; }} .sel-beam .sel {{ color:#8250df; }} .sel-rl .sel {{ color:#bc4c00; }}
 .ref-row {{ background:#fbfbfc; color:var(--muted); }} .ref-row .sel {{ color:#57606a; }}
 .ok {{ font-size:10px; color:#0a5c2b; border:1px solid #accbb1; border-radius:4px; padding:0 5px; margin-left:4px; }}
 .warn {{ font-size:10px; color:#9a6200; border:1px solid #d9b271; border-radius:4px; padding:0 5px; margin-left:4px; }}
 .spark-cell {{ text-align:center; }}
 svg.spark .ref {{ stroke:#c9d1d9; stroke-width:1; stroke-dasharray:3 3; }}
 svg.spark .ref.safe {{ stroke:#f0c9c9; }}
 .nodata {{ color:#a0a8b0; font-size:12px; font-style:italic; }}
 .legend {{ margin-top:18px; color:var(--muted); font-size:12px; line-height:1.6; max-width:1000px; }}
 .legend b {{ color:#1f2328; }} code {{ font-family:ui-monospace,monospace; background:#eef1f4; padding:1px 5px; border-radius:4px; font-size:12px; }}
 footer {{ color:var(--muted); font-size:11.5px; padding:0 40px 30px; }}
</style></head><body>
<header>
 <h1>EventSat · World-Model Planner Board</h1>
 <div class="sub">Onboard (AO) planners over the frozen LeWM world model, compared under the real
 Jetson power tax (+7&nbsp;W while the onboard core plans). The axis of interest is the <b>action
 selector</b> over the same predictor — CEM-MPC vs a policy-guided top-k <b>beam</b> vs model-free RL —
 alongside the plan-hold / Jetson duty cycle and horizon. Higher is better for utility / downlink / SOC;
 lower for safety-overrides / latency. Green = best among WM runs.</div>
</header>
<main>
 <h2>World-model planners</h2>
 <table class="board"><thead><tr><th class="exp">Selector · run</th>{head_cells}<th>SOC over run</th></tr></thead>
 <tbody>{wm_rows}</tbody></table>
 {ref_block}
 <div class="legend">
  <b>Selector</b> — <span style="color:#0969da">CEM (MPC)</span>: sample mode sequences, refine elites.
  <span style="color:#8250df">Beam (RL top-k)</span>: an RL/policy step proposes the top-k actions, each
  propagated through the LeWM predictor for H steps (same WM, different search). <span style="color:#bc4c00">DreamerV3</span>:
  model-free RL policy (heuristic fallback until a trained policy artifact is supplied).<br>
  <b>Jetson duty</b> — fraction of steps the onboard core runs inference. <code>1/12</code> = plan once,
  execute the cached 12-step schedule with the Jetson asleep. Continuous (<code>every step</code>) planning
  is power-negative at this orbit's 62% sun fraction.<br>
  <b>SOC curve</b> — battery state over the whole run; dashed lines mark reserve (0.5) and the safe floor (0.2).
 </div>
</main>
<footer>Auto-generated {ts} by <code>scripts/build_wm_board.py</code> (via <code>refresh_board.py</code>).
 Served on <code>:8801</code>. Add a run: drop a config in <code>configs/experiments/world_model/</code> and it appears here.</footer>
</body></html>"""


def main() -> None:
    wm, ref = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(wm, ref))
    print(f"wm board: {len(wm)} WM run(s), {len(ref)} reference → {OUT}")


if __name__ == "__main__":
    main()

"""Extract compact per-episode telemetry from DEBUG decision traces for the board.
Downsamples to ~1500 points; writes data/figures/telemetry.json
{run: {label, steps[], soc[], stored[], downlinked[], pass[], anomaly[], mode[]}}."""
import json
from pathlib import Path

SOURCES = {
    "telemetry_symb_hd_ah": "Symbolic · AH — EventSat (1 episode)",
    "telemetry_symb_hd_ao": "Symbolic · AO — EventSat (1 episode)",
    "telemetry_symb_hd_ag": "Symbolic · AG (fixed planner) — EventSat (1 episode)",
    "eventsat_sas_sda_hyre_hd_ah": "LLM 122B single-shot · AH — EventSat (rerun ep0)",
}
out = {}
for rid, label in SOURCES.items():
    tr = Path(f"data/results/{rid}/decisions_ep0.jsonl")
    if not tr.exists():
        continue
    rows = [json.loads(l) for l in tr.open()]
    k = max(1, len(rows) // 1500)
    rows_ds = rows[::k]
    out[rid] = {
        "label": label,
        "steps":      [r["step"] for r in rows_ds],
        "soc":        [round(r.get("battery_soc") or 0, 4) for r in rows_ds],
        "stored":     [round(r.get("data_stored_mb") or 0, 2) for r in rows_ds],
        "downlinked": [round(r.get("data_downlinked_mb") or 0, 2) for r in rows_ds],
        "gpass":      [int(bool(r.get("ground_pass_active"))) for r in rows_ds],
        "anomaly":    [int(bool(r.get("anomaly_forced_safe"))) for r in rows_ds],
        "mode":       [r.get("mode", "?") for r in rows_ds],
    }
Path("data/figures/telemetry.json").write_text(json.dumps(out))
print("telemetry extracted:", list(out))

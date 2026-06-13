"""Extract compact per-episode telemetry for the board's Episode inspector.

Preferred source: the ``telemetry`` block now embedded in each run's
``results.json`` (ExperimentRunner writes it for the first few episodes,
regardless of log level — see TELEMETRY_SAMPLE_EPISODES). Falls back to the
legacy DEBUG ``decisions_ep0.jsonl`` trace for older runs that predate the
embedded block.

Writes data/figures/telemetry.json:
  {run: {label, steps[], soc[], stored[], downlinked[], gpass[], anomaly[], mode[]}}
"""
import json
from pathlib import Path

# run_id -> inspector label. Embedded-telemetry runs are preferred; the
# legacy telemetry_* single-episode DEBUG runs remain as a fallback source.
SOURCES = {
    "eventsat_sas_sda_symb_hd_ah": "Symbolic · AH — EventSat (ep0)",
    "eventsat_sas_sda_symb_hd_ao": "Symbolic · AO — EventSat (ep0)",
    "eventsat_sas_sda_symb_hd_ag": "Symbolic · AG — EventSat (ep0)",
    "eventsat_sas_sda_symb_hd_cg": "Symbolic · CG — EventSat (ep0)",
    "eventsat_sas_sda_hyre_hd_ah": "LLM 122B single-shot · AH — EventSat (ep0)",
}
MAX_POINTS = 1500


def _from_embedded(rid: str):
    """Read the telemetry block from results.json (episode 0), if present."""
    rj = Path(f"data/results/{rid}/results.json")
    if not rj.exists():
        return None
    try:
        data = json.loads(rj.read_text())
    except (ValueError, OSError):
        return None
    eps = data.get("episodes") or []
    if not eps:
        return None
    tel = eps[0].get("telemetry")
    if not tel:
        return None
    return {
        "steps": tel.get("steps", []),
        "soc": tel.get("soc", []),
        "stored": tel.get("stored", []),
        "downlinked": tel.get("downlinked", []),
        "gpass": tel.get("gpass", []),
        "anomaly": tel.get("anomaly", []),
        "mode": tel.get("mode", []),
    }


def _from_debug_trace(rid: str):
    """Legacy fallback: downsample the DEBUG decisions_ep0.jsonl trace."""
    tr = Path(f"data/results/{rid}/decisions_ep0.jsonl")
    if not tr.exists():
        return None
    rows = [json.loads(l) for l in tr.open()]
    if not rows:
        return None
    k = max(1, len(rows) // MAX_POINTS)
    rows_ds = rows[::k]
    return {
        "steps":      [r["step"] for r in rows_ds],
        "soc":        [round(r.get("battery_soc") or 0, 4) for r in rows_ds],
        "stored":     [round(r.get("data_stored_mb") or 0, 2) for r in rows_ds],
        "downlinked": [round(r.get("data_downlinked_mb") or 0, 2) for r in rows_ds],
        "gpass":      [int(bool(r.get("ground_pass_active"))) for r in rows_ds],
        "anomaly":    [int(bool(r.get("anomaly_forced_safe"))) for r in rows_ds],
        "mode":       [r.get("mode", "?") for r in rows_ds],
    }


def main() -> None:
    out = {}
    for rid, label in SOURCES.items():
        series = _from_embedded(rid) or _from_debug_trace(rid)
        # Final fallback: the legacy dedicated single-episode telemetry_* run.
        if series is None:
            legacy = rid.replace("eventsat_sas_sda_", "telemetry_").replace("_hd_", "_hd_")
            series = _from_debug_trace(legacy)
        if series is None:
            continue
        out[rid] = {"label": label, **series}
    Path("data/figures/telemetry.json").write_text(json.dumps(out))
    print("telemetry extracted:", list(out))


if __name__ == "__main__":
    main()

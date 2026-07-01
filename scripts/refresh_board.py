"""Refresh extract.json with any results.json newer than it, then rebuild the board.
Numbers auto-refresh; validity STATUS is never auto-promoted — it flips only after
the substrate-integrity verification (deliberate)."""
import json, statistics, subprocess, sys
from pathlib import Path

EXTRACT = Path("data/figures/extract.json")
KEYS = ["utility","mean_aoi_s","peak_aoi_s","robustness_mean_recovery_steps",
        "data_downlink_efficiency","value_of_information","constraint_violation_rate",
        "commanding_effort","observation_hours","downlinked_mb",
        "operator_load","explainability_score","mean_latency_s","final_battery_soc",
        "mean_ground_latency_s",
        "anomaly_events","safety_overrides","resource_efficiency",
        "llm_api_calls","llm_cache_hits","llm_live_calls","llm_cache_hit_rate",
        "llm_total_latency_s","llm_mean_call_latency_s",
        "llm_tokens_prompt","llm_tokens_completion",
        "llm_schedule_entries",
        "planner_latency_s","orin_planner_latency_ms","planner_rollouts_per_s",
        "candidate_count","cem_iterations","model_size_mb","peak_memory_mb",
        "probe_validation_error","train_dataset_steps","policy_loaded",
        "constraint_violations_per_episode",
        "coverage_rate","successful_observations","duplicate_observation_rate",
        "mean_revisit_steps","coordination_messages"]

# Episodes excluded by substrate-integrity screening (run_id -> episode indices).
# Append-only, manual, evidence required — same policy as MEASURED status flips.
# Excluded episodes are nulled in per_ep (preserving index alignment for paired
# seeds) and the means are recomputed over the surviving episodes only.
EXCLUDED_EPISODES = {
    # run_id -> {episode indices excluded by substrate-integrity screening}.
    # Append-only, evidence required. (Cleared at the 2026-06-13 rename; the old
    # superseded runs cleared at the 2026-06-13 rename.)
}


def _mean(values):
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def _live_calls(api, cache):
    if api is None and cache is None:
        return None
    return max(0.0, float(api or 0.0) - float(cache or 0.0))


def _repair_llm_latency(d):
    """Use weighted live-call latency, not cached replay latency.

    ``llm_mean_call_latency_s`` is emitted per episode as live latency / live
    calls. Averaging that field across episodes is wrong when later episodes are
    fully cached: zeros from cached-only episodes would dilute the live-call cost.
    For the board extract, compute the experiment value from totals instead.
    """
    per = d.get("per_ep", {})
    api = per.get("llm_api_calls", [])
    cache = per.get("llm_cache_hits", [])
    total_latency = per.get("llm_total_latency_s", [])
    n = max(len(api), len(cache), len(total_latency))
    if n == 0:
        return

    live = [
        _live_calls(api[i] if i < len(api) else None, cache[i] if i < len(cache) else None)
        for i in range(n)
    ]
    per["llm_live_calls"] = live

    api_total = sum(v for v in api if v is not None)
    cache_total = sum(v for v in cache if v is not None)
    live_total = sum(v for v in live if v is not None)
    latency_total = sum(v for v in total_latency if v is not None)

    mean = d.setdefault("mean", {})
    mean["llm_live_calls"] = _mean(live)
    if api_total > 0:
        mean["llm_cache_hit_rate"] = cache_total / api_total
    if live_total > 0:
        mean["llm_mean_call_latency_s"] = latency_total / live_total
    elif api_total > 0:
        # There were LLM calls, but all came from cache: live latency is unknown.
        mean["llm_mean_call_latency_s"] = None

EXTRACT.parent.mkdir(parents=True, exist_ok=True)
if EXTRACT.exists():
    data = {d["id"]: d for d in json.loads(EXTRACT.read_text())}
else:
    data = {}
changed = 0
for rj in Path("data/results").glob("*/results.json"):
    try:
        r = json.loads(rj.read_text())
    except Exception:
        continue
    rid = r.get("experiment_id", rj.parent.name)
    eps = r.get("episodes", [])
    mean = r.get("experiment_statistics", {}).get("mean", {})
    rec = {
        "id": rid, "n": len(eps),
        "mean": {k: mean.get(k) for k in KEYS},
        "per_ep": {k: [(e.get("episode_metrics", {}).get("aggregated", {}) or {}).get(k) for e in eps]
                   for k in KEYS},
        "flag": "", "desc": r.get("description", "")}
    _repair_llm_latency(rec)
    data[rid] = rec
    changed += 1
for rid, bad in EXCLUDED_EPISODES.items():
    d = data.get(rid)
    if not d:
        continue
    before = json.dumps(d, sort_keys=True)
    per = d["per_ep"]
    for k, vals in per.items():
        per[k] = [None if i in bad else v for i, v in enumerate(vals)]
    n_total = max((len(v) for v in per.values()), default=0)
    d["n"] = sum(1 for i in range(n_total) if i not in bad)
    d["mean"] = {k: (statistics.mean(vs) if (vs := [v for v in per[k] if v is not None]) else None)
                 for k in per}
    _repair_llm_latency(d)
    d["flag"] = f"episodes {sorted(bad)} excluded by substrate screening"
    if json.dumps(d, sort_keys=True) != before:
        changed += 1
# Prune entries whose result directory no longer exists (removed / renamed runs)
# so the board never resolves stale ids.
for rid in [k for k in data if not (Path("data/results") / k / "results.json").exists()]:
    del data[rid]
    changed += 1
if changed:
    EXTRACT.write_text(json.dumps(list(data.values())))
subprocess.run([sys.executable, "scripts/extract_telemetry.py"], check=True)
subprocess.run([sys.executable, "scripts/build_results_board.py"], check=True)
subprocess.run([sys.executable, "scripts/build_wm_board.py"], check=True)
subprocess.run([sys.executable, "scripts/build_index.py"], check=True)
print(f"refreshed {changed} experiment(s)")

# Metrics Definitions and Rationale

**Status (2026-03-17):** All 7 core research metrics implemented in `EventSatMetricsCollector`: Utility, Data Downlink Efficiency, Latency, Robustness, Resource Efficiency, Operator Load, Explainability. Only remaining gap: Scale & Complexity (requires multi-constellation experiments). Physics fidelity updated: J2 propagator, launch lottery Monte Carlo, and environment-enforced anomaly safe mode added (see `docs/scenarios.md`).

---

## Overview

The metrics framework is designed to capture multiple performance dimensions for fair comparison of cognitive architectures. All metrics are collected uniformly regardless of which architecture variant is being evaluated.

## Core Metrics

### 1. Utility

**Definition:** Total value achieved from completed tasks/objectives.

**Rationale:** Primary performance metric — does the system accomplish its mission?

**Measurement:** Scenario-dependent reward function (TBD upon scenario selection).

**Open Questions:**
- How to define utility for each operational scenario?
- Single scalar vs. multi-objective utility?
- Time-discounting of future utility?

---

### 2. Latency

**Definition:** Decision-making computational time.

**Rationale:** Real-time constraints in space operations — decisions must be timely.

**Measurement:** Wall-clock time per decision cycle (`time.perf_counter()` around `DecisionLoop.process()`).

**Considerations:**
- Symbolic representations are typically faster than LLM-based ones.
- Neural representations have inference-time vs. training-time trade-offs.
- Must account for communication overhead in distributed organizations.

---

### 3. Robustness

**Definition:** Consistency of mission performance across varying initial conditions (orbit insertion geometry, launch lottery RAAN/ArgP/TA randomisation).

**Rationale:** An architecture that achieves high mean utility but whose results vary widely across episodes depending on the specific orbit geometry is not robust. The EventSat launch lottery explicitly randomises insertion parameters per episode (simulating rideshare uncertainty) so that Monte Carlo averaging over episodes captures this variability. An architecture that delivers reliably similar utility regardless of orbit geometry is truly robust.

**Measurement:** Coefficient of variation of utility across episodes:

> `robustness_cv = std(utility_per_episode) / mean(utility_per_episode)`

Lower CV = more consistent = better robustness. Computed at experiment level from all episode utility values.

**Secondary diagnostic:** `robustness_mean_recovery_steps` tracks the within-episode steps from anomaly onset to recovery, providing a complementary measure of fault-recovery behaviour.

**Implementation:** `robustness_cv` computed in `EventSatMetricsCollector.compute_statistics()`. Can also be derived directly from episode DataFrames as `episode_df.groupby("experiment_id")["utility"].agg(lambda x: x.std()/x.mean())`.

---

### 4. Resource Efficiency

**Definition:** Achieved utility per unit resource consumed.

**Rationale:** Satellites have limited power, data bandwidth, and computation.

**Measurement:** `utility / total_resources_consumed` (resource model is scenario-dependent).

**Open Questions:**
- Which resources to track? (power, bandwidth, memory, computation)
- How to normalise across different constellation sizes?

---

### 5. Operator Load

**Definition:** Required human intervention frequency.

**Rationale:** Autonomy goal is reducing operator burden.

**Measurement (candidates):**
- Count of constraint violations requiring human override.
- Count of failed actions that need manual recovery.
- Number of decision cycles where the agent defers to a human.

**Open Questions:**
- How to define "intervention" precisely?
- Is operator load purely count-based or does severity matter?

---

### 6. Scale & Complexity

**Definition:** Performance degradation across a 2D scalability space: *constellation size* (number of satellites) × *structural complexity* (topology, from centralized to fully distributed).

**Rationale:** RQ3 frames scalability along two independent axes. Raw satellite count captures the size dimension; structural complexity captures the coordination overhead that grows super-linearly as topology moves from centralized towards distributed (following Sinha & de Weck, 2013). Different architecture families may degrade differently along each axis.

**Measurement:**
- Track all metrics as a function of `constellation_size` (1 → 500) and a `complexity_index` encoding topology class (0 = centralized, 1 = hierarchical, 2 = fully distributed).
- Fit joint scaling surfaces (e.g., power-law or polynomial) over the 2D grid.
- Derive architecture-selection heuristics from these surfaces for a target (size, complexity) operating point.

**Analysis:**
- 2D heatmaps of metric degradation over (size × complexity).
- Identify scaling bottlenecks per architecture family.
- Test whether composability limits are hit at specific (size, complexity) combinations.

---

### 7. Explainability

**Definition:** The degree to which an architecture's decisions can be interpreted and justified to human operators.

**Rationale:** Mission safety and human-machine trust require that operators understand *why* the system acted as it did, not just *what* it did. RQ2 explicitly asks how architecture choice determines the type and degree of explainability available. Architectures differ fundamentally: symbolic representations are inherently interpretable; neural/emergent representations require post-hoc methods.

**Measurement (candidates — require theoretical development):**
- Presence and completeness of decision traces or reasoning logs (binary or graded).
- Human-evaluable justification rate: fraction of decisions accompanied by an accessible, operator-readable rationale.
- Compliance with operator-interpretable rules (for symbolic architectures).
- For neural architectures: attention visualization coverage, SHAP value availability.

**Considerations:**
- The metric must be architecture-agnostic in *collection* but architecture-sensitive in *interpretation*.
- Link to operator load: higher explainability may reduce the intervention frequency required.
- Safety implications: some scenarios (e.g., collision avoidance in SSA) may require a minimum explainability threshold as a hard constraint.

**Open Questions:**
- How to define a common explainability scale across fundamentally different architecture families?
- Is explainability a continuous metric or a categorical one (none / partial / full)?
- What level of explainability satisfies mission safety requirements for the chosen scenarios?

---

## Statistical Analysis

### Per-Experiment Statistics
- Mean, standard deviation, min, max across episodes.
- Confidence intervals (95% CI).

### Cross-Experiment Comparison
- Paired statistical tests (Wilcoxon signed-rank for non-normal distributions).
- Effect size measures (Cohen's d or equivalent).
- Pareto frontier analysis for multi-metric trade-offs.

### Pareto Frontier
The framework includes a Pareto frontier computation (`src/orchestration/analysis.py`) to identify architectures that represent optimal trade-offs between metrics (e.g., utility vs. latency).

---

## Implementation Notes

The `MetricsCollector` abstract class in `src/orchestration/metrics_collector.py` defines the collection pipeline:
1. `collect_step_metrics()` — per-timestep collection.
2. `aggregate_episode_metrics()` — episode-level aggregation.
3. `compute_statistics()` — cross-episode statistics.

### EventSat Metrics (Implemented)

The `EventSatMetricsCollector` (`src/orchestration/eventsat_metrics.py`) collects per-step and per-episode metrics for the EventSat scenario.

**Per-step metrics:**
- `battery_soc`, `data_stored_mb`, `data_downlinked_mb`, `observation_hours`
- `jetson_raw_mb`, `jetson_compressed_mb`, `obc_data_mb` — 3-pool pipeline telemetry
- `total_detections`, `max_achievable_downlink_mb` — detection + pipeline capacity
- `in_sunlight`, `ground_pass_active`, `in_transition`
- `forced`, `anomaly`, `anomaly_forced_safe`, `safety_override`
- `energy_consumed_wh` — from SoC delta × battery capacity
- `decision_latency_s`, `has_rationale`

**Episode-level aggregates (research metrics):**
- `utility` — mission objective achievement: `w_obs × obs_ratio + w_dl × dl_ratio − w_anomaly × anomaly_rate`
  Targets scaled from 90-day mission to episode length (7-day default).
- `data_downlink_efficiency` — `data_downlinked_mb / max_achievable_downlink_mb`
  Measures how effectively available ground contact time was used for downlink.
  Source: Proposal Section 6.1 — "useful observation time is limited by downlink capacity"
- `mean_latency_s`, `max_latency_s` — decision loop timing
- `robustness_mean_recovery_steps` — steps to recover from anomaly events (onset to clearance — via ground pass for conventional ops, via onboard FDIR for autonomous ops)
- `resource_efficiency` — `utility / total_energy_consumed_wh`
- `operator_load` — fraction of steps with safety overrides
- `explainability_score` — fraction of steps with a rationale string

**Other episode-level telemetry:**
- `observation_hours`, `downlinked_mb`, `final_battery_soc`, `total_energy_consumed_wh`
- `safety_overrides`, `anomaly_events`, `total_detections`, `max_achievable_downlink_mb`

---

## Implementation Gap Analysis

> **Status as of 2026-03-16:** All 7 core research metrics are operationalised in `EventSatMetricsCollector` (7/8 priorities done). Only remaining: Scale & Complexity (Priority 7) requires multi-constellation experiments.

### Gap Summary

| Metric | Spec requirement | Current status |
|---|---|---|
| Utility | Scalar mission value from objectives | ✅ Implemented: `w_obs × obs_ratio + w_dl × dl_ratio − anomaly_penalty` |
| Data Downlink Efficiency | Fraction of downlink capacity used | ✅ Implemented: `downlinked_mb / max_achievable_downlink_mb` (Proposal Section 6.1) |
| Latency | Wall-clock time per decision cycle | ✅ Collected from `decision_metrics` passed by experiment runner |
| Robustness | Consistency across varying initial conditions (orbit geometry) | ✅ `robustness_cv = std(utility)/mean(utility)` across episodes; `robustness_mean_recovery_steps` retained as secondary diagnostic |
| Resource Efficiency | Utility per unit resource consumed | ✅ `utility / total_energy_consumed_wh` |
| Operator Load | Human intervention frequency | ✅ `safety_overrides / n_steps` (environment safety overrides as proxy); `anomaly_forced_safe` distinguishes FDIR-forced safe from voluntary safe |
| Scale & Complexity | Metrics vs. constellation_size × complexity_index | ⚠️ Metadata recorded; 2D scaling surface requires multi-constellation experiments |
| Explainability | Decision trace completeness | ✅ `decisions_with_rationale / n_steps` (`rule_based_eventsat` yields 1.0) |

---

## Required Implementation Work

### ~~Priority 1 — Fix schema mismatch~~ ✅ Done

`EventSatMetricsCollector` uses `StepMetrics` / `EpisodeMetrics` / `ExperimentStatistics` from the base class.

### ~~Priority 2 — Instrument latency measurement~~ ✅ Done

`decision_latency_s` collected from `decision_metrics` passed by experiment runner.

### ~~Priority 3 — Define and compute Utility~~ ✅ Done

```
Utility = w_obs × (obs_hours / scaled_obs_target)
        + w_dl  × (dl_mb / scaled_dl_target)
        - w_anomaly × anomaly_rate
```

Weights configurable via experiment YAML (`utility_weights`). Targets scaled from 90-day mission to episode length.

### ~~Priority 4 — Compute Resource Efficiency~~ ✅ Done

`resource_efficiency = utility / total_energy_consumed_wh`. Energy from SoC delta × battery capacity.

### ~~Priority 5 — Define Operator Load~~ ✅ Done

`operator_load = safety_overrides / n_steps` (environment safety overrides as proxy).

### ~~Priority 6 — Add Robustness measurement~~ ✅ Done

- Within-episode: `robustness_mean_recovery_steps` from anomaly onset/recovery tracking.
- Cross-episode: `robustness_cv = std(utility) / mean(utility)` in `compute_statistics()`.

### Priority 7 — Record Scale & Complexity metadata

Every `ExperimentStatistics` result must record:
- `constellation_size` — from experiment config
- `complexity_index` — 0 (centralized), 1 (hierarchical), 2 (distributed), derived from `agent_organization` config field
- All research metrics stored indexed by `(constellation_size, complexity_index)` to enable the 2D scaling surface analysis from RQ3

Status: ⚠️ Metadata recorded; 2D scaling surface requires multi-constellation experiments (Flamingo scenario).

### ~~Priority 8 — Explainability instrumentation~~ ✅ Done

`explainability_score = decisions_with_rationale / n_steps`. Symbolic (`rule_based_eventsat`) yields 1.0; neural/hybrid will yield lower values.

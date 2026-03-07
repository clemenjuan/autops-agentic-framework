# Metrics Definitions and Rationale

**Status:** Theoretical framework defined. EventSat-specific metrics implemented in `src/orchestration/eventsat_metrics.py`.

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

**Definition:** Performance stability under perturbations and uncertainty.

**Rationale:** Space environment is unpredictable — architectures must handle failures gracefully.

**Measurement (candidates — require theoretical development):**
- Variance of utility across episodes with perturbation injection.
- Recovery time after simulated failures.
- Graceful degradation curve as failure rate increases.

**Open Questions:**
- What types of perturbations to inject? (satellite failures, communication drops, unexpected tasks)
- How to separate inherent stochasticity from robustness?

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

The `EventSatMetricsCollector` (`src/orchestration/eventsat_metrics.py`) provides the first concrete implementation, collecting per-step and per-episode metrics:

**Per-step metrics:**
- `reward` — step reward from the environment
- `battery_soc` — battery state of charge
- `data_stored_mb` — onboard data storage usage
- `data_downlinked_mb` — cumulative downlinked data
- `in_sunlight` — whether satellite is in sunlight
- `ground_pass_active` — whether a ground pass is active
- `forced_mode` — whether the requested mode was overridden
- `anomaly` — whether an anomaly event occurred
- `observation_hours` — cumulative observation time

**Episode-level aggregates:**
- `episode_reward` — total reward
- `total_observation_hours` — total science observation time
- `total_downlinked_mb` — total data successfully downlinked
- Mean/min/max battery SoC across the episode

# Metrics Definitions and Rationale

**Status:** Theoretical framework — specific operationalisations require deeper study.

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

### 6. Scalability

**Definition:** Performance degradation as constellation size increases.

**Rationale:** Research question RQ3 directly addresses scaling behaviour.

**Measurement:** Track all other metrics as a function of `constellation_size`.

**Analysis:**
- Plot metric vs. constellation size curves.
- Fit scaling laws (linear, polynomial, exponential degradation).
- Identify scaling bottlenecks per architecture.

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

Concrete implementations will be created when the operational scenario is selected and metric formulas are theoretically justified.

# Operational Scenarios

**Status:** EventSat implemented. Flamingo and Space Data Centers planned for later phases.

---

## Overview

Three concrete scenarios have been selected to cover the 2D scalability space of RQ3: *constellation size* × *structural complexity*. Each scenario is implemented sequentially, building on the environment abstraction defined in `src/environment/`.

The scenario choice does **not** affect the cognitive architecture comparison methodology — the same morphological matrix dimensions are evaluated across all scenarios.

---

## Scenario 1: EventSat Mission

**Phase:** 2 (starting point) | **Scale:** 1 satellite | **Complexity:** minimal (single-agent) | **Status: Implemented**

### Description

TUM's own EventSat satellite. Full access to subsystem models and mission data enables high-fidelity environment modeling. This is the baseline scenario where the full cognitive architecture comparison begins before growing in scale.

### Tasks

- Observation scheduling (target selection, pointing, exposure management).
- Onboard resource management: power (duty cycles), data storage, thermal budget.
- Anomaly detection and autonomous response to off-nominal states.
- Downlink scheduling to ground stations.

### Constraints

- Orbit-dependent visibility windows to targets and ground stations.
- Power budget (solar charging vs. consumption duty cycle).
- Onboard storage capacity and downlink data rate.
- Thermal limits on onboard processing.

### Metrics Emphasis

- Utility: observation task completion and quality.
- Resource efficiency: power and data budget usage.
- Decision latency: onboard compute constraints are real.
- Explainability: ground operators must understand autonomous decisions.

### Data Sources

Internal TUM/AUTOPS mission data — high confidence in accurate subsystem modeling.

### Implementation

**Environment:** `src/environment/scenarios/eventsat_env.py`
**Scenario config:** `configs/scenarios/eventsat.yaml`
**Experiment config:** `configs/experiments/eventsat_baseline.yaml`
**Metrics collector:** `src/orchestration/eventsat_metrics.py`
**Representation:** `src/representation/rule_based_eventsat.py`
**Decision loop:** `src/decision_loop/sda_loop.py`

### Satellite Modes

The environment models five operational modes: `charging`, `communication`, `payload_observe`, `payload_compress`, `safe`. Mode transitions are constrained by battery state-of-charge, ground pass availability, and anomaly status.

### Orbital Mechanics

Eclipse intervals and ground station passes are pre-computed at episode reset via `src/environment/orbital/context.py`. Two backends are available:
- **Simplified** (default): Phase-fraction eclipse model, stochastic pass generation.
- **Orekit** (optional, install with `uv sync --extra orbital`): Geometric shadow computation, elevation-based pass detection.

### Operations Paradigm Integration

The EventSat baseline runs with `autonomous_hybrid` (agent has full real-time state, acts every step). The `conventional_ground` paradigm can be used to simulate traditional ground operations with stale telemetry and uplink-gated commanding during passes only.

---

## Scenario 2: Vyoma Flamingo Constellation

**Phase:** 3 | **Scale:** up to 12 satellites | **Complexity:** medium (multi-agent, hierarchical/distributed topologies)

### Description

Vyoma's Flamingo constellation — AUTOPS project partners. Planned to reach 12 satellites total, making it a natural medium-scale use case. The AUTOPS collaboration provides access to realistic mission parameters and operational requirements. This scenario is the primary arena for comparing agent organization variants (centralized vs. hierarchical vs. distributed) under RQ2.

### Mission Domain

Space Situational Awareness (SSA): tracking resident space objects (RSOs), coverage optimization across the constellation, data fusion.

### Tasks

- Distributed observation scheduling: which satellite tracks which RSO.
- Coverage optimization: maximizing revisit frequency across the RSO catalog.
- Inter-satellite data fusion and coordination.
- Handoff management when an RSO moves between satellite coverage zones.

### Constraints

- Inter-satellite link (ISL) availability and bandwidth.
- Individual sensor FOV and pointing constraints.
- Revisit time requirements per RSO priority class.
- Per-satellite power and downlink budgets.
- Communication latency between satellites.

### Metrics Emphasis

- Utility: target coverage rate and detection probability.
- Robustness: performance under satellite failures or link drops.
- Operator load: coordination complexity vs. autonomy benefit.
- Scale & complexity: how metrics evolve from 3 → 6 → 12 satellites.
- Explainability: multi-agent decisions are harder to trace — key RQ2 question.

### Data Sources

AUTOPS project collaboration with Vyoma — high confidence in obtaining useful modeling data.

### Implementation

**File:** `src/environment/scenarios/flamingo.py`
**Config:** `configs/scenarios/flamingo.yaml`

---

## Scenario 3: Space-Based Data Centers

**Phase:** 3–4 | **Scale:** 100+ satellites | **Complexity:** high (fully distributed, heterogeneous)

### Description

An emerging large-constellation concept for orbital computation — satellites hosting onboard data centers that process jobs offloaded from ground. Represents the high-scale, high-complexity endpoint of the scalability study. Less mission-specific data is available at this stage; the scenario will be modeled using published literature and synthetic parameters, with fidelity to be refined as the field matures.

This scenario is the primary arena for RQ3 scaling law derivation and composability limit analysis.

### Tasks

- Computational job scheduling across a large heterogeneous fleet.
- Thermal management: enforcing duty cycle constraints to avoid overheating.
- Inter-satellite job migration: relocating running jobs to better-positioned satellites.
- Resource allocation: CPU, memory, power across hundreds of nodes.
- Ground-facing load balancing: routing requests to satellites with favourable geometry.

### Constraints

- Power budget and thermal limits per satellite.
- ISL bandwidth for job data transfer.
- Orbital position determines latency to ground clients.
- Job deadline and QoS requirements.
- Fleet heterogeneity (different hardware generations).

### Metrics Emphasis

- Utility: job completion rate and QoS compliance.
- Resource efficiency: energy-per-job, utilization rate.
- Scale & complexity: this scenario is the stress test — scaling laws derived here.
- Composability: integrating heterogeneous cognitive components without emergent negative side effects.
- Explainability: at large scale, post-hoc interpretability methods become critical.

### Data Sources

Literature-based modeling (lower fidelity than Scenarios 1–2). Key references: published space data center architecture proposals, cloud scheduling literature adapted to orbital constraints.

### Implementation

**File:** `src/environment/scenarios/space_data_centers.py`
**Config:** `configs/scenarios/space_data_centers.yaml`

---

## Progression Strategy

| Scenario             | Phase | Satellites | Complexity Index | Primary RQ  |
|----------------------|-------|-----------|-----------------|-------------|
| EventSat             | 2     | 1         | 0 (centralized) | RQ1         |
| Flamingo             | 3     | 3–12      | 1–2 (hier./dist.) | RQ1, RQ2  |
| Space Data Centers   | 3–4   | 50–500    | 2 (distributed) | RQ2, RQ3   |

The three scenarios together sweep the 2D scalability space, enabling joint scaling law derivation over constellation size × structural complexity for RQ3.

---

## Implementation Steps (per scenario)

1. Define the environment subclass in `src/environment/scenarios/`.
2. Define scenario-specific task types and constraints.
3. Define the reward function (maps to utility metric).
4. Define the resource model.
5. Create scenario configuration in `configs/scenarios/`.
6. Write comprehensive tests in `tests/`.

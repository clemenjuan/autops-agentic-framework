# Operational Scenarios

**Status:** Awaiting researcher decision — select one scenario for Phase 2.

---

## Overview

The satellite environment requires a concrete operational scenario that defines mission objectives, task types, constraints, and reward structures. The choice of scenario shapes much of the experimental framework but does **not** affect the cognitive architecture comparison methodology.

## Candidate Scenarios

### Option 1: Space-Based Data Centers

**Tasks:**
- Computational job scheduling across satellites.
- Thermal management (duty cycle constraints).
- Resource allocation (CPU, memory, power).

**Constraints:**
- Power budget per satellite.
- Thermal limits on computation.
- Inter-satellite link bandwidth.
- Orbital position affects latency to ground.

**Metrics:**
- Job completion rate (utility).
- Energy efficiency (resource efficiency).
- Job queue wait time (latency).

**Relevance:** Emerging concept in space industry; computationally rich scenario.

---

### Option 2: Communications Constellation

**Tasks:**
- Ground contact scheduling.
- Data routing through constellation.
- Handoff coordination between satellites.

**Constraints:**
- RF bandwidth per satellite.
- Visibility windows to ground stations.
- End-to-end latency requirements.
- Inter-satellite link availability.

**Metrics:**
- Data throughput (utility).
- Communication latency (latency).
- Coverage continuity (robustness).

**Relevance:** Well-studied domain; rich literature for validation.

---

### Option 3: Space Situational Awareness (SSA)

**Tasks:**
- Observation scheduling (which RSOs to track).
- Sensor tasking (pointing, exposure settings).
- Anomaly detection and alerting.

**Constraints:**
- Sensor field of view (FOV).
- Power budget for sensor operations.
- Data downlink capacity.
- Revisit time requirements.

**Metrics:**
- Target coverage rate (utility).
- Detection probability (utility component).
- Revisit time statistics (robustness).

**Relevance:** Directly aligns with AUTOPS project; strong SSA heritage at TUM.

---

## Decision Criteria

Select the scenario based on:

1. **AUTOPS project relevance** — which scenario best supports the project goals?
2. **Data availability** — can we obtain or generate realistic scenario data?
3. **Complexity appropriateness** — is the scenario rich enough but tractable for PhD scope?
4. **Supervisor input** — Vincenzo's research perspective.

## Implementation Steps (after selection)

1. Define the environment subclass in `src/environment/scenarios/`.
2. Define scenario-specific task types and constraints.
3. Define the reward function (maps to utility metric).
4. Define the resource model.
5. Create scenario configuration in `configs/scenarios/`.
6. Write comprehensive tests.

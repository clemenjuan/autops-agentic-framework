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

- Mode selection: choose among 7 operational modes each timestep (60s).
- Observation scheduling: when to record event camera data.
- Data pipeline management: compression → detection → RS-485 send → downlink.
- Power management: balance observation/processing against battery SoC.
- Anomaly detection and autonomous response to off-nominal states.
- Downlink scheduling: exploit limited ground passes (22–422s, 2–3/day).

### Constraints

- Orbit-dependent visibility windows to targets and ground stations (400 km SSO).
- Power budget (solar charging vs. consumption duty cycle; 84 Wh battery, 24 W solar). Jetson-based onboard cores (subsymbolic/hybrid onboard, AO/AH) add a ~7 W Jetson-on draw (`power.onboard_compute_w`) to modes where the Jetson would otherwise be off (charging/communication/safe), but not to the Jetson-on payload modes (observe/compress/detect/send already include the powered Jetson); symbolic onboard (OBC rules) and ground paradigms (AG/CG) have no such overhead.
- 3-pool data pipeline with RS-485 bottleneck (50 kbps Jetson→OBC).
- Daily downlink budget (27 MB with GSaaS; configurable).
- Multi-step processing: compression (2× obs time), detection (5 min), send (rate-limited).
- Thermal model removed (heat dissipation design in progress; not a current constraint).

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
**Experiment configs:** `configs/experiments/eventsat_sas_symbolic_ah.yaml` (autonomous hybrid), `configs/experiments/eventsat_sas_symbolic_cg.yaml` (conventional ground)
**Metrics collector:** `src/orchestration/eventsat_metrics.py`
**Representation:** `src/representation/rule_based_eventsat.py`
**Decision loop:** `src/decision_procedure/sda_loop.py`

### Satellite Modes

Seven operational modes (from PDR Chapter 3 & Table 3.1):

| Mode | Description | Constraint |
|------|-------------|------------|
| `charging` | Sun-pointing, payload off (default) | — |
| `communication` | S-band + UHF downlink from OBC | Ground pass active |
| `payload_observe` | Event camera recording → Jetson raw | SoC > 0.4 |
| `payload_compress` | Jetson compresses raw data (2× obs time) | SoC > 0.3 |
| `payload_detect` | Jetson CV inference on compressed data (5 min) | SoC > 0.3 |
| `payload_send` | RS-485 Jetson→OBC transfer (50 kbps, one-way) | SoC > 0.3 |
| `safe` | UHF only; entered via FDIR on anomaly | — |

Mode transitions to `payload_observe` or `communication` incur 135s ADCS attitude settling overhead (P2, from ADCS thesis).

### Data Pipeline

3-pool pipeline (Jetson raw → Jetson compressed → OBC → S-band downlink):

```
observe → jetson_raw_mb (9.41 MB/obs)
compress → jetson_compressed_mb (1.84 MB/obs, 5.11:1 ratio)
detect  → reads compressed data, produces metadata (0.01 MB → OBC)
send    → jetson_compressed_mb → obc_data_mb (RS-485, 50 kbps = 0.375 MB/step)
comm    → obc_data_mb → downlinked (S-band, 128 kbps)
```

Pipeline backpressure: agent stops observing when `obc_mb + jetson_compressed_mb > daily_downlink_budget_mb` (27 MB/day with GSaaS, configurable). Source: Proposal Section 6.1 — "useful observation time is limited by downlink capacity."

### Orbital Mechanics

Eclipse intervals and ground station passes are pre-computed at episode reset via `src/environment/orbital/context.py`. Two backends are available:
- **Orekit** (default when installed, `uv sync --extra orbital`): J2-perturbed propagation via EcksteinHechler, geometric shadow computation, elevation-based pass detection at Ottobrunn (48.05°N, 11.66°E, min elevation 10°). Ground passes are fully deterministic from orbital mechanics — no stochastic component.
- **Simplified** (fallback when Orekit is not installed): Phase-fraction eclipse model, stochastic pass generation. A warning is logged when this fallback is used in production.

#### J2 Propagator

The `EcksteinHechlerPropagator` (Orekit analytical) is used instead of two-body Keplerian propagation. This models J2 secular perturbations, which drive RAAN precession of ~1.04°/day at 400 km, 97.4° inclination — the mechanism that maintains SSO sun-synchronous geometry. Without J2, RAAN would remain fixed and lighting conditions would be unrealistic over multi-day simulations.

Propagator selection is controlled by `orbit.propagator` in `configs/scenarios/eventsat.yaml` (`j2` or `keplerian`).

#### Launch Lottery (Monte Carlo)

EventSat is a rideshare mission with no onboard propulsion. The exact RAAN and argument of perigee at deployment depend on the launch vehicle and other payloads — a "launch lottery." To capture this uncertainty in Monte Carlo experiments:

- At each episode `reset(seed)`, RAAN, Argument of Perigee, and True Anomaly are drawn uniformly from [0°, 360°) using the episode seed.
- Altitude (400 km), inclination (97.4°), and eccentricity (0.001) remain fixed.
- Post-deployment, these parameters evolve only under natural perturbations (J2 RAAN drift modelled; drag/SRP not yet included).
- Controlled by `orbit.launch_lottery: true` in `configs/scenarios/eventsat.yaml`.

The seed contract is: draws 1–3 of the seeded RNG stream are RAAN/ArgP/TA; subsequent draws are anomaly injection. This ordering ensures full reproducibility per seed.

#### Anomaly Handling

Anomalies are injected stochastically (`anomaly_prob = 0.001` per step) and represent FDIR-level events (currently `"thermal_warning"`).

**Environment-enforced safe mode:** When an anomaly is active, `_resolve_mode()` returns `"safe"` regardless of the agent's request. The agent cannot override this.

**Recovery:** After the minimum forced duration (3–10 steps), recovery depends on the operations paradigm:
- **Conventional Ground:** anomaly clears only when a ground pass is active — simulating the flight controller sending a resume command during the next overpass. The satellite stays in safe mode until the next contact window.
- **Autonomous Hybrid:** onboard FDIR clears the anomaly as soon as the countdown expires, without waiting for ground contact.

Step info includes `anomaly_forced_safe: float` (1.0 during active anomaly, 0.0 otherwise) to distinguish environment-forced safe from agent-initiated safe in post-run analysis.

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

## Scenario 3: Large-Scale Constellation

**Phase:** 3–4 | **Scale:** 100+ satellites | **Complexity:** high (fully distributed, heterogeneous)

### Description

A large-scale constellation scenario at the high end of the scalability spectrum. Represents the high-scale, high-complexity endpoint of the scalability study. The specific mission type will be determined during Phase 2. Less mission-specific data is available at this stage; the scenario will be modeled using published literature and synthetic parameters, with fidelity to be refined as the field matures.

This scenario is the primary arena for RQ3 scaling law derivation and composability limit analysis.

### Tasks

To be defined based on selected mission type (candidates include resource scheduling, coordination, coverage optimisation).

### Constraints

- Power budget per satellite.
- ISL bandwidth.
- Orbital position determines latency to ground.
- Fleet heterogeneity (different hardware generations).

### Metrics Emphasis

- Mission utility, resource efficiency, coordination overhead, explainability.
- Scale & complexity: this scenario is the stress test for scaling law derivation.

### Data Sources

Literature-based modeling (lower fidelity than Scenarios 1–2); to be refined as the field matures.

### Implementation

**File:** TBD (Phase 2)
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

# Operational Scenarios

**Status:** EventSat implemented (benchmark scenario); MultiEventsat implemented as the EventSat-compatible multi-satellite reference; SSA implemented as the constellation-scale organisation scenario.

---

## Overview

EventSat is the single-satellite benchmark scenario. MultiEventsat preserves the EventSat per-satellite contract at constellation scale for RLlib and organisation plumbing. SSA adds resident-space-object (RSO) observation, inter-satellite sharing, and delivered-to-ground collective utility so the organisation axis is exercised on a real constellation task. Each scenario builds on the environment abstraction in `src/core/satellite_env.py`.

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

**Environment:** `src/eventsat/env.py`
**Scenario config:** `configs/scenarios/eventsat.yaml`
**Experiment configs:** `configs/experiments/eventsat_sas_ah_symb_symb.yaml` (autonomous hybrid, symbolic both cores), `configs/experiments/eventsat_sas_conventional_symb.yaml` (conventional, symbolic)
**Metrics collector:** `src/eventsat/metrics.py`
**Representation:** `src/eventsat/symbolic.py`
**Decision loop:** `src/core/decision_procedure/sda_loop.py`

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

Eclipse intervals and ground station passes are pre-computed at episode reset via `src/orbital/context.py`. Two backends are available:
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

## Scenario 2: MultiEventsat Reference

**Status:** implemented | **Scale:** configurable N satellites | **Primary use:** multi-agent RL / RLlib bridge validation

MultiEventsat composes N EventSat-class satellites (`sat_0` ... `sat_{N-1}`) inside one integrated environment. Each satellite keeps the EventSat power, data-pipeline, anomaly, reward, and 25D RL observation contract, while the environment exposes per-satellite `SatelliteState` and reward dictionaries for the multi-agent bridge.

### Implementation

**Environment:** `src/eventsat/multieventsat_env.py`
**Scenario config:** `configs/scenarios/multieventsat.yaml`
**Example experiment:** `configs/experiments/multieventsat_imas_sda_subm_le_ah.yaml`
**Reward blend:** `MultiEventsatRewardFunction` in `src/eventsat/rewards.py`
**RLlib bridge:** `src/rl/rllib_env.py`

### Design Notes

- Satellites share one integrated environment step and expose `sat_i` ids, but each satellite retains EventSat-compatible telemetry and local reward fields.
- `IndependentMAS` maps `sat_agent_i` to `sat_i` and scopes each observation to that satellite only.
- The reward function can mix local reward with a team term via `local_weight`, `team_weight`, and `team_reducer` (`mean`, `sum`, or `min`).
- This reference scenario is deliberately minimal: no RSO catalog, no ISL, and no ground archive. Those live in SSA.

---

## Scenario 3: SSA Constellation

**Status:** implemented | **Scale:** N = 3 and N = 5 committed AO slice | **Primary use:** organisation axis and M-10 scale efficiency

SSA is "EventSat at constellation scale + inter-satellite links + collective RSO observation-sharing + the organisation axis." It subclasses `MultiEventsatEnv`, keeps the EventSat physical backbone per satellite, and adds a fixed RSO catalog, anti-nadir optical access, ISL knowledge sharing, onboard best-estimate state, and a ground archive that defines delivered mission utility.

### Tasks

- Select one of eight modes per satellite: the seven EventSat modes plus `isl_share`.
- Detect every RSO inside the anti-nadir +/-5 degree FOV during `payload_observe`; actions do not carry `target_id`.
- Maintain a fixed N x M binary detection matrix for onboard knowledge.
- Keep the single best onboard estimate per object while archiving every downlinked record on the ground.
- Share estimates over feasible ISLs by OR-merging the matrix and retaining the higher-quality estimate.
- Maximise delivered-to-ground RSO coverage under Collective-Negative mission utility.

### Physics And Data

- RSO catalogs are generated from seeded randomized SSO orbital elements or supplied as fixed positions for cheap smoke runs.
- Optical range follows the AUTOPS-RL optic payload equation `D_max = a*d/(2.44*lambda)`: 52.7 km for a = 1 m, d = 0.09 m, lambda = 700 nm.
- ISL feasibility ports the AUTOPS-RL UHF/QPSK link budget: free-space loss -> received power -> SNR -> ideal rate -> BER -> effective rate.
- Propagation uses `src/orbital/propagator.py` / Orekit when available, with deterministic fallback for tests.

### Metrics

SSA keeps the EventSat metrics and adds:

- `ssa_onboard_coverage` and `ssa_delivered_coverage`.
- `duplicate_observation_rate` for wasted repeated detections.
- `mean_revisit_steps` over observed objects.
- `isl_connectivity` for successful ISL shares / attempts.
- M-10 `eta_scale = (utility / N) / baseline_utility_n1`, where SSA utility is delivered RSO coverage.

### Organisation And Matrix

SSA uses naming `ssa_<org>_<paradigm>_<rep>_n<N>` and AH names both cores onboard-first: `ssa_<org>_ah_<onboard>_<ground>_n<N>`.

The committed in-scope generator is `scripts/generate_ssa_configs.py`, which emits the AO backbone:

- `{ao_symb, ao_rl}` x `{sas, cmas, dmas, imas, hmas}` x `N in {3,5}` = 20 configs.
- RL configs are `rl_mock: true` for run-time smoke checks; PPO training remains owner-gated.
- Ground paradigms AG/CG are valid for SSA only with SAS or CMAS. Live LLM ground cells, world-model cells, and N > 5 are owner-gated.

### Implementation

**Environment:** `src/ssa/env.py`
**Targets / optical access:** `src/ssa/targets.py`
**ISL link budget:** `src/orbital/isl.py`
**Scenario config:** `configs/scenarios/ssa.yaml`
**Config generator:** `scripts/generate_ssa_configs.py`
**Symbolic representation:** `src/ssa/symbolic.py` registered as `rule_based_ssa`
**Rewards / metrics:** `src/ssa/rewards.py`, `src/ssa/metrics.py`

---

## Implementation Steps (per scenario)

1. Define the environment subclass in the scenario-owned package (`src/eventsat/`, `src/ssa/`, or a new scenario package).
2. Define scenario-specific task types and constraints.
3. Define the reward function (maps to utility metric).
4. Define the resource model.
5. Create scenario configuration in `configs/scenarios/`.
6. Write comprehensive tests in `tests/`.

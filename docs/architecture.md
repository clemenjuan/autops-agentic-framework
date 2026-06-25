# System Architecture

**Project:** Cognitive Satellite Constellation Autonomy Experimental Framework
**Researcher:** Clemente J. Juan Oliver, TUM Chair of Spacecraft Systems

---

## Overview

This framework is the **engine** for the EventSat **operations-system (O) benchmark**. An architecture is organisation × representation (cognitive substrate × action space) × operational paradigm. Full definition: [`morphological_matrix.md`](morphological_matrix.md). This document covers only the *system* view — data flow, component interactions, and directory layout.

## Architecture Diagram

![AUTOPS overall system architecture diagram](images/autops-overall-system-architecture.svg)

## Layered View (Parallel to the Matrix)

Bhati (2026) [Z5TF79HY] proposes a six-layer reference architecture for **agentic
software engineering** systems. The autops framework is positioned as a **parallel
reference architecture** in a sibling domain (autonomous satellite operations), not
a structural adoption. The layered view below is complementary to the
morphological matrix (which remains canonical) and to the diagram above; it makes
the autops architectural choices legible to the broader agentic-AI literature. For
the framing see [`morphological_matrix.md`](morphological_matrix.md);
for the per-component mapping see
[`implementations.md` → Layer Mapping](implementations.md#layer-mapping-bhati-2026).

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  L5  Governance & Safety                                                │
│      src/core/operations/  (AO / AH / AG / CG)                               │
│      env-enforced safe mode (src/eventsat/env.py)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  L4  Orchestration                                                      │
│      src/core/organization/  (SAS / CentralizedMAS / DMAS / IMAS / HMAS)│
│      src/core/experiment_runner.py                             │
├─────────────────────────────────────────────────────────────────────────┤
│  L3  Tools & Environment                                                │
│      src/core/satellite_env.py, src/eventsat/, src/ssa/, src/orbital/              │
├─────────────────────────────────────────────────────────────────────────┤
│  L2  Agent–Computer Interface                                           │
│      scenario action schemas and tools (for example src/eventsat/agentic_tools.py)                      │
├─────────────────────────────────────────────────────────────────────────┤
│  L1  Reasoning · Memory · Self-Reflection                               │
│      src/core/decision_procedure/  (SDA)                           │
│      src/core/memory/  (FixedMemory / WritableMemory)                        │
│      src/core/behaviour/  (BehaviourController, PPO, PromptOptimizer)        │
├─────────────────────────────────────────────────────────────────────────┤
│  L0  Foundation Model    [gap for symbolic variants]                    │
│      src/core/llm_client.py  (LLM backend)                    │
│      subsymbolic policy network  (RL substrate)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The `src/core/decision_procedure/` SDA driver at L1 is **held fixed** in
the EventSat benchmark — they are not a framework component (see
[`morphological_matrix.md`](morphological_matrix.md)).

**Dependency runs upward; feedback flows downward.** Each layer depends on those
below it (orchestration uses environment + tools; reasoning operates over memory and
foundation substrate). Governance/safety constraints — env-enforced safe mode,
ground-pass gating in CG/AG — reach into all layers from L5 downward, matching the
semantics of Bhati's Figure 3.

**L0 asymmetry.** Pure-symbolic variants (`symb`) have no L0 substrate; they sit at
L1 directly. This is by design: it isolates the cognitive-substrate effect while
keeping L2–L5 fixed across the framework.

## Design Principles

1. **Per-core representation**: each active core (onboard and/or ground) carries a representation = cognitive substrate × action space; the action space (single-shot vs agentic) is richer only for LLM-bearing cores. Full rationale: [`morphological_matrix.md`](morphological_matrix.md).
2. **Modularity**: Components can be swapped without affecting others.
3. **Reproducibility**: Configuration-driven experiments with seed control.
4. **Fair Comparison**: Same environment and metrics for all variants.
5. **Scientific Rigor**: Implementations follow established research papers.

## Component Interactions

### Experiment Flow

1. `ExperimentRunner` loads a YAML configuration.
2. Components are instantiated via factories (organization, decision loops, memory, environment, operations paradigm).
3. For each episode:
   a. Environment is reset (pre-computes orbital context: eclipses, ground passes).
   b. Memory and operations paradigm are reset.
   c. For each timestep:
      1. Environment provides full observation.
      2. Operations paradigm filters the observation (full state or stale ground knowledge).
      3. Organization distributes filtered observation to agents.
      4. Each agent's decision loop processes its observation and produces an action.
      5. Organization collects and aggregates agent actions.
      6. Operations paradigm processes actions (pass-through or buffer/gate by ground pass).
      7. Environment executes the processed actions and returns results.
      8. Ground knowledge is updated if downlink occurs during a ground pass.
      9. Metrics are collected.
4. Results are saved with full provenance (configuration + metrics + logs).

### Data Flow

```
Environment → Observation → OperationsParadigm.filter_observation()
                                              ↓
                            Organization → AgentObservation
                                              ↓
                                        DecisionProcedure.process()
                                              ↓
              AgentAction → Organization → OperationsParadigm.process_action()
                                              ↓
                                     Environment.step()
```

### Output Artifacts

Each run writes to `data/results/<experiment_id>/` (git-ignored):

- **`results.json`** — the compact, experiment-level artifact: `config`, `timestamp`,
  `experiment_statistics` (means/std), and per-episode **aggregated** metrics. It deliberately
  does **not** embed raw per-step observation/state snapshots — those ballooned the file to
  multi-GB on long symbolic runs (`e3ac71f`). The board (`scripts/refresh_board.py`) reads only
  this file.
- **Embedded telemetry** — the first `ExperimentRunner.TELEMETRY_SAMPLE_EPISODES` (3) episodes
  carry a compact, downsampled (≤1500-point) scalar-only `telemetry` block in `results.json`:
  battery, mode, data pools (stored/downlinked/jetson_raw/obc), ground-pass, sunlight, anomaly.
  Written regardless of log level; ~tens of KB/episode. Powers the board's Episode inspector and
  presentation graphs (`scripts/extract_telemetry.py`, `45cefce`).
- **`decisions_ep<N>.jsonl`** — the full per-step decision trace (rationale + raw telemetry),
  written **only when `log_level: DEBUG`**. `scripts/recompute_metrics.py` recomputes research
  metrics offline from this trace; the embedded telemetry block is the lighter, always-on subset.
- **`config.json`** — the resolved configuration; **`checkpoints/`** — per-episode snapshots
  when `save_checkpoints: true`.

### Orbital Context

Eclipse intervals and ground station passes are **pre-computed at episode reset** for the entire simulation duration. The `OrbitalContext` object (in `src/orbital/context.py`) stores these events and is queried each step to determine sunlight status and pass availability.

Two computation backends are supported:
- **Simplified** (always available): Analytical phase-fraction model for eclipses, stochastic pass generation for ground access.
- **Orekit** (optional): High-fidelity geometric shadow model and elevation-based pass computation using the Orekit astrodynamics library via `orekit-jpype`.

## Directory Layout

| Directory | Purpose |
|-----------|---------|
| `src/core/satellite_env.py` | Shared satellite environment interfaces |
| `src/orbital/` | Orbital mechanics (eclipse, ground access, optional Orekit) |
| `src/eventsat/` | EventSat and MultiEventsat environments, metrics, representations, trace export |
| `src/ssa/` | SSA constellation environment, RSO targets, rewards, metrics, symbolic planner |
| `src/rl/` | RLlib `MultiAgentEnv` bridge, space adapters, policy-sharing helpers, actor-critic model |
| `src/core/organization/` | Agent coordination patterns |
| `src/core/decision_procedure/` | Decision-making temporal patterns |
| `src/core/memory/` | `FixedMemory` (all cells, default); `WritableMemory` (agentic online-learning — CoALA §3) |
| `src/core/behaviour/` | `BehaviourController` (`@register`), legacy PPO trainer, RLlib PPO trainer, `PromptOptimizer` |
| `src/core/operations/` | Operations paradigm (autonomous onboard / hybrid / ground, conventional ground) |
| `src/core/` | Experiment runner, config, shared metrics collector, base interfaces |
| `configs/` | YAML experiment configurations |
| `tests/` | Comprehensive test suite |
| `docs/` | Architecture and design documentation |

## Key Design Decisions

### Why Fixed Memory?

All architecture variants access the **same** `FixedMemory` structure by default to ensure fair comparison. Only the representation module determines how stored information is interpreted and used. This isolates the effect of the cognitive architecture from memory design choices.

**Exception**: the agentic online-learning variant (`behaviour_config.mechanism = "writable_coala"`) uses `WritableMemory`, which adds writable semantic and episodic stores on top of `FixedMemory`. This deviation is intentional — it is compared against the same agentic cell with fixed memory, not against other cells. See `src/core/memory/writable_memory.py` and CLAUDE.md.

### Why YAML Configuration?

Every experimental choice is captured in a YAML file — no hardcoded decisions. This ensures:
- Full reproducibility from configuration + seed.
- Easy batch generation of experiment variants.
- Clear documentation of what was tested.

### Why Abstract Base Classes?

All components define clear ABCs before any implementation. This enforces:
- A stable contract between components.
- Independent development and testing.
- Easy addition of new variants without changing existing code.

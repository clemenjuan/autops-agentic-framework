# System Architecture

**Project:** Cognitive Satellite Constellation Autonomy Experimental Framework
**Researcher:** Clemente J. Juan Oliver, TUM Chair of Spacecraft Systems

---

## Overview

This framework enables systematic comparison of cognitive architectures for autonomous satellite constellation management. The design follows a **two-tier morphological matrix** (see [`FOUNDATION_SPEC.md` §3](FOUNDATION_SPEC.md#3-morphological-matrix-structure)): structural axes (Organization × Representation-substrate × Decision Procedure × Operations Paradigm, with a reactive/agentic action-space flavor under the hybrid substrate) plus a **Behaviour** overlay (hand-designed vs emergent) over the cognitive modules.

## Architecture Diagram

![AUTOPS overall system architecture diagram](images/autops-overall-system-architecture.svg)

## Layered View (Parallel to the Matrix)

Bhati (2026) [Z5TF79HY] proposes a six-layer reference architecture for **agentic
software engineering** systems. The autops framework is positioned as a **parallel
reference architecture** in a sibling domain (autonomous satellite operations), not
a structural adoption. The layered view below is complementary to the
morphological matrix (which remains canonical) and to the diagram above; it makes
the autops architectural choices legible to the broader agentic-AI literature. For
the framing see [`FOUNDATION_SPEC.md` §2.1](FOUNDATION_SPEC.md#21-parallel-reference-architecture-bhati-2026);
for the per-component mapping see
[`implementations.md` → Layer Mapping](implementations.md#layer-mapping-bhati-2026).

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  L5  Governance & Safety                                                │
│      src/operations/  (AH / AG / CG)                                    │
│      env-enforced safe mode (eventsat_env.py)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  L4  Orchestration                                                      │
│      src/agent_organization/  (SAS / CentralizedMAS / DMAS / IMAS / HMAS)│
│      src/orchestration/experiment_runner.py                             │
├─────────────────────────────────────────────────────────────────────────┤
│  L3  Tools & Environment                                                │
│      src/environment/  (satellite_env, scenarios, orbital)              │
├─────────────────────────────────────────────────────────────────────────┤
│  L2  Agent–Computer Interface                                           │
│      src/tools/  (BaseTool + scenario action defs)                      │
├─────────────────────────────────────────────────────────────────────────┤
│  L1  Reasoning · Memory · Self-Reflection                               │
│      src/decision_procedure/  (SDA / OODA / ReAct)                           │
│      src/memory/  (FixedMemory / WritableMemory)                        │
│      src/behaviour/  (BehaviourController, PPO, PromptOptimizer)        │
├─────────────────────────────────────────────────────────────────────────┤
│  L0  Foundation Model    [gap for symbolic variants]                    │
│      src/representation/llm_client.py  (LLM backend)                    │
│      subsymbolic policy network  (RL substrate)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Dependency runs upward; feedback flows downward.** Each layer depends on those
below it (orchestration uses environment + tools; reasoning operates over memory and
foundation substrate). Governance/safety constraints — env-enforced safe mode,
ground-pass gating in CG/AG — reach into all layers from L5 downward, matching the
semantics of Bhati's Figure 3.

**L0 asymmetry.** Pure-symbolic variants (`symb`) have no L0 substrate; they sit at
L1 directly. This is by design: it isolates the cognitive-paradigm effect that RQ1
targets while keeping L2–L5 fixed across the matrix.

## Design Principles

1. **Two-tier structure**: structural axes describe what the agent *is*; the Behaviour overlay describes which module is learned vs specified. Axes are not all mutually independent — action-space richness varies only under the hybrid substrate; Behaviour is an overlay, not a peer axis ([`FOUNDATION_SPEC.md` §3](FOUNDATION_SPEC.md#3-morphological-matrix-structure)).
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

### Orbital Context

Eclipse intervals and ground station passes are **pre-computed at episode reset** for the entire simulation duration. The `OrbitalContext` object (in `src/environment/orbital/context.py`) stores these events and is queried each step to determine sunlight status and pass availability.

Two computation backends are supported:
- **Simplified** (always available): Analytical phase-fraction model for eclipses, stochastic pass generation for ground access.
- **Orekit** (optional): High-fidelity geometric shadow model and elevation-based pass computation using the Orekit astrodynamics library via `orekit-jpype`.

## Directory Layout

| Directory | Purpose |
|-----------|---------|
| `src/environment/` | Satellite environment (abstract + scenario subclasses) |
| `src/environment/orbital/` | Orbital mechanics (eclipse, ground access, optional Orekit) |
| `src/environment/scenarios/` | Scenario environments (EventSat, Flamingo, ...) |
| `src/agent_organization/` | Agent coordination patterns |
| `src/decision_procedure/` | Decision-making temporal patterns |
| `src/representation/` | Knowledge & decision representations |
| `src/memory/` | `FixedMemory` (all variants, default); `WritableMemory` (`_lec_` only — CoALA §3) |
| `src/behaviour/` | Emergence controller (`@register`), `PPOTrainer`, `PromptOptimizer` |
| `src/operations/` | Operations paradigm (autonomous hybrid, conventional ground) |
| `src/tools/` | Action interfaces per scenario |
| `src/orchestration/` | Experiment runner, config, metrics, analysis |
| `configs/` | YAML experiment configurations |
| `tests/` | Comprehensive test suite |
| `docs/` | Architecture and design documentation |

## Key Design Decisions

### Why Fixed Memory?

All architecture variants access the **same** `FixedMemory` structure by default to ensure fair comparison. Only the representation module determines how stored information is interpreted and used. This isolates the effect of the cognitive architecture from memory design choices.

**Exception**: `_lec_` configs (`behaviour_config.mechanism = "writable_coala"`) use `WritableMemory`, which adds writable semantic and episodic stores on top of `FixedMemory`. This deviation is intentional — these variants are compared against the hand-designed agentic baseline only, not against other representation types. See `src/memory/writable_memory.py` and CLAUDE.md for the rationale.

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

# System Architecture

**Project:** Cognitive Satellite Constellation Autonomy Experimental Framework
**Researcher:** Clemente J. Juan Oliver, TUM Chair of Spacecraft Systems

---

## Overview

This framework enables systematic comparison of cognitive architectures for autonomous satellite constellation management. The design follows a **morphological matrix** approach, where each experimental dimension (agent organization, decision loop, representation, emergence mode, operations paradigm) is an orthogonal axis that can be varied independently.

## Architecture Diagram

![AUTOPS overall system architecture diagram](images/autops-overall-system-architecture.svg)

## Design Principles

1. **Orthogonality**: Each dimension (organization, loop, representation, emergence, operations paradigm) is independent.
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
                                        DecisionLoop.process()
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
| `src/decision_loop/` | Decision-making temporal patterns |
| `src/representation/` | Knowledge & decision representations |
| `src/memory/` | Fixed memory system (shared across all variants) |
| `src/emergence/` | Hand-designed vs. learned factory |
| `src/operations/` | Operations paradigm (autonomous hybrid, conventional ground) |
| `src/tools/` | Action interfaces per scenario |
| `src/orchestration/` | Experiment runner, config, metrics, analysis |
| `configs/` | YAML experiment configurations |
| `tests/` | Comprehensive test suite |
| `docs/` | Architecture and design documentation |

## Key Design Decisions

### Why Fixed Memory?

All architecture variants access the **same** memory structure to ensure fair comparison. Only the representation module determines how stored information is interpreted and used. This isolates the effect of the cognitive architecture from memory design choices.

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

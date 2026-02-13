# System Architecture

**Project:** Cognitive Satellite Constellation Autonomy Experimental Framework
**Researcher:** Clemente J. Juan Oliver, TUM Chair of Spacecraft Systems

---

## Overview

This framework enables systematic comparison of cognitive architectures for autonomous satellite constellation management. The design follows a **morphological matrix** approach, where each experimental dimension (agent organization, decision loop, representation, emergence mode) is an orthogonal axis that can be varied independently.

## Architecture Diagram

![AUTOPS overall system architecture diagram](images/autops-overall-system-architecture.svg)

## Design Principles

1. **Orthogonality**: Each dimension (organization, loop, representation, emergence) is independent.
2. **Modularity**: Components can be swapped without affecting others.
3. **Reproducibility**: Configuration-driven experiments with seed control.
4. **Fair Comparison**: Same environment and metrics for all variants.
5. **Scientific Rigor**: Implementations follow established research papers.

## Component Interactions

### Experiment Flow

1. `ExperimentRunner` loads a YAML configuration.
2. Components are instantiated via factories (organization, decision loops, memory, environment).
3. For each episode:
   a. Environment is reset.
   b. Memory is reset.
   c. For each timestep:
      - Organization distributes observations to agents.
      - Each agent's decision loop processes its observation and produces an action.
      - Organization collects and aggregates agent actions.
      - Environment executes the aggregated actions and returns results.
      - Metrics are collected.
4. Results are saved with full provenance (configuration + metrics + logs).

### Data Flow

```
Environment → Observation → Organization → AgentObservation
                                              ↓
                                        DecisionLoop.process()
                                              ↓
AgentAction → Organization → env_actions → Environment.step()
```

## Directory Layout

| Directory | Purpose |
|-----------|---------|
| `src/environment/` | Satellite environment (abstract + Orekit integration) |
| `src/agent_organization/` | Agent coordination patterns |
| `src/decision_loop/` | Decision-making temporal patterns |
| `src/representation/` | Knowledge & decision representations |
| `src/memory/` | Fixed memory system (shared across all variants) |
| `src/emergence/` | Hand-designed vs. learned factory |
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

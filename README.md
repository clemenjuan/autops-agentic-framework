# AUTOPS Experimental Framework

Systematic experimental framework for comparing cognitive architectures in autonomous satellite constellation management.

Part of the AUTOPS project at TUM Chair of Spacecraft Systems.

## Overview

This framework implements a **morphological matrix** approach to systematically explore
the design space of autonomous satellite constellation agents. The five independent
dimensions are:

| Dimension              | Options                                |
|------------------------|----------------------------------------|
| **Organization**       | Centralized, Hierarchical, Distributed |
| **Decision Loop**      | SDA, OODA, ReAct, (CoALA planned)      |
| **Representation**     | Symbolic, Hybrid (Neuro-symbolic)      |
| **Emergence**          | Hand-designed, Learned                 |
| **Operations Paradigm**| Autonomous Hybrid, Autonomous Ground, Conventional Ground |

Each combination defines a unique architecture that can be evaluated under
identical scenario conditions.

### Current Status

**Phase 3** — three decision loops × three operations paradigms = 9 experiment configurations:
- **Decision loops**: SDA (reactive baseline), OODA (Boyd's cycle with CBR orient), ReAct (iterative reason-act-observe with grounding checks)
- **Operations paradigms**: Autonomous Hybrid (onboard real-time), Autonomous Ground (algorithmic scheduler, pass-based), Conventional Ground (human-realistic with planning delay and cognitive constraints)
- **Representations**: Rule-based EventSat (OODA-aware + ReAct-capable), Schedule-based EventSat, Conventional Schedule EventSat (human cognitive constraints)
- Complete environment simulation (power, 3-pool data pipeline, comms, anomalies, detection)
- Orbital mechanics (analytical + optional Orekit J2 propagation, launch lottery)
- 7 research metrics + loop-specific metrics (OODA orient latency/urgency, ReAct reasoning depth/convergence)
- DecisionContext interface decoupling loops from representations

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup
```bash
# Install dependencies (including dev tools and optional orbital mechanics)
uv sync --extra dev --extra orbital

# Run the test suite
uv run python -m pytest tests/ -v -o "addopts="
```

### Running an Experiment
```bash
# Run EventSat experiments (naming: <scenario>_<org>_<loop>_<repr>_<emrg>_<ops>_v<N>)
# SDA loop (baseline)
uv run autops run configs/experiments/eventsat_cen_sda_symb_hd_ah.yaml    # autonomous hybrid
uv run autops run configs/experiments/eventsat_cen_sda_symb_hd_ag.yaml    # autonomous ground
uv run autops run configs/experiments/eventsat_cen_sda_symb_hd_cg.yaml    # conventional ground

# OODA loop
uv run autops run configs/experiments/eventsat_cen_ooda_symb_hd_ah.yaml   # autonomous hybrid
uv run autops run configs/experiments/eventsat_cen_ooda_symb_hd_ag.yaml   # autonomous ground
uv run autops run configs/experiments/eventsat_cen_ooda_symb_hd_cg.yaml   # conventional ground

# ReAct loop
uv run autops run configs/experiments/eventsat_cen_react_symb_hd_ah.yaml  # autonomous hybrid
uv run autops run configs/experiments/eventsat_cen_react_symb_hd_ag.yaml  # autonomous ground
uv run autops run configs/experiments/eventsat_cen_react_symb_hd_cg.yaml  # conventional ground

# Quick smoke test (1 episode, 100 steps)
uv run autops run configs/experiments/eventsat_cen_sda_symb_hd_ah.yaml --episodes 1 --steps 100

# Run and auto-generate analysis figures
uv run autops run configs/experiments/eventsat_cen_sda_symb_hd_ah.yaml --analyze
```

### Batch Experiments
```bash
# Generate config combinations from template
uv run autops generate --template configs/experiments/template.yaml

# Run all generated configs or all experiments in a folder
uv run autops batch configs/experiments/generated/
uv run autops batch configs/experiments/
```

### Analyzing Results
```bash
# Generate figures and summary from existing results
uv run autops analyze data/results/eventsat_cen_sda_symb_hd_ah/
```

For interactive exploration, use the Jupyter notebooks:
- `notebooks/telemetry.ipynb` — per-step satellite telemetry (battery, data, modes)
- `notebooks/analysis.ipynb` — research metrics comparison across architectures

## Project Structure

```
autops-demo/
+-- src/
|   +-- environment/          # Satellite constellation simulation (ABC + scenarios)
|   |   +-- orbital/          # Orbital mechanics (eclipse, ground access, Orekit wrapper)
|   |   +-- scenarios/        # Scenario environments (eventsat_env.py, ...)
|   +-- agent_organization/   # Centralized / Hierarchical / Distributed
|   +-- decision_loop/        # SDA / OODA / ReAct (+ DecisionContext interface)
|   +-- representation/       # Symbolic / Neuro-symbolic
|   +-- memory/               # Memory abstraction + FixedMemory impl
|   +-- emergence/            # Emergence controller & registry
|   +-- operations/           # Operations paradigm (autonomous_hybrid, autonomous_ground, conventional_ground)
|   +-- orchestration/        # Config loader, experiment runner, metrics, analysis
|   +-- tools/                # Domain-specific tools (placeholder)
+-- configs/
|   +-- experiments/          # YAML experiment configurations
|   +-- scenarios/            # Scenario definitions (eventsat.yaml, ...)
+-- scripts/
|   +-- generate_experiment_configs.py
|   +-- run_batch.py
+-- tests/                    # Unit and integration tests
+-- docs/
|   +-- FOUNDATION_SPEC.md    # Foundation specification
|   +-- implementations.md    # Implementation registry (components, paper basis, design decisions)
|   +-- architecture.md       # Architecture overview
|   +-- metrics.md            # Metrics definitions
|   +-- scenarios.md          # Scenario descriptions
+-- data/
|   +-- results/              # Experiment outputs (git-ignored)
|   +-- trained_models/       # Learned representations (git-ignored)
```

## Configuration

Experiments are defined via YAML files validated by Pydantic:

```yaml
experiment_id: "eventsat_cen_sda_symb_hd_ah"
agent_organization: centralized
decision_loop: sda
representation: symbolic
emergence_mode: hand_designed
operations_paradigm: autonomous_hybrid
environment:
  scenario: eventsat
  constellation_size: 1
  timestep_seconds: 60
  max_steps: 10080
num_episodes: 5
max_steps: 10080
```

See `configs/experiments/template.yaml` for the full schema.

## Testing

```bash
# All tests
uv run python -m pytest tests/ -v -o "addopts="

# Specific module
uv run python -m pytest tests/test_eventsat_physics.py -v -o "addopts="
```

## Documentation

- [Foundation Specification](docs/FOUNDATION_SPEC.md) — the governing spec
- [Implementation Registry](docs/implementations.md) — all components, paper basis, design decisions
- [Architecture Overview](docs/architecture.md)
- [Metrics Definitions](docs/metrics.md)
- [Scenario Descriptions](docs/scenarios.md)

## Contact

TUM Chair of Spacecraft Systems  
Clemente J. Juan Oliver  
clemente.juan@tum.de

---

**Supported by**:  
AUTOPS project, Bavarian Joint Research Program (BayVFP), MRF-2307-0004
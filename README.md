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
| **Decision Loop**      | SDA, OODA, ReAct                       |
| **Representation**     | Symbolic, Subsymbolic, Hybrid          |
| **Emergence**          | Hand-designed, Learned                 |
| **Operations Paradigm**| Autonomous Hybrid, Autonomous Ground, Conventional Ground |

Each combination defines a unique architecture that can be evaluated under
identical scenario conditions.

### Current Status

**Phase 4 complete** — 37 experiment configurations across the full morphological matrix:
- **Decision loops**: SDA (reactive baseline), OODA (Boyd's cycle with CBR orient), ReAct (iterative reason-act-observe with grounding checks)
- **Operations paradigms**: Autonomous Hybrid (onboard real-time), Autonomous Ground (algorithmic scheduler, pass-based), Conventional Ground (human-realistic with planning delay and cognitive constraints)
- **Representations** (4 types, 5 implementations):
  - *Symbolic*: Rule-based (OODA-aware + ReAct-capable), Schedule-based, Conventional Schedule (human cognitive constraints)
  - *Hybrid — LLM single-shot*: `llm_eventsat` (Rodriguez-Fernandez et al. 2024)
  - *Hybrid — Agentic*: `agentic_eventsat` (CoALA, Sumers et al. 2024) — multi-step Plan-Tool-Reflect-Decide with 6 domain tools
  - *Subsymbolic — RL*: `subsymbolic_eventsat` (PPO, Oliver et al. 2025) — 25D obs, MultiDiscrete actions, trainable policy
- **Inference gating**: Ground-based paradigms (AG/CG) only run LLM/agentic inference during ground passes (Rossi et al. 2023)
- Complete environment simulation (power, 3-pool data pipeline, comms, anomalies, detection)
- Orbital mechanics (analytical + optional Orekit J2 propagation, launch lottery)
- 7 research metrics + loop-specific + representation-specific metrics
- DecisionContext interface decoupling loops from representations
- 493 tests across 18 test modules

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
# Run EventSat experiments (naming: <scenario>_<org>_<loop>_<repr>_<emrg>_<ops>)
# org: sas | cmas | dmas    loop: sda | ooda | react    repr: symb | hybr | subm | agnt
# emrg: hd | le | lep | lec    ops: ah | ag | cg
# SDA loop (baseline)
uv run autops run configs/experiments/eventsat_sas_sda_symb_hd_ah.yaml    # autonomous hybrid

# Quick smoke test (1 episode, 100 steps)
uv run autops run configs/experiments/eventsat_sas_sda_symb_hd_ah.yaml --episodes 1 --steps 100

# Run and auto-generate analysis figures
uv run autops run configs/experiments/eventsat_sas_sda_symb_hd_ah.yaml --analyze
```

### Batch Experiments
```bash
# Generate config combinations from template
uv run autops generate --template configs/experiments/template.yaml

# Quick sanity check
uv run autops batch configs/experiments --episodes 1 --steps 200

# Run batch of specific configurations
uv run autops batch configs/experiments/eventsat_cmas_*.yaml --episodes 1 --steps 200  # all CentralizedMAS configs
uv run autops batch configs/experiments/eventsat_cmas_*_symb_*.yaml --episodes 1 --steps 200  # all CentralizedMAS symbolic configs

# Run all generated configs or all experiments in a folder
uv run autops batch configs/experiments/generated/
uv run autops batch configs/experiments/
uv run autops batch configs/experiments --episodes 5 --steps 10080
```

### Analyzing Results
```bash
# Generate figures and summary from existing results
uv run autops analyze data/results/eventsat_sas_sda_symb_hd_ah/
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
|   +-- agent_organization/   # SAS / CentralizedMAS / DecentralizedMAS / IndependentMAS / HybridMAS
|   +-- decision_loop/        # SDA / OODA / ReAct (+ DecisionContext interface)
|   +-- representation/       # Symbolic / Subsymbolic / Hybrid + LLM client + agentic tools
|   +-- memory/               # Memory abstraction + FixedMemory impl
|   +-- emergence/            # Emergence controller, registry, rollout buffer, training pipeline
|   +-- operations/           # Operations paradigm (autonomous_hybrid, autonomous_ground, conventional_ground)
|   +-- orchestration/        # Config loader, experiment runner, metrics, analysis
+-- configs/
|   +-- experiments/          # 37 YAML experiment configurations + template
|   +-- scenarios/            # Scenario definitions (eventsat.yaml, ...)
+-- scripts/
|   +-- generate_experiment_configs.py
|   +-- run_batch.py
|   +-- train_subsymbolic.py  # PPO training script for RL representation
+-- tests/                    # 18 test modules, 493 tests
+-- docs/
|   +-- FOUNDATION_SPEC.md    # Foundation specification
|   +-- implementations.md    # Implementation registry (components, paper basis, design decisions)
|   +-- architecture.md       # Architecture overview
|   +-- metrics.md            # Metrics definitions
|   +-- scenarios.md          # Scenario descriptions
+-- data/
|   +-- results/              # Experiment outputs (git-ignored)
|   +-- trained_models/       # Learned representations (git-ignored)
|   +-- llm_cache/            # LLM response cache with prompts (git-ignored)
```

## Configuration

Experiments are defined via YAML files validated by Pydantic:

```yaml
experiment_id: "eventsat_sas_sda_symb_hd_ah"
agent_organization: sas
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
 
Clemente J. Juan Oliver  
clemente.juan@tum.de

---

**Supported by**:  
AUTOPS project, Bavarian Joint Research Program (BayVFP), MRF-2307-0004.
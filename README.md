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
| **Decision Loop**      | Reactive, Deliberative, Layered-hierarchical |
| **Representation**     | Symbolic, Hybrid (Neuro-symbolic)      |
| **Emergence**          | Hand-designed, Learned                 |
| **Operations Paradigm**| Autonomous Hybrid, Conventional Ground |

Each combination defines a unique architecture that can be evaluated under
identical scenario conditions.

### Current Status

**EventSat baseline** (TUM single-satellite mission) is fully implemented with:
- Complete environment simulation (power, 3-pool data pipeline, comms, anomalies, detection)
- Orbital mechanics module (simplified analytical models + optional Orekit integration)
- Pre-computed eclipse intervals and ground station passes
- Pipeline backpressure (observation limited by downlink capacity, per Proposal Section 6.1)
- Rule-based SDA decision loop with symbolic representation
- Both operations paradigms (autonomous hybrid, conventional ground)
- 7 research metrics collected per episode (utility, data downlink efficiency, latency, robustness, resource efficiency, operator load, explainability)
- 159 tests passing (4 Orekit-specific tests skipped when Orekit is not installed)

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
uv run autops run configs/experiments/eventsat_cen_sda_symb_hd_ah.yaml     # autonomous hybrid
uv run autops run configs/experiments/eventsat_cen_sda_symb_hd_cg.yaml    # conventional ground

# Quick test with fewer episodes and shorter sim
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
|   +-- decision_loop/        # SDA / OODA / CoALA / custom loops
|   +-- representation/       # Symbolic / Neuro-symbolic
|   +-- memory/               # Memory abstraction + FixedMemory impl
|   +-- emergence/            # Emergence controller & registry
|   +-- operations/           # Operations paradigm (autonomous_hybrid, conventional_ground)
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
|   +-- architecture.md       # Architecture overview
|   +-- metrics.md            # Metrics definitions
|   +-- scenarios.md          # Scenario descriptions
|   +-- implementation_guide.md
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

- [Foundation Specification](docs/FOUNDATION_SPEC.md) -- the governing spec
- [Architecture Overview](docs/architecture.md)
- [Metrics Definitions](docs/metrics.md)
- [Scenario Descriptions](docs/scenarios.md)
- [Implementation Guide](docs/implementation_guide.md)

## Contact

TUM Chair of Spacecraft Systems  
Clemente J. Juan Oliver  
clemente.juan@tum.de

---

**Supported by**:  
AUTOPS project, Bavarian Joint Research Program (BayVFP), MRF-2307-0004
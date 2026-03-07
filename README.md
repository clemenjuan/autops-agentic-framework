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
- Complete environment simulation (power, data, comms, anomalies)
- Orbital mechanics module (simplified analytical models + optional Orekit integration)
- Pre-computed eclipse intervals and ground station passes
- Rule-based SDA decision loop with symbolic representation
- Both operations paradigms (autonomous hybrid, conventional ground)
- 92 tests passing (4 Orekit-specific tests skipped when Orekit is not installed)

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup
```bash
# Install dependencies
uv sync

# Install dev tools (pytest, etc.)
uv sync --extra dev

# Run the test suite
uv run pytest tests/ -v
```

### Running an Experiment
```bash
# Run EventSat baseline experiment
uv run python -c "
from src.orchestration.config_loader import load_config
from src.orchestration.experiment_runner import ExperimentRunner
cfg = load_config('configs/experiments/eventsat_baseline.yaml')
runner = ExperimentRunner(config=cfg)
results = runner.run()
"

# Batch run all configs in a directory
uv run python scripts/run_batch.py configs/experiments/

# Generate full morphological matrix configs
uv run python scripts/generate_experiment_configs.py
```

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
experiment_id: "eventsat_baseline"
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
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ -v --cov=src --cov-report=html

# Specific module
uv run pytest tests/test_environment.py -v
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
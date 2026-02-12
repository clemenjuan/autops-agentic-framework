# AUTOPS Experimental Framework

Systematic experimental framework for comparing cognitive architectures in autonomous satellite constellation management.

Part of the AUTOPS project at TUM Chair of Spacecraft Systems.

## Overview

This framework implements a **morphological matrix** approach to systematically explore
the design space of autonomous satellite constellation agents. The four independent
dimensions are:

| Dimension          | Options                                |
|--------------------|----------------------------------------|
| **Organization**   | Centralized, Hierarchical, Distributed |
| **Decision Loop**  | Reactive, Deliberative, Hybrid         |
| **Representation** | Symbolic, Sub-symbolic, Hybrid         |
| **Emergence**      | Hand-designed, Learned                 |

Each combination defines a unique architecture that can be evaluated under
identical scenario conditions.

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
# Single experiment from a YAML config
uv run python -c "
from src.orchestration.config_loader import load_config
from src.orchestration.experiment_runner import ExperimentRunner
cfg = load_config('configs/experiments/template.yaml')
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
|   +-- agent_organization/   # Centralized / Hierarchical / Distributed
|   +-- decision_loop/        # Reactive / Deliberative / Hybrid (ABC)
|   +-- representation/       # Symbolic / Sub-symbolic / Hybrid (ABC)
|   +-- memory/               # Memory abstraction + FixedMemory impl
|   +-- emergence/            # Emergence controller & registry
|   +-- orchestration/        # Config loader, experiment runner, metrics, analysis
|   +-- tools/                # Domain-specific tools (placeholder)
+-- configs/
|   +-- experiments/          # YAML experiment configurations
|   +-- scenarios/            # Scenario definitions
+-- scripts/
|   +-- generate_experiment_configs.py
|   +-- run_batch.py
+-- tests/                    # 55 tests, 78% coverage
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
experiment_name: "example"
organization: centralized
decision_loops:
  - type: reactive
representation: symbolic
emergence_mode: hand_designed
environment:
  scenario: "coverage_optimization"
  num_satellites: 6
  max_steps: 500
num_episodes: 10
random_seed: 42
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

**Supported by**: AUTOPS project, Bavarian Joint Research Program (BayVFP), MRF-2307-0004
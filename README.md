# AUTOPS Experimental Framework

Benchmark framework for comparing cognitive architectures for autonomous satellite operations.

Part of the AUTOPS project at the TUM Chair of Spacecraft Systems. The current
benchmark is EventSat: a controlled operations-system comparison over
organisation, representation, and operational paradigm. For the full research
definition, see [docs/morphological_matrix.md](docs/morphological_matrix.md).

New contributors should start with [START_HERE.md](START_HERE.md).

## Repository Map

```text
src/
  core/       orchestration, config loading, operations paradigms, memory, behaviour
  eventsat/   EventSat and MultiEventsat environments, representations, schedulers, metrics, rewards
  flamingo/   Flamingo-lite multi-satellite scenario
  rl/         RLlib bridge, space adapters, policy mapping, actor-critic model
  orbital/    eclipse, ground access, link budget, optional Orekit wrapper
configs/
  experiments/  canonical experiment YAML files
  scenarios/    scenario definitions
scripts/         config generation, board building, smoke runs, maintenance utilities
tests/           regression, physics, orchestration, and representation tests
docs/            canonical framework docs and implementation notes
data/            generated outputs and local artifacts, mostly git-ignored
archive/         historical notes and admin scripts outside the student path
```

## Install

Prerequisites: Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync --extra dev --extra orbital
```

Optional extras:

```bash
uv sync --extra dev --extra rl
uv sync --extra dev --extra llm
```

## Minimal Run

```bash
uv run autops run configs/experiments/eventsat_sas_ag_symb.yaml --episodes 1 --steps 100
```

Results are written under `data/results/`.

## Common Commands

```bash
uv run python -m pytest tests/
uv run autops batch configs/experiments --episodes 1 --steps 200
uv run autops generate --template configs/experiments/template.yaml
uv run autops train configs/experiments/eventsat_sas_ao_rl.yaml
uv run autops train configs/experiments/multieventsat_imas_sda_subm_le_ah.yaml
uv run python scripts/refresh_board.py
```

The generated boards are written to `data/figures/`; open `data/figures/index.html`
after refreshing.

## Where To Look

- Core orchestration: `src/core/experiment_runner.py`, `src/core/config_loader.py`
- EventSat algorithms: `src/eventsat/`
- Experiment configs: `configs/experiments/`
- Scenario configs: `configs/scenarios/`
- Script map: `scripts/README.md`
- Generated-data policy: `data/README.md`
- Framework spec: `docs/morphological_matrix.md`
- Implementation registry: `docs/implementations.md`
- Architecture overview: `docs/architecture.md`

Generated results, logs, caches, notebooks, coverage reports, model checkpoints,
and local run artifacts should stay out of Git.

## Contact

Clemente J. Juan Oliver  
clemente.juan@tum.de

Supported by AUTOPS project, Bavarian Joint Research Program (BayVFP),
MRF-2307-0004.

# Start Here

## What This Repository Does

- Runs the EventSat operations-system benchmark.
- Compares architectures across organisation, representation, and operational paradigm.
- Uses YAML configs so experiments are reproducible and easy to inspect.

## Setup

- Install Python 3.11+ and `uv`.
- Run `uv sync --extra dev --extra orbital`.
- Use `uv sync --extra dev --extra rl` for RL training.
- Use `uv sync --extra dev --extra llm` for live LLM experiments.

## Run One Smoke Experiment

```bash
uv run autops run configs/experiments/eventsat_sas_ag_symb.yaml --episodes 1 --steps 100
```

The result lands in `data/results/eventsat_sas_ag_symb/`.

## Where To Read Code

- `src/core/experiment_runner.py`: wires configs into environments, algorithms, operations, and metrics.
- `src/core/config_loader.py`: validates YAML configs and resolves framework cells.
- `src/eventsat/`: EventSat environment, representation implementations, schedulers, rewards, and metrics.
- `src/core/operations/`: ground, onboard, hybrid, and conventional execution constraints.
- `tests/`: executable examples of expected behaviour.

## Modify A Core Algorithm

- Start in the relevant `src/eventsat/` representation file.
- Keep public function and class names stable unless the benchmark design changes.
- Register new representation classes with `@register(...)` in `src/core/behaviour/controller.py`.
- Ensure the module is imported in `ExperimentRunner._create_decision_loops()` if registration is needed.
- Add or update focused tests next to the existing representation tests.

## Generated Outputs

- `data/results/`: experiment outputs.
- `data/figures/`: generated result boards.
- `data/llm_cache/`: cached LLM responses.
- `data/trained_models/`, `data/trained_prompts/`, `data/writable_memory_state/`: learned artifacts.
- `logs/`: local run logs.

These are local artifacts and should not be committed.

## Before You Commit

- Run the smallest useful test first, then the full suite when practical.
- Do not commit generated outputs, notebooks, caches, coverage reports, or local environment files.
- Update `docs/implementations.md` and `docs/morphological_matrix.md` when framework behaviour changes.

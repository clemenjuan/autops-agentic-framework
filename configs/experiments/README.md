# Experiment Configurations

YAML configurations for the EventSat operations-system (O) benchmark. Each file
defines one complete experiment. Canonical spec: [`docs/morphological_matrix.md`](../../docs/morphological_matrix.md).

## Naming convention

```
eventsat_sas_<paradigm>_<rep>.yaml
```

- `paradigm`: `conventional` · `ag` · `ao` · `ah`
- `rep` (7 cells): `symb` · `rl` · `hrl` · `llm-s` · `llm-a` · `hllm-s` · `hllm-a`
  (`ao` onboard is restricted to `symb`/`rl`/`hrl`)
- the dual-core `ah` names **both** cores, onboard-first:
  `eventsat_sas_ah_<onboard>_<ground>` (e.g. `eventsat_sas_ah_rl_llm-s`). A dual-core
  AH config uses nested `onboard:` / `ground:` blocks, each with its own
  `representation` + `representation_config` (see `template.yaml`); onboard ∈
  {symb, rl, hrl} (no per-step LLM onboard), ground ∈ the 7 cells. Single-rep AH
  (both cores same) keeps the flat single `representation`.

The full matrix is **32 experiments** (conventional 1 + ag 7 + ao 3 + ah 21).

## Current configurations

Increment 1 of the code remap ships the framework-valid subset that already has
runnable cores; the remaining cells (the `hrl`/`llm-a` placeholders, the LLM
ground schedulers, and the 21 `ah_<onboard>_<ground>` pairs) land in later
increments.

| Config | Paradigm | Representation |
|--------|----------|----------------|
| `eventsat_sas_conventional_symb` | Conventional | symbolic |
| `eventsat_sas_ag_symb` | Autonomous Ground | symbolic |
| `eventsat_sas_ag_rl` | Autonomous Ground | RL (ground scheduler) |
| `eventsat_sas_ag_hllm-s` | Autonomous Ground | single-shot hybrid LLM |
| `eventsat_sas_ag_hllm-a` | Autonomous Ground | agentic hybrid LLM |
| `eventsat_sas_ao_symb` | Autonomous Onboard | symbolic |
| `eventsat_sas_ao_rl` | Autonomous Onboard | RL (PPO) |
| `eventsat_sas_ah_symb_symb` | Autonomous Hybrid | symbolic onboard + symbolic ground |
| `eventsat_sas_ah_rl_rl` | Autonomous Hybrid | RL onboard + RL ground |
| `eventsat_sas_ah_rl_symb` | Autonomous Hybrid | RL onboard + symbolic ground |
| `eventsat_sas_ah_symb_hllm-a` | Autonomous Hybrid | symbolic onboard + agentic-LLM ground |

`template.yaml` documents every field.

## Usage

```bash
uv run autops run configs/experiments/eventsat_sas_ag_symb.yaml
uv run autops run configs/experiments/eventsat_sas_ag_symb.yaml --episodes 1 --steps 100  # smoke test
uv run autops batch configs/experiments/   # run all
```

## Validation

All configs are validated on load by Pydantic (`src/orchestration/config_loader.py`);
invalid configurations raise clear errors. The `representation` field accepts the
7-cell tokens above (normalised to the internal substrate + action space) as well
as the legacy substrate values.

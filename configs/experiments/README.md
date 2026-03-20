# Experiment Configuration Guide

This directory contains YAML configuration files for experiments.
Each file defines a complete experimental setup.

## Naming Convention

```
eventsat_<org>_<loop>_<repr>_<emergence>_<ops>.yaml
```

- `org`: cen (centralized)
- `loop`: sda, ooda, react
- `repr`: symb (symbolic)
- `emergence`: hd (hand-designed)
- `ops`: ah (autonomous hybrid), ag (autonomous ground), cg (conventional ground)

## Current Configurations (Phase 3)

| Config | Loop | Ops Paradigm | Representation |
|--------|------|--------------|----------------|
| `eventsat_cen_sda_symb_hd_ah` | SDA | Autonomous Hybrid | Rule-based |
| `eventsat_cen_sda_symb_hd_ag` | SDA | Autonomous Ground | Schedule-based |
| `eventsat_cen_sda_symb_hd_cg` | SDA | Conventional Ground | Conventional Schedule |
| `eventsat_cen_ooda_symb_hd_ah` | OODA | Autonomous Hybrid | Rule-based |
| `eventsat_cen_ooda_symb_hd_ag` | OODA | Autonomous Ground | Schedule-based |
| `eventsat_cen_ooda_symb_hd_cg` | OODA | Conventional Ground | Conventional Schedule |
| `eventsat_cen_react_symb_hd_ah` | ReAct | Autonomous Hybrid | Rule-based |
| `eventsat_cen_react_symb_hd_ag` | ReAct | Autonomous Ground | Schedule-based |
| `eventsat_cen_react_symb_hd_cg` | ReAct | Conventional Ground | Conventional Schedule |

## Usage

```bash
uv run autops run configs/experiments/eventsat_cen_sda_symb_hd_ah.yaml
uv run autops run configs/experiments/eventsat_cen_sda_symb_hd_ah.yaml --episodes 1 --steps 100  # smoke test
uv run autops batch configs/experiments/  # run all
```

## Validation

All configs are validated on load using Pydantic. Invalid configurations
will raise clear error messages listing the issue.

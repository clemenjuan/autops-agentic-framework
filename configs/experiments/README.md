# Experiment Configuration Guide

This directory contains YAML configuration files for experiments.
Each file defines a complete experimental setup.

## Usage

1. **Copy the template**:
   ```bash
   cp template.yaml exp_001_centralized_sda_symbolic.yaml
   ```

2. **Edit the copy** to set the desired morphological matrix dimensions
   and parameters.

3. **Run the experiment**:
   ```bash
   python -m scripts.run_batch configs/experiments/exp_001_centralized_sda_symbolic.yaml
   ```

## Naming Convention

Use descriptive names following the pattern:
```
exp_<number>_<organization>_<loop>_<representation>[_<extra>].yaml
```

Examples:
- `exp_001_centralized_sda_symbolic.yaml`
- `exp_002_distributed_coala_hybrid.yaml`
- `exp_010_hierarchical_ooda_neural_learned.yaml`

## Batch Generation

Use `scripts/generate_experiment_configs.py` to generate config files
for all combinations of morphological matrix dimensions.

## Validation

All configs are validated on load using Pydantic. Invalid configurations
will raise clear error messages listing the issue.

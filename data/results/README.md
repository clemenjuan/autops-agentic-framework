# Experiment Results

This directory stores experiment output data.
Each experiment creates a subdirectory named after its `experiment_id`.

Structure per experiment:
```
<experiment_id>/
├── results.json        # Full results with metrics
├── config.json         # Copy of the configuration used
├── experiment.log      # Execution log
└── checkpoints/        # Episode checkpoints (if enabled)
```

**Note:** This directory is git-ignored to avoid committing large result files.
Add results to `.gitignore` if not already present.

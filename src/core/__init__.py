"""
Orchestration Module.

Experiment orchestration layer providing:
- Configuration management (YAML per experiment)
- Metrics collection (utility, latency, robustness, operator load)
- Reproducibility (seed control, logging, checkpointing)
- Statistical analysis & Pareto frontier computation

Reference-architecture layer: L4 (Orchestration) in the Bhati 2026 mapping.
See ``docs/implementations.md`` "Layer Mapping (Bhati 2026)" for details.
"""

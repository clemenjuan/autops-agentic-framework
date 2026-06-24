# Data And Generated Artifacts

This directory is intentionally mostly empty in Git. Runtime outputs are local and
ignored so the repository stays small for students and collaborators.

Common generated paths:

- `data/results/`: experiment results and logs.
- `data/figures/`: generated HTML result boards and compact extracts.
- `data/llm_cache/`: cached LLM responses keyed by prompt hash.
- `data/trained_models/`: learned policy checkpoints.
- `data/trained_prompts/`: prompt-optimized system prompts.
- `data/writable_memory_state/`: persisted writable-memory state for agentic runs.
- `data/world_model/`: exported world-model datasets and traces.

Keep generated artifacts out of commits unless a future release explicitly defines
a small canonical fixture.

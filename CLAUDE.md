# CLAUDE.md — Project Context for Claude Code

## Project Overview
AUTOPS Experimental Framework — PhD research at TUM Chair of Spacecraft Systems.
Systematic comparison of cognitive architectures for autonomous satellite constellation management using a morphological matrix approach (Organization × Decision Loop × Representation × Emergence).

Scalability is treated as a **2D space**: constellation size (1 → 500+ satellites) × structural complexity (centralized → distributed), enabling joint scaling law derivation across both axes (RQ3).

**Three concrete scenarios** define the experimental progression:
1. **EventSat** — TUM's own single satellite; Phase 2 baseline (full subsystem data available).
2. **Vyoma Flamingo** — AUTOPS partner constellation, up to 12 satellites; medium-scale Phase 3.
3. **Space-Based Data Centers** — large-scale 100+ satellite concept; Phase 3–4 scalability stress test.

**Seven performance metrics:** utility, latency, robustness, resource efficiency, operator load, explainability, scale & complexity.

## Repository Layout
- `src/` — Core framework (environment, agent_organization, decision_loop, representation, memory, emergence, orchestration, tools)
- `configs/` — YAML experiment configs and scenario definitions
- `scripts/` — Batch runners and config generators
- `tests/` — Test suite (pytest)
- `docs/` — Specifications, architecture docs, metrics, scenarios
- `data/results/` and `data/trained_models/` — Git-ignored outputs
- `old-coala-framework` branch — Previous CoALA-based prototype (catalog, visualization, orbital toolkit). Preserved for potential reuse.

## Tech Stack & Conventions
- **Python 3.11+** with **uv** package manager
- **pytest** for testing (`uv run pytest tests/ -v`)
- **Pydantic** for config validation
- ABCs define extension points (environment, decision loops, representations)
- YAML for experiment configuration
- Code lives in `src/`, tests mirror structure in `tests/`

## Common Commands
```bash
uv sync                              # Install deps
uv sync --extra dev                  # Install dev deps
uv run pytest tests/ -v              # Run all tests
uv run pytest tests/ -v --cov=src    # Tests with coverage
uv run python scripts/run_batch.py configs/experiments/  # Batch run
```

## Key Design Decisions
- Each architecture dimension is an independent axis — implementations must be composable
- Experiments are fully defined by YAML config — no code changes needed to run a new combination
- The ExperimentRunner orchestrates everything: config → agent assembly → simulation → metrics
- Metrics are domain-specific (coverage gap, response time, fuel efficiency, etc.)

## Workflow Preferences
- Always run tests after code changes (`uv run pytest tests/ -v`)
- Use feature branches for new work (`feature/<short-description>`)
- Commit messages: conventional commits style (feat:, fix:, refactor:, test:, docs:)
- Ask before making architectural decisions that affect the morphological matrix dimensions
- Reference the Foundation Spec (`docs/FOUNDATION_SPEC.md`) as the governing document

## What NOT to Do
- Don't modify the morphological matrix dimensions without explicit approval
- Don't commit to main directly — use feature branches + PRs
- Don't add dependencies without discussing first
- Don't create files in `data/results/` or `data/trained_models/` (git-ignored, runtime only)

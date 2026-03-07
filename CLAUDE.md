# CLAUDE.md

## What this is
PhD experimental framework (TUM Chair of Spacecraft Systems). Compares cognitive architectures for autonomous satellite ops via a 5D morphological matrix: Organization × Decision Loop × Representation × Emergence × Operations Paradigm.

## Commands
```bash
uv sync --extra dev                  # Install all deps
uv run pytest tests/ -v -o "addopts="  # Run tests (addopts override needed — no pytest-cov)
```

## Rules
- **Run tests after every code change**
- Feature branches only (`feature/<name>`), never commit to main
- Conventional commits (feat:, fix:, refactor:, test:, docs:)
- Ask before changing morphological matrix dimensions or adding dependencies
- `docs/FOUNDATION_SPEC.md` is the governing document — check it first

## Layout
- `src/` — environment (+ orbital/, scenarios/), agent_organization, decision_loop, representation, memory, emergence, operations, orchestration, tools
- `configs/` — experiment YAMLs + scenario definitions
- `tests/` — pytest suite (92 pass, 4 skip without Orekit)
- `data/results/`, `data/trained_models/` — git-ignored, runtime only

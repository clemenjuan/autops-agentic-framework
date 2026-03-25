# CLAUDE.md

## What this is
PhD experimental framework (TUM Chair of Spacecraft Systems). Compares cognitive architectures for autonomous satellite ops via a 5D morphological matrix: Organization × Decision Loop × Representation × Emergence × Operations Paradigm.

## Path & venv
- Project is at `C:\Users\Clemente\autops-demo` (local, not OneDrive).
- The project `.venv` is at the repo root.
- **Always use `uv run`** — it picks the correct `.venv`.
- If `uv sync` fails with hardlink errors, run with `UV_LINK_MODE=copy uv sync ...`.

## Commands
```bash
uv sync --extra dev --extra orbital   # Install all deps (including Orekit)
uv run pytest tests/ -v -o "addopts="  # Run tests

# Run experiments (naming: <scenario>_<org>_<loop>_<repr>_<emrg>_<ops>_v<N>)
uv run autops run configs/experiments/eventsat_cen_sda_symb_hd_ah.yaml    # autonomous hybrid
uv run autops run configs/experiments/eventsat_cen_sda_symb_hd_cg.yaml   # conventional ground

# Quick smoke test (1 episode, 100 steps)
uv run autops run configs/experiments/eventsat_cen_sda_symb_hd_ah.yaml --episodes 1 --steps 100

# Run and auto-analyze
uv run autops run configs/experiments/eventsat_cen_sda_symb_hd_ah.yaml --analyze

# Analyze existing results
uv run autops analyze data/results/eventsat_cen_sda_symb_hd_ah/

# Batch run / generate configs — see uv run autops --help
```

## Rules
- **Run tests after every code change**
- Feature branches only (`feature/<name>`), never commit to main
- Conventional commits (feat:, fix:, refactor:, test:, docs:)
- Ask before changing morphological matrix dimensions or adding dependencies
- **Before planning new components:** check `docs/FOUNDATION_SPEC.md` (scientific grounding) + Zotero library (literature). Every implementation must be grounded in specific papers.
- **After implementation:** update `docs/implementations.md` (what was built) + `docs/FOUNDATION_SPEC.md` if scientific grounding changed. Do NOT create new doc files.

## Doc map (single source per topic — no duplication)
- **Research grounding** (RQs, morphological matrix, cognitive paradigm taxonomy, design principles, phase roadmap): `docs/FOUNDATION_SPEC.md`
- **What's been built** (component registry, paper basis, design decisions): `docs/implementations.md`
- **How to add new components** (step-by-step guide): `docs/implementation_guide.md`
- **Metrics definitions**: `docs/metrics.md`
- **Scenario specifications**: `docs/scenarios.md`
- **System architecture diagram + data flow**: `docs/architecture.md`
- **EventSat physics parameters**: this file (CLAUDE.md), section below

## Layout
- `src/` — environment (+ orbital/, scenarios/), agent_organization, decision_loop, representation, memory, emergence, operations, orchestration, tools
- `configs/` — experiment YAMLs + scenario definitions
- `tests/` — pytest suite
- `data/results/`, `data/trained_models/` — git-ignored, runtime only

## EventSat scenario physics (current)
- Orbit: 400 km SSO (Proposal Section 13), period 5554s, inclination 97.4°
- Propagator: EcksteinHechler J2 (via Orekit); models ~1.04°/day RAAN precession. Set `orbit.propagator: keplerian` for two-body only.
- Launch lottery: RAAN/ArgP/TA randomized per episode (`orbit.launch_lottery: true`); altitude/inclination/eccentricity fixed. Monte Carlo across episodes captures rideshare insertion uncertainty.
- Ground passes: Orekit elevation-based (Ottobrunn 48.05°N, min 10° elevation) when Orekit installed; stochastic fallback otherwise.
- Data pipeline: Jetson raw → Jetson compressed → OBC → S-band (3-pool, P3)
- Observation size: 9.41 MB raw/obs (measured: 6.64 MB/42.36 s × 60 s)
- Compression: 5.11:1 (measured: 6.64 MB → 1.3 MB), 2× obs time (P1); Detection: 5 min CV (P4)
- Mode transitions: 135s ADCS settling for observe/communicate (P2)
- Daily downlink budget: 27 MB configurable (`daily_downlink_budget_mb` in eventsat.yaml)
- Jetson→OBC transfer: RS-485 one-way (50 kbps), requires explicit `payload_send` mode
- Anomalies: environment-enforced safe mode (agent cannot override); recovery depends on ops paradigm — conventional ground requires active ground pass (resume command), autonomous hybrid clears via onboard FDIR once countdown expires
- Modes: charging, communication, payload_observe, payload_compress, payload_detect, payload_send, safe
- Metrics: `utility` (mission threshold) + `data_downlink_efficiency` (achieved/max achievable) + `anomaly_forced_safe` (FDIR-forced vs voluntary safe)

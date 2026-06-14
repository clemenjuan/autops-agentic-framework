# CLAUDE.md

## What this is
PhD experimental framework (TUM Chair of Spacecraft Systems). Benchmarks **cognitive architectures for autonomous satellite operations** on the **EventSat** mission (an event-camera CubeSat). Scenario-specific and metric-based — **no mission tradespace, no test catalogue, no multi-fidelity surrogate.**

An architecture is an **operations system (O)** with **three components** (canonical spec: `docs/morphological_matrix.md`):
- **Organisation** — `sas` for EventSat (MAS variants belong to the future multi-satellite scenario).
- **Representation** = cognitive **substrate** (symbolic · subsymbolic-RL · subsymbolic-LLM · hybrid) × **action space** (single-shot · agentic) → **7 cells**: `symb · rl · hrl · llm-s · llm-a · hllm-s · hllm-a`.
- **Operational paradigm** — `conventional` (symbolic only, one-pass delay) · `ag` (autonomous ground, all 7) · `ao` (autonomous onboard, no-LLM → `symb/rl/hrl`) · `ah` (autonomous hybrid, **dual-core**: onboard {3} × ground {7}).

EventSat·SAS = **32 experiments** (1 + 7 + 3 + 21), scored on 14 metrics (`morphological_matrix.md` §6). Names: `eventsat_sas_<paradigm>_<rep>`; AH names both cores onboard-first: `eventsat_sas_ah_<onboard>_<ground>`.

## Execution environment
- **Live LLM experiments** (anything with `llm_mock: false`, plus `lep` training) are I/O-bound on Ollama. Run them on a machine with low-latency reach to an Ollama endpoint (ideally co-located) that can stay up uninterrupted for hours. A workstation over HTTPS works but is slow and fragile.
- Workstations are fine for: editing code, `pytest`, mocked smoke tests (`llm_mock: true`), config inspection, plan and doc work.
- `data/results/`, `data/llm_cache/`, `data/trained_models/`, `data/trained_prompts/`, `data/writable_memory_state/` are runtime artifacts (all gitignored). Treat the canonical-run machine's copy as ground truth; workstation copies are stale.
- Orekit needs a JVM. On Linux: `apt install openjdk-17-jre-headless` and place `orekit-data.zip` at repo root before `uv sync --extra orbital`. WSL has known issues — use a native Linux VM or Windows.
- See `CLAUDE.local.md` (gitignored) for the canonical machine, hostnames, and personal paths if present.

## Path & venv
- Always use **`uv run`** — it picks the correct `.venv` at repo root.
- If `uv sync` fails with hardlink errors: `UV_LINK_MODE=copy uv sync ...`
- A `.venv/` built on a different OS will not work — if `pyvenv.cfg` points to a foreign Python (e.g. `linux-x86_64` on a Windows host), delete and rebuild with `UV_LINK_MODE=copy uv sync --extra dev --extra orbital --extra llm`.

## Commands
```bash
uv sync --extra dev --extra orbital        # Install all deps (including Orekit)
uv sync --extra dev --extra llm            # Add LLM providers (openai, requests)
uv sync --extra dev --extra rl             # Add RL deps (torch, gymnasium)
uv run pytest tests/ -v -o "addopts="     # Full test suite (692 tests: 669 pass, 23 RL skipped without --extra rl)
uv run pytest tests/test_llm_representation.py -v -o "addopts="  # Single module

# Run experiments — name = eventsat_sas_<paradigm>_<rep>  (morphological_matrix.md §5)
#   paradigm: conventional | ag | ao | ah
#   rep:      symb | rl | hrl | llm-s | llm-a | hllm-s | hllm-a   (ao: symb/rl/hrl only)
#   ah names both cores onboard-first:  eventsat_sas_ah_<onboard>_<ground>
uv run autops run configs/experiments/eventsat_sas_ag_symb.yaml
uv run autops run configs/experiments/eventsat_sas_ao_rl.yaml            # RL onboard (PPO; --extra rl)
uv run autops run configs/experiments/eventsat_sas_ag_llm-s.yaml         # single-shot LLM ground (qwen3.6:35b)
uv run autops run configs/experiments/eventsat_sas_ah_rl_llm-s.yaml      # RL onboard + LLM ground

# Quick smoke test (1 episode, 100 steps)
uv run autops run configs/experiments/eventsat_sas_ag_symb.yaml --episodes 1 --steps 100

# Training (RL cells)
uv run autops train configs/experiments/eventsat_sas_ao_rl.yaml                  # PPO

# Batch run / analyze — see uv run autops --help
```

## Rules
- **Run tests after every code change**
- Trunk-based: commit small focused changes directly to `main`; tests must stay green per commit
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- Ask before changing the O-framework components (organisation / representation / paradigm) or adding dependencies
- **Before planning:** read `docs/morphological_matrix.md` + Zotero library. Every implementation must cite specific papers.
- **After implementation:** update `docs/implementations.md` + `docs/morphological_matrix.md` if the framework changed. Do NOT create new doc files.

## Doc map (single source per topic)
| Topic | File |
|---|---|
| **Canonical spec** — the O framework (3 components, 7 reps, 4 paradigms, 32 EventSat experiments), naming, the 14 metrics | `docs/morphological_matrix.md` |
| Component registry, paper basis, design decisions | `docs/implementations.md` |
| How to add new components (step-by-step) | `docs/implementation_guide.md` |
| Scenario specifications | `docs/scenarios.md` |
| System architecture diagram + data flow | `docs/architecture.md` |
| EventSat physics parameters | This file (below) |

## Architecture
```
src/
  environment/        # Satellite sim (ABC + EventSat scenario + orbital/)
  agent_organization/ # SAS (EventSat) + CentralizedMAS / DecentralizedMAS / IndependentMAS / HybridMAS (Kim et al. 2025; multi-satellite scenario)
  decision_procedure/      # per-step decision driver (+DecisionContext interface)
  representation/     # Symbolic / Subsymbolic / Hybrid cores + llm_client.py
  memory/             # FixedMemory (default, all cells); WritableMemory (agentic online-learning variant — see below)
  behaviour/          # controller.py @register() factory; training_pipeline.py (PPO); prompt_optimizer.py
  operations/         # autonomous_onboard / autonomous_hybrid / autonomous_ground / conventional_ground
  orchestration/      # config_loader.py (Pydantic) + experiment_runner.py
  tools/              # BaseTool interface + per-scenario action definitions (stateless, YAML-serializable)
configs/experiments/  # EventSat experiment configs + template (the 32-experiment matrix — morphological_matrix.md §4)
tests/                # test suite (683 pass, 23 RL skipped without --extra rl)
```

**Key interfaces:**
- `DecisionContext(state, loop_type, memory, enrichments, loop_metadata)` — passes from loop → representation
- `@register("name")` decorator on representation class → auto-registers in `BehaviourController`
- New representations must be imported in `experiment_runner.py` `_create_decision_loops()` to trigger registration

## Memory invariant
All cells use `FixedMemory` for fair comparison. **Exception**: the **agentic online-learning
variant** (`writable_coala`) uses `WritableMemory`, which adds writable semantic + episodic
stores (CoALA §3, Sumers et al. 2024) — the *online-learning* internal action available only to
agentic cells (`llm-a` / `hllm-a`). It deviates from the fairness invariant intentionally and is
compared against the *same* agentic cell with fixed memory, not against other representations.
See `src/memory/writable_memory.py`.

## Coding conventions
- Pydantic v2 for all config validation (`src/orchestration/config_loader.py`)
- The config `representation` field + `action_space` resolve to the concrete `@register` class via `ExperimentConfig.resolved_representation_type` (e.g. symbolic+AH→`rule_based_eventsat`, hybrid+agentic+AH→`agentic_eventsat`, hybrid+reactive+AG→`llm_scheduler_eventsat`). NB: the config *content* values still use the legacy substrate terms (`symbolic/subsymbolic/hybrid`) — mapping them onto the 7 framework cells (`symb/rl/hrl/llm-s/llm-a/hllm-s/hllm-a`) is part of the step-by-step code work, not done yet. `representation_config.type` is an optional explicit override.
- Loop-specific data goes in `context.enrichments`, never in representation state
- All representations must implement `encode_observation()` + `select_action()`; optionally `reason()` for ReAct, `update()` for learned variants
- Rationale strings always set `self._last_rationale` for explainability metrics
- `TYPE_CHECKING` guard for `DecisionContext` imports to avoid circular imports

## Testing
```bash
uv run pytest tests/ -v -o "addopts="                     # All tests (clears coverage flags)
uv run pytest tests/test_X.py::TestClass::test_method -v -o "addopts="  # Single test
```
- `pyproject.toml` default `addopts` adds coverage — override with `-o "addopts="` to suppress
- LLM tests use `llm_mock: true` in config — **never require a live LLM in tests**
- `test_orbital.py` requires Orekit JVM; may fail if Orekit not installed (expected)

## Gotchas
- `uv run` not `python` — running `python` directly misses the venv
- `representation_config.type` must match an `@register("name")` string exactly; typos give `KeyError` from `BehaviourController`
- LLM experiments require `OLLAMA_HOST` env var (TUM: `https://ollama.sps.ed.tum.de`) or `OPENAI_API_KEY`; use `llm_mock: true` for local dev without LLM access
- LLM response cache at `data/llm_cache/` — delete to force fresh calls  
- `autonomous_ground` and `conventional_ground` ops paradigms require `operations_paradigm_config.orbital_period_steps: 93`
- Config validator warns (not errors) on degenerate loop × representation combinations (deterministic rep + non-SDA loop)
- `data/results/` and `data/trained_models/` are git-ignored — never commit experiment output

## EventSat scenario physics
- Orbit: 400 km SSO, period 5554s, inclination 97.4°
- Propagator: EcksteinHechler J2 (Orekit); stochastic fallback without Orekit
- Launch lottery: RAAN/ArgP/TA randomized per episode; altitude/inclination fixed
- Ground station: Ottobrunn 48.05°N, min 10° elevation
- Data pipeline (3-pool): Jetson raw → Jetson compressed → OBC → S-band
- Observation: 9.41 MB raw, 5.11:1 compression, 2× obs time to compress, 5 min CV detection
- ADCS settling: 135s for observe/communicate mode transitions
- Modes: `charging`, `communication`, `payload_observe`, `payload_compress`, `payload_detect`, `payload_send`, `safe`
- Anomaly: environment-enforced safe mode; onboard paradigms (AO/AH) clear via onboard FDIR, AG/CG require ground pass resume command
- Daily downlink budget: 27 MB (configurable in `eventsat.yaml`)
- Jetson-based onboard cores (subsymbolic/hybrid onboard, AO/AH) keep the Jetson powered → `power.onboard_compute_w` (≈7 W) **added** to modes where it'd otherwise be off (charging/communication/safe), but NOT to the Jetson-on payload modes `power.jetson_active_modes` (observe/compress/detect/send — the event camera + pipeline already keep the Jetson powered; no double-count). Symbolic onboard runs on the OBC (sub-watt) → no overhead; ground paradigms (AG/CG) → no overhead. Wired via `config.onboard_uses_jetson` → `env.onboard_compute_active`
- Jetson→OBC: RS-485 50 kbps one-way, requires explicit `payload_send` mode

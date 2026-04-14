# CLAUDE.md

## What this is
PhD experimental framework (TUM Chair of Spacecraft Systems). Compares cognitive architectures for autonomous satellite constellation ops via a **5D morphological matrix**: Organization × Decision Loop × Representation × Emergence × Operations Paradigm. Each combination is a unique architecture evaluated under identical conditions.

## Path & venv
- Project: `C:\Users\Clemente\autops-demo` (local, not OneDrive)
- Always use **`uv run`** — it picks the correct `.venv` at repo root
- If `uv sync` fails with hardlink errors: `UV_LINK_MODE=copy uv sync ...`

## Commands
```bash
uv sync --extra dev --extra orbital        # Install all deps (including Orekit)
uv sync --extra dev --extra llm            # Add LLM providers (openai, requests)
uv sync --extra dev --extra rl             # Add RL deps (torch, gymnasium)
uv run pytest tests/ -v -o "addopts="     # Full test suite (552 tests)
uv run pytest tests/test_llm_representation.py -v -o "addopts="  # Single module

# Run experiments (naming: <scenario>_<org>_<loop>_<repr>_<emrg>_<ops>)
# org: sas | cmas | dmas    loop: sda | ooda | react    repr: symb | hybr | subm | agnt
# emrg: hd | le | lep | lec    ops: ah | ag | cg
uv run autops run configs/experiments/eventsat_sas_sda_symb_hd_ah.yaml
uv run autops run configs/experiments/eventsat_sas_sda_hybr_hd_ah.yaml  # LLM hybrid
uv run autops run configs/experiments/eventsat_sas_sda_subm_le_ah.yaml  # RL subsymbolic
uv run autops run configs/experiments/eventsat_sas_sda_agnt_hd_ah.yaml  # Agentic hybrid
uv run autops run configs/experiments/eventsat_sas_sda_agnt_lec_ah.yaml # Writable CoALA
uv run autops run configs/experiments/eventsat_sas_sda_agnt_lep_ah.yaml # Prompt-optimized

# Quick smoke test (1 episode, 100 steps)
uv run autops run configs/experiments/eventsat_sas_sda_symb_hd_ah.yaml --episodes 1 --steps 100

# Training learned-emergence variants
uv run autops train configs/experiments/eventsat_sas_sda_subm_le_ah.yaml         # PPO
uv run autops train configs/experiments/eventsat_sas_sda_hybr_lep_ah.yaml        # prompt-opt
uv run autops train configs/experiments/eventsat_sas_sda_agnt_lec_ah.yaml        # writable CoALA

# Batch run / analyze — see uv run autops --help
```

## Rules
- **Run tests after every code change**
- Trunk-based: commit small focused changes directly to `main`; tests must stay green per commit
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- Ask before changing morphological matrix dimensions or adding dependencies
- **Before planning:** read `docs/FOUNDATION_SPEC.md` + Zotero library. Every implementation must cite specific papers.
- **After implementation:** update `docs/implementations.md` + `docs/FOUNDATION_SPEC.md` if scientific grounding changed. Do NOT create new doc files.

## Doc map (single source per topic)
| Topic | File |
|---|---|
| RQs, morphological matrix, cognitive paradigm taxonomy, phase roadmap | `docs/FOUNDATION_SPEC.md` |
| Component registry, paper basis, design decisions | `docs/implementations.md` |
| How to add new components (step-by-step) | `docs/implementation_guide.md` |
| Metrics definitions | `docs/metrics.md` |
| Scenario specifications | `docs/scenarios.md` |
| System architecture diagram + data flow | `docs/architecture.md` |
| EventSat physics parameters | This file (below) |

## Architecture
```
src/
  environment/        # Satellite sim (ABC + EventSat scenario + orbital/)
  agent_organization/ # SAS / CentralizedMAS / DecentralizedMAS / IndependentMAS / HybridMAS (Kim et al. 2025)
  decision_loop/      # SDA / OODA / ReAct  (+DecisionContext interface)
  representation/     # Symbolic / Hybrid / Subsymbolic + llm_client.py
  memory/             # FixedMemory (default, all variants); WritableMemory (_lec_ only — see below)
  emergence/          # controller.py @register() factory; training_pipeline.py (PPO); prompt_optimizer.py
  operations/         # autonomous_hybrid / autonomous_ground / conventional_ground
  orchestration/      # config_loader.py (Pydantic) + experiment_runner.py
  tools/              # BaseTool interface + per-scenario action definitions (stateless, YAML-serializable)
configs/experiments/  # 84 YAML experiment configs + 1 template (96 total incl. learned variants)
tests/                # 21 test modules, 552 tests
```

**Key interfaces:**
- `DecisionContext(state, loop_type, memory, enrichments, loop_metadata)` — passes from loop → representation
- `@register("name")` decorator on representation class → auto-registers in `EmergenceController`
- New representations must be imported in `experiment_runner.py` `_create_decision_loops()` to trigger registration

## Memory invariant
All hand-designed and non-CoALA learned variants use `FixedMemory` for fair comparison.
**Exception**: `_lec_` configs (`emergence_config.mechanism = "writable_coala"`) use `WritableMemory`,
which adds writable semantic + episodic stores (CoALA §3, Sumers et al. 2024). This deviates
from the fairness invariant intentionally — these variants are compared against the
hand-designed agentic baseline (`_agnt_hd_`), not against other representations.
See `src/memory/writable_memory.py` for the implementation.

## Coding conventions
- Pydantic v2 for all config validation (`src/orchestration/config_loader.py`)
- `representation: symbolic | subsymbolic | hybrid` — top-level dimension; `representation_config.type` picks the specific implementation (e.g. `rule_based_eventsat`, `llm_eventsat`)
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
- `representation_config.type` must match an `@register("name")` string exactly; typos give `KeyError` from `EmergenceController`
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
- Anomaly: environment-enforced safe mode; AH clears via onboard FDIR, CG requires ground pass resume command
- Daily downlink budget: 27 MB (configurable in `eventsat.yaml`)
- Jetson→OBC: RS-485 50 kbps one-way, requires explicit `payload_send` mode

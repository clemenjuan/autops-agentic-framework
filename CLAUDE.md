# CLAUDE.md

## What this is
PhD experimental framework (TUM Chair of Spacecraft Systems). Compares cognitive architectures for autonomous satellite constellation ops via a **two-tier morphological matrix** (see `docs/FOUNDATION_SPEC.md` §3): structural axes — Organization × Representation (substrate: symbolic/subsymbolic/hybrid) × Decision Procedure × Operations Paradigm, with a reactive/agentic action-space flavor under the hybrid substrate — plus a **Behaviour** overlay (hand-designed vs emergent). Each combination is a unique architecture evaluated under identical conditions.

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
uv run pytest tests/ -v -o "addopts="     # Full test suite (631 tests; 23 RL skipped without --extra rl)
uv run pytest tests/test_llm_representation.py -v -o "addopts="  # Single module

# Run experiments (naming: <scenario>_<org>_<proc>_<repr>_<beh>_<ops>)
# org:  sas | cmas (instantiated)                proc: sda | ooda | react
# repr: symb | hyre | subm | hyag                ops:  ao | ah | ag | cg
# beh:  hd (hand_designed) | le (ppo) | lep (prompt_optimized) | lec (writable_coala)
#
# Canonical config values (as accepted by config_loader.py):
#   agent_organization: sas | centralized_mas | decentralized_mas | independent_mas | hybrid_mas
#   behaviour_config.mechanism: hand_designed | ppo | prompt_optimized | writable_coala
# (dmas/imas/hmas are registered but deferred to Flamingo N>=3 scenarios — see Architecture.)
uv run autops run configs/experiments/eventsat_sas_sda_symb_hd_ah.yaml
uv run autops run configs/experiments/eventsat_sas_sda_hyre_hd_ah.yaml  # LLM hybrid
uv run autops run configs/experiments/eventsat_sas_sda_subm_le_ah.yaml  # RL subsymbolic
uv run autops run configs/experiments/eventsat_sas_sda_hyag_hd_ah.yaml  # Agentic hybrid
uv run autops run configs/experiments/eventsat_sas_sda_hyag_lec_ah.yaml # Writable CoALA
uv run autops run configs/experiments/eventsat_sas_sda_hyag_lep_ah.yaml # Prompt-optimized

# Quick smoke test (1 episode, 100 steps)
uv run autops run configs/experiments/eventsat_sas_sda_symb_hd_ah.yaml --episodes 1 --steps 100

# Training learned-emergence variants
uv run autops train configs/experiments/eventsat_sas_sda_subm_le_ah.yaml         # PPO
uv run autops train configs/experiments/eventsat_sas_sda_hyre_lep_ah.yaml        # prompt-opt
uv run autops train configs/experiments/eventsat_sas_sda_hyag_lec_ah.yaml        # writable CoALA

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
  agent_organization/ # SAS / CentralizedMAS (instantiated) + DecentralizedMAS / IndependentMAS / HybridMAS (Kim et al. 2025; deferred to Flamingo N>=3)
  decision_procedure/      # SDA / OODA / ReAct  (+DecisionContext interface)
  representation/     # Symbolic / Hybrid / Subsymbolic + llm_client.py
  memory/             # FixedMemory (default, all variants); WritableMemory (_lec_ only — see below)
  behaviour/          # controller.py @register() factory; training_pipeline.py (PPO); prompt_optimizer.py
  operations/         # autonomous_onboard / autonomous_hybrid / autonomous_ground / conventional_ground
  orchestration/      # config_loader.py (Pydantic) + experiment_runner.py
  tools/              # BaseTool interface + per-scenario action definitions (stateless, YAML-serializable)
configs/experiments/  # 84 experiment configs + 1 template (48 hand-designed + 36 learned variants)
tests/                # 22 test modules, 631 tests (23 RL skipped without --extra rl)
```

**Key interfaces:**
- `DecisionContext(state, loop_type, memory, enrichments, loop_metadata)` — passes from loop → representation
- `@register("name")` decorator on representation class → auto-registers in `BehaviourController`
- New representations must be imported in `experiment_runner.py` `_create_decision_loops()` to trigger registration

## Memory invariant
All hand-designed and non-CoALA learned variants use `FixedMemory` for fair comparison.
**Exception**: `_lec_` configs (`behaviour_config.mechanism = "writable_coala"`) use `WritableMemory`,
which adds writable semantic + episodic stores (CoALA §3, Sumers et al. 2024). This deviates
from the fairness invariant intentionally — these variants are compared against the
hand-designed agentic baseline (`_hyag_hd_`), not against other representations.
See `src/memory/writable_memory.py` for the implementation.

## Coding conventions
- Pydantic v2 for all config validation (`src/orchestration/config_loader.py`)
- `representation: symbolic | subsymbolic | hybrid` — substrate. The concrete implementation class is **resolved** from `representation × representation_config.action_space (hybrid only) × operations_paradigm` (e.g. symbolic+AH→`rule_based_eventsat`, hybrid+agentic+AH→`agentic_eventsat`, hybrid+reactive+AG→`llm_scheduler_eventsat`). `representation_config.type` is an **optional override** (e.g. the `_algobase` CG cell). See `ExperimentConfig.resolved_representation_type`.
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
- Jetson-based onboard cores (subsymbolic/hybrid onboard, AO/AH) keep the Jetson powered → a power **floor** `power.onboard_compute_w` (≈7 W) applied as `max(per_mode, floor)`, not added: idle/charging draws the floor, but compress/send/comms dominate (no double-count). Symbolic onboard runs on the OBC (sub-watt) → no floor; ground paradigms (AG/CG) → no floor. Wired via `config.onboard_uses_jetson` → `env.onboard_compute_active`
- Jetson→OBC: RS-485 50 kbps one-way, requires explicit `payload_send` mode

# AUTOPS Experimental Framework

Systematic experimental framework for comparing cognitive architectures in autonomous satellite constellation management.

Part of the AUTOPS project at TUM Chair of Spacecraft Systems.

## Overview

This framework implements a **two-tier morphological matrix** to systematically explore
the design space of autonomous satellite constellation agents. **Structural axes** describe
what the agent *is*; a **Behaviour overlay** describes where its competence comes from.

| Structural axis        | Options                                                      |
|------------------------|--------------------------------------------------------------|
| **Organization**       | SAS, Centralized MAS (instantiated); Decentralized / Independent / Hybrid MAS deferred to constellation scenarios (Kim et al. 2025) |
| **Representation** (substrate) | Symbolic, Subsymbolic, Hybrid — hybrid split by **action space**: reactive vs agentic |
| **Decision Procedure** | SDA, OODA, ReAct                                            |
| **Operations Paradigm**| Autonomous Onboard, Autonomous Hybrid, Autonomous Ground, Conventional Ground |

**Behaviour overlay**: Hand-designed vs Emergent — the learning mechanism (PPO / prompt-optimized /
writable-CoALA) is derived from the substrate, not chosen separately.

Each combination defines a unique architecture evaluated under identical conditions. The concrete
representation class is resolved from `representation × action_space × operations_paradigm` (see
[docs/FOUNDATION_SPEC.md §3](docs/FOUNDATION_SPEC.md#3-morphological-matrix-structure)).

### Current Status

**Phase 5 complete** — 91 experiment configurations across the full morphological matrix:
- **Decision loops**: SDA (reactive baseline), OODA (Boyd's cycle with CBR orient), ReAct (iterative reason-act-observe with grounding checks)
- **Operations paradigms**: Autonomous Onboard (onboard-only, per-step real-time), Autonomous Hybrid (onboard + ground plan + override), Autonomous Ground (algorithmic scheduler, pass-based), Conventional Ground (human-realistic with planning delay and cognitive constraints). Jetson-based onboard cores (subsymbolic/hybrid onboard, AO/AH) add a ~7 W Jetson-on draw (`power.onboard_compute_w`) to non-Jetson modes (not to compress/detect/send, which already include the Jetson); symbolic onboard runs on the OBC with no overhead.
- **Representations** (3 substrates; concrete class resolved from substrate × action_space × ops):
  - *Symbolic*: Rule-based (OODA-aware + ReAct-capable), Schedule-based, Conventional Schedule (human cognitive constraints)
  - *Hybrid — reactive* (single-shot LLM): `llm_eventsat` (Rodriguez-Fernandez et al. 2024)
  - *Hybrid — agentic* (tool-using): `agentic_eventsat` (CoALA, Sumers et al. 2024) — multi-step Plan-Tool-Reflect-Decide with 6 domain tools
  - *Subsymbolic — RL*: `subsymbolic_eventsat` (PPO, Juan Oliver et al. 2025) — trainable per-step policy (AH)

  Under ground paradigms (AG/CG), non-symbolic representations act as **schedule producers** —
  **distinct** long-horizon planners (a full-pass RL planner; real LLM/agentic planners), shared by
  AH and AG. AH additionally runs the per-step onboard policy that can override the uplinked plan, so
  AH-vs-AG isolates the onboard-override effect. These planners (`subsymbolic_scheduler_eventsat`,
  `llm_scheduler_eventsat`, `agentic_scheduler_eventsat`) are currently symbolic-planner
  **placeholders** (`is_placeholder`), pending Phase 4.e.
- **Learned-emergence for LLM representations** (Phase 5):
  - `prompt_optimized` (`_lep_`): offline bootstrap few-shot prompt optimization (DSPy-style; 24 configs)
  - `writable_coala` (`_lec_`): online CoALA memory accretion with writable semantic + episodic stores (12 configs)
- **`autops train` CLI**: dispatches PPO training, prompt optimization, or CoALA guidance by config
- **Inference gating**: Ground-based paradigms (AG/CG) only run LLM/agentic inference during ground passes (Rossi et al. 2023)
- Complete environment simulation (power, 3-pool data pipeline, comms, anomalies, detection)
- Orbital mechanics (analytical + optional Orekit J2 propagation, launch lottery)
- 7 research metrics + loop-specific + representation-specific metrics
- DecisionContext interface decoupling decision procedures from representations
- 660 tests (637 passing; 23 RL tests skipped without the `rl` extra)

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup
```bash
# Install dependencies (including dev tools and optional orbital mechanics)
uv sync --extra dev --extra orbital

# Run the test suite
uv run python -m pytest tests/ -v -o "addopts="
```

### Running an Experiment
```bash
# Run EventSat experiments (naming: <scenario>_<org>_<proc>_<repr>_<beh>_<ops>)
# org:  sas | cmas  (canonical YAML values: sas | centralized_mas; dmas/imas/hmas deferred)
# proc: sda | ooda | react           repr: symb | hyre | subm | hyag
# beh:  hd (hand_designed) | le (ppo) | lep (prompt_optimized) | lec (writable_coala)
# ops:  ao (autonomous_onboard) | ah | ag | cg
# SDA loop (baseline)
uv run autops run configs/experiments/eventsat_sas_sda_symb_hd_ah.yaml    # hand-designed symbolic
uv run autops run configs/experiments/eventsat_sas_sda_hyag_hd_ah.yaml    # hand-designed agentic
uv run autops run configs/experiments/eventsat_sas_sda_hyag_lec_ah.yaml   # writable-CoALA
uv run autops run configs/experiments/eventsat_sas_sda_hyag_lep_ah.yaml   # prompt-optimized

# Quick smoke test (1 episode, 100 steps)
uv run autops run configs/experiments/eventsat_sas_sda_symb_hd_ah.yaml --episodes 1 --steps 100

# Run and auto-generate analysis figures
uv run autops run configs/experiments/eventsat_sas_sda_symb_hd_ah.yaml --analyze
```

### Training Learned-Emergence Variants
```bash
# PPO (subsymbolic)
uv run autops train configs/experiments/eventsat_sas_sda_subm_le_ah.yaml

# Prompt-optimization (LLM / agentic)
uv run autops train configs/experiments/eventsat_sas_sda_hyre_lep_ah.yaml
uv run autops train configs/experiments/eventsat_sas_sda_hyag_lep_ah.yaml

# Writable CoALA (no pre-training; memory accretes at runtime)
uv run autops train configs/experiments/eventsat_sas_sda_hyag_lec_ah.yaml
```

### Batch Experiments
```bash
# Generate config combinations from template
uv run autops generate --template configs/experiments/template.yaml

# Quick sanity check
uv run autops batch configs/experiments --episodes 1 --steps 200

# Run batch of specific configurations
uv run autops batch configs/experiments/eventsat_cmas_*.yaml --episodes 1 --steps 200  # all CentralizedMAS configs
uv run autops batch configs/experiments/eventsat_cmas_*_symb_*.yaml --episodes 1 --steps 200  # all CentralizedMAS symbolic configs

# Run all generated configs or all experiments in a folder
uv run autops batch configs/experiments/generated/
uv run autops batch configs/experiments/
uv run autops batch configs/experiments --episodes 5 --steps 10080
```

### Analyzing Results
```bash
# Generate figures and summary from existing results
uv run autops analyze data/results/eventsat_sas_sda_symb_hd_ah/
```

For interactive exploration, use the Jupyter notebooks:
- `notebooks/telemetry.ipynb` — per-step satellite telemetry (battery, data, modes)
- `notebooks/analysis.ipynb` — research metrics comparison across architectures

## Project Structure

```
autops-demo/
+-- src/
|   +-- environment/          # Satellite constellation simulation (ABC + scenarios)
|   |   +-- orbital/          # Orbital mechanics (eclipse, ground access, Orekit wrapper)
|   |   +-- scenarios/        # Scenario environments (eventsat_env.py, ...)
|   +-- agent_organization/   # SAS / CentralizedMAS / DecentralizedMAS / IndependentMAS / HybridMAS
|   +-- decision_procedure/        # SDA / OODA / ReAct (+ DecisionContext interface)
|   +-- representation/       # Symbolic / Subsymbolic / Hybrid + LLM client + agentic tools
|   +-- memory/               # FixedMemory (all variants) + WritableMemory (_lec_ only, CoALA §3)
|   +-- behaviour/            # controller.py, training_pipeline.py (PPO), prompt_optimizer.py
|   +-- operations/           # Operations paradigm (autonomous_onboard, autonomous_hybrid, autonomous_ground, conventional_ground)
|   +-- orchestration/        # Config loader, experiment runner, metrics, analysis
+-- configs/
|   +-- experiments/          # 91 experiment configs + 1 template
|   +-- scenarios/            # Scenario definitions (eventsat.yaml, ...)
+-- scripts/
|   +-- generate_experiment_configs.py
|   +-- run_batch.py
|   +-- train_subsymbolic.py  # PPO training script for RL representation
+-- tests/                    # 660 tests (637 pass; 23 RL skipped without --extra rl)
+-- docs/
|   +-- FOUNDATION_SPEC.md    # Foundation specification
|   +-- implementations.md    # Implementation registry (components, paper basis, design decisions)
|   +-- architecture.md       # Architecture overview
|   +-- metrics.md            # Metrics definitions
|   +-- scenarios.md          # Scenario descriptions
+-- data/
|   +-- results/              # Experiment outputs (git-ignored)
|   +-- trained_models/       # PPO policy checkpoints (git-ignored)
|   +-- trained_prompts/      # Prompt-optimized system prompts (git-ignored)
|   +-- writable_memory_state/# WritableMemory persistence for _lec_ runs (git-ignored)
|   +-- llm_cache/            # LLM response cache with prompts (git-ignored)
```

## Configuration

Experiments are defined via YAML files validated by Pydantic:

```yaml
experiment_id: "eventsat_sas_sda_symb_hd_ah"
agent_organization: sas
decision_procedure: sda
representation: symbolic
behaviour: hand_designed
operations_paradigm: autonomous_hybrid
environment:
  scenario: eventsat
  constellation_size: 1
  timestep_seconds: 60
  max_steps: 10080
num_episodes: 5
max_steps: 10080
```

Hybrid experiments add `representation_config.action_space: reactive | agentic`; the concrete
representation class is then resolved from `representation × action_space × operations_paradigm`
(no `representation_config.type` needed — it remains only as an optional override). See
`configs/experiments/template.yaml` for the full schema.

## Testing

```bash
# All tests
uv run python -m pytest tests/ -v -o "addopts="

# Specific module
uv run python -m pytest tests/test_eventsat_physics.py -v -o "addopts="
```

## Documentation

- [Foundation Specification](docs/FOUNDATION_SPEC.md) — the governing spec
- [Implementation Registry](docs/implementations.md) — all components, paper basis, design decisions
- [Architecture Overview](docs/architecture.md)
- [Metrics Definitions](docs/metrics.md)
- [Scenario Descriptions](docs/scenarios.md)

## Contact
 
Clemente J. Juan Oliver  
clemente.juan@tum.de

---

**Supported by**:  
AUTOPS project, Bavarian Joint Research Program (BayVFP), MRF-2307-0004.
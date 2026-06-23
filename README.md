# AUTOPS Experimental Framework

**NO TRASH FILES.** Keep the repo concise: add only necessary source, configs, tests, and canonical docs. Generated data, notebooks, coverage reports, caches, scratch experiments, and one-off artifacts stay ignored or outside the repo.

Systematic experimental framework for comparing cognitive architectures in autonomous satellite constellation management.

Part of the AUTOPS project at TUM Chair of Spacecraft Systems.

## Overview

This framework is the **engine** for the **EventSat operations-system (O) benchmark** — a
controlled, metric-based comparison of cognitive architectures for autonomous satellite operations
on a real event-camera CubeSat (canonical spec: [docs/morphological_matrix.md](docs/morphological_matrix.md)).
Scenario-specific (EventSat now; multi-satellite later); no mission tradespace, no test catalogue,
no multi-fidelity surrogate.

An architecture has **three components**:

| Component | Options |
|-----------|---------|
| **Organisation** | `sas` (EventSat); `cmas/imas/dmas/hmas` for the future multi-satellite scenario |
| **Representation** = substrate × action space | `symb · rl · hrl · llm-s · llm-a · hllm-s · hllm-a` (7 cells) |
| **Operational paradigm** | `conventional` · `ag` · `ao` · `ah` (`ah` is dual-core: onboard + ground) |

EventSat·SAS yields **32 experiments**, named `eventsat_sas_<paradigm>_<rep>` (and
`eventsat_sas_ah_<onboard>_<ground>` for the dual-core hybrid). See
[docs/morphological_matrix.md](docs/morphological_matrix.md) for the full framework, naming, and the
14 metrics.

### Current Status

Implemented components (the code is being mapped onto the 7-representation framework step by step — see `morphological_matrix.md`):
- **Operational paradigms**: Autonomous Onboard (per-step real-time), Autonomous Hybrid (onboard + ground plan + override), Autonomous Ground (algorithmic scheduler, pass-based), Conventional Ground (human-realistic, one-pass planning delay). Jetson-based onboard cores (RL/hybrid onboard, AO/AH) add a ~7 W Jetson-on draw (`power.onboard_compute_w`) to non-Jetson modes; symbolic onboard runs on the OBC with no overhead.
- **Representation cores** (current `@register` classes): `rule_based_eventsat` (symbolic), `subsymbolic_eventsat` (RL/PPO, Juan Oliver et al. 2025), `llm_eventsat` (single-shot LLM, Rodriguez-Fernandez et al. 2024), `agentic_eventsat` (tool-using loop, CoALA — Sumers et al. 2024). Ground-slot schedule producers are real for symbolic, hybrid LLM, pure LLM, and agentic LLM cells; only HRL/RL scheduler placeholders remain explicitly flagged.
- **`autops train` CLI**: PPO training and online CoALA memory accretion.
- Complete environment simulation (power, 3-pool data pipeline, comms, anomalies, detection); orbital mechanics (analytical + optional Orekit J2, launch lottery); the 14 metrics.
- 683 tests (passing; 23 RL tests skipped without the `rl` extra)

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup
```bash
# Install dependencies (including dev tools and Orekit orbital mechanics)
uv sync --extra dev --extra orbital

# Run the test suite
uv run python -m pytest tests/ -v
```

### Running an Experiment
```bash
# Run EventSat experiments — name = eventsat_sas_<paradigm>_<rep>
#   paradigm: conventional | ag | ao | ah        (ah: _<onboard>_<ground>)
#   rep:      symb | rl | hrl | llm-s | llm-a | hllm-s | hllm-a   (ao: symb/rl/hrl only)
uv run autops run configs/experiments/eventsat_sas_ag_symb.yaml       # symbolic ground planner
uv run autops run configs/experiments/eventsat_sas_ag_llm-s.yaml      # single-shot LLM ground
uv run autops run configs/experiments/eventsat_sas_ah_rl_llm-s.yaml   # RL onboard + LLM ground

# Quick smoke test (1 episode, 100 steps)
uv run autops run configs/experiments/eventsat_sas_ag_symb.yaml --episodes 1 --steps 100
```

### Training Learned-Emergence Variants
```bash
# PPO (RL onboard)
uv run autops train configs/experiments/eventsat_sas_ao_rl.yaml

# Writable CoALA (agentic online learning; memory accretes at runtime)
uv run autops train configs/experiments/eventsat_sas_ag_llm-a.yaml
```

### Batch Experiments
```bash
# Generate config combinations from template
uv run autops generate --template configs/experiments/template.yaml

# Quick sanity check
uv run autops batch configs/experiments --episodes 1 --steps 200

# Run batch of specific configurations
uv run autops batch configs/experiments/eventsat_sas_*.yaml --episodes 1 --steps 200  # all EventSat configs

# Run all generated configs or all experiments in a folder
uv run autops batch configs/experiments/generated/
uv run autops batch configs/experiments/
uv run autops batch configs/experiments --episodes 5 --steps 10080
```

### Results Board
Refresh the generated boards and open `data/figures/index.html`:

```bash
uv run python scripts/refresh_board.py
```

Exploratory notebooks are local scratch artifacts and are intentionally not part of the committed repo.
Coverage is opt-in. Use `uv run python -m pytest tests/ --cov=src --cov-report=term` only when coverage is needed; do not leave generated coverage artifacts behind.

## Project Structure

```
autops-demo/
+-- src/
|   +-- core/                 # Runner, config, base interfaces, org, ops, memory, SDA, behaviour
|   +-- eventsat/             # EventSat env, metrics, representations, rewards, trace export
|   +-- flamingo/             # Flamingo env, metrics, symbolic planner
|   +-- orbital/              # Orbital mechanics (eclipse, ground access, Orekit wrapper)
+-- configs/
|   +-- experiments/          # EventSat experiment configs + 1 template
|   +-- scenarios/            # Scenario definitions (eventsat.yaml, ...)
+-- scripts/
|   +-- generate_experiment_configs.py
|   +-- run_batch.py
+-- tests/                    # 683 tests (passing; 23 RL skipped without --extra rl)
+-- docs/
|   +-- morphological_matrix.md  # Canonical O-framework spec (3 components, 32 experiments, 14 metrics)
|   +-- implementations.md    # Implementation registry (components, paper basis, design decisions)
|   +-- architecture.md       # Architecture overview
|   +-- scenarios.md          # Scenario descriptions
+-- data/
|   +-- results/              # Experiment outputs (git-ignored)
|   +-- trained_models/       # PPO policy checkpoints (git-ignored)
|   +-- trained_prompts/      # Prompt-optimized system prompts (git-ignored)
|   +-- writable_memory_state/# WritableMemory for agentic online-learning runs (git-ignored)
|   +-- llm_cache/            # LLM response cache with prompts (git-ignored)
```

## Configuration

Experiments are defined via YAML files validated by Pydantic:

```yaml
experiment_id: "eventsat_sas_ag_symb"
agent_organization: sas
representation: symbolic        # content value; framework cell = symb
operations_paradigm: autonomous_ground
decision_procedure: sda        # held fixed (not a framework component)
behaviour: hand_designed       # held at default
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
uv run python -m pytest tests/ -v

# Specific module
uv run python -m pytest tests/test_eventsat_physics.py -v
```

## Documentation

- [Research Tracker](docs/research-tracker.md) — current EventSat, Flamingo, and world-model scheduling handoffs
- [Operations-System (O) Framework](docs/morphological_matrix.md) — the canonical spec: 3 components, 7 representations, 4 paradigms, 32 EventSat experiments, naming, and the 14 metrics
- [Implementation Registry](docs/implementations.md) — all components, paper basis, design decisions
- [Architecture Overview](docs/architecture.md)
- [Scenario Descriptions](docs/scenarios.md)

## Contact
 
Clemente J. Juan Oliver  
clemente.juan@tum.de

---

**Supported by**:  
AUTOPS project, Bavarian Joint Research Program (BayVFP), MRF-2307-0004.
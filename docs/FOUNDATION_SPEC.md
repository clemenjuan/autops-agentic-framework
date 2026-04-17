# Implementation Foundation for PhD Experimental Framework

**Project:** Custom Modular Architecture for Cognitive Satellite Constellation Autonomy Research
**Researcher:** Clemente J. Juan Oliver, TUM Chair of Spacecraft Systems
**Repository:** autops-demo
**Date:** February 26, 2026 (updated)

***

## 1. Project Overview

### Objective

Build a modular experimental framework to systematically compare cognitive architectures for autonomous satellite constellation management. The framework must support testing combinations of:

- **Agent Organizations**: Centralized, Hierarchical, Distributed
- **Decision Loops**: SDA (Sense-Decide-Act), OODA, ReAct, and others
- **Representations**: Symbolic, Hybrid, Subsymbolic
- **Emergence Modes**: Hand-designed, Learned
- **Operations Paradigms**: Autonomous Hybrid, Autonomous Ground, Conventional Ground
- **Constellation Sizes**: 1, 5, 20-30, 100+ satellites


### Research Questions

**Fundamental Research Question:** How do cognitive architecture and agent organization choices shape the performance trade-space of autonomous constellation management, and how does this trade-space evolve as constellation scale and structural complexity grow?

Scalability is framed as a **2D space**: *constellation size* (1 → 500+ satellites) × *structural complexity* (centralized → distributed, with super-linear effort scaling). The following interconnected sub-questions operationalize the fundamental RQ:

**RQ1 — Cognitive Architecture**
- How do **decision-making loops** (SDA, OODA, ReAct, LATS), **knowledge representations** (symbolic, subsymbolic, hybrid), and **degree of emergence** affect key performance metrics (utility, latency, robustness, resource efficiency, operator load, and explainability)?
- Can Pareto frontiers between competing objectives (e.g., utility vs. resource efficiency vs. operator interventions) be characterised for different cognitive architecture configurations?
- Which cognitive architecture patterns offer the most favourable trade-offs for which operational scenarios?

**RQ2 — Agent Organization**
- How do different agent organizations (single centralized agent, one agent per satellite, hierarchical, fully distributed) affect the performance/robustness trade-off under identical cognitive components?
- Are certain cognitive architectures better matched to certain organizations (e.g., emergent subsymbolic agents in distributed constellations vs. hybrid agents in hierarchical setups)?
- How does the choice of architecture family determine the type and degree of explainability available to human operators — and how does this interact with mission safety requirements?
- **Testable hypothesis (Kim et al. 2025 [FVFQ73RF]):** Satellite mode selection is sequential constraint satisfaction → centralized org should outperform distributed. Capability saturation effect predicts multi-agent overhead negates gains once single-agent baselines exceed ~45%.

**RQ3 — Scale & Complexity**
- How do different cognitive and agent architectures degrade or adapt as constellation size, task load, and constraint density grow (e.g., from 5 to 500 satellites)?
- How does structural complexity — increasing from centralized towards distributed topologies, with super-linear effort scaling — interact with the performance trade-offs of different architecture families?
- Do different architectures exhibit fundamentally different composability trade-offs as scale grows (e.g., integrating heterogeneous cognitive components without emergent negative side effects)?
- Can scaling laws be derived jointly over constellation size and structural complexity and converted into architecture-selection heuristics for a target mission profile?


### Key Design Principles

1. **Orthogonality**: Each dimension (organization, loop, representation, emergence, operations paradigm) is independent
2. **Modularity**: Components can be swapped without affecting others
3. **Reproducibility**: Configuration-driven experiments with seed control
4. **Fair Comparison**: Same environment and metrics for all variants
5. **Scientific Rigor**: Implementations follow established research papers

***

## 2. System Architecture

![AUTOPS overall system architecture diagram](images/autops-overall-system-architecture.svg)


***

## 3. Directory Structure

```
autops-demo/
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── satellite_env.py         # Core environment (abstract)
│   │   ├── orbital_mechanics.py     # Legacy (unused)
│   │   ├── orbital/                 # Orbital mechanics module
│   │   │   ├── __init__.py
│   │   │   ├── propagator.py        # Orekit wrapper (optional dep)
│   │   │   ├── eclipse.py           # Eclipse/sunlight computation
│   │   │   ├── ground_access.py     # Ground station visibility
│   │   │   └── context.py           # OrbitalContext pre-computation
│   │   └── scenarios/
│   │       ├── __init__.py
│   │       └── eventsat_env.py      # EventSat environment
│   ├── agent_organization/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract AgentOrganization
│   │   ├── single_agent_system.py   # SAS (Kim et al. 2025)
│   │   ├── centralized_mas.py       # Centralized MAS
│   │   ├── decentralized_mas.py     # Decentralized MAS (stub, N≥3)
│   │   ├── independent_mas.py       # Independent MAS (stub, N≥3)
│   │   └── hybrid_mas.py            # Hybrid MAS (stub, N≥3)
│   ├── decision_loop/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract DecisionLoop
│   │   └── README.md                # Implementations follow research papers
│   ├── representation/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract Representation
│   │   └── README.md                # Implementation guidelines
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── fixed_memory.py          # Single fixed memory design
│   ├── emergence/
│   │   ├── __init__.py
│   │   └── controller.py            # Emergence mode manager
│   ├── operations/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract OperationsParadigm
│   │   ├── autonomous_hybrid.py     # Autonomous ops (onboard/ground)
│   │   └── conventional_ground.py   # Traditional ground-based ops
│   ├── tools/
│   │   ├── __init__.py
│   │   └── README.md                # Tools defined per operational scenario
│   └── orchestration/
│       ├── __init__.py
│       ├── experiment_runner.py     # Main orchestrator
│       ├── config_loader.py         # YAML configuration
│       ├── metrics_collector.py     # Metrics framework (abstract)
│       └── analysis.py              # Statistical analysis
├── configs/
│   ├── experiments/
│   │   ├── template.yaml            # Configuration template
│   │   ├── eventsat_cen_sda_symb_hd_ah.yaml  # autonomous hybrid (reference)
│   │   └── eventsat_cen_sda_symb_hd_cg.yaml  # conventional ground
│   └── scenarios/
│       └── eventsat.yaml            # EventSat scenario parameters
├── tests/
│   ├── test_environment.py
│   ├── test_agent_organization.py
│   ├── test_decision_loops.py
│   ├── test_representations.py
│   ├── test_operations_paradigm.py
│   └── test_orchestration.py
├── data/
│   ├── results/                     # Experiment outputs
│   └── trained_models/              # Learned policies (if applicable)
├── notebooks/
│   └── analysis.ipynb               # Experiment analysis
├── scripts/
│   ├── generate_experiment_configs.py
│   └── run_batch.py
├── docs/
│   ├── architecture.md              # Detailed architecture documentation
│   ├── metrics.md                   # Metrics definitions and rationale
│   ├── scenarios.md                 # Operational scenarios
│   └── implementation_guide.md      # Step-by-step implementation
├── pyproject.toml
├── uv.lock
└── README.md
```


***

## 4. Core Component Specifications

### 4.1 Abstract Interfaces

All components must define clear abstract base classes before implementation.

#### Satellite Environment

**File:** `src/environment/satellite_env.py`

**Purpose:** Unified environment for all experiments. Handles orbital mechanics, task generation, and constraint management. Must be operational-scenario-agnostic at the base level.

**Key Methods:**

- `reset()`: Initialize constellation state
- `step(actions)`: Execute one time step
- `get_observation()`: Return current observation
- `get_metrics()`: Return current performance metrics

**Note:** Specific task types, rewards, and constraints depend on chosen operational scenario (large-scale constellations, communications, or SSA). These will be defined in scenario-specific subclasses.

***

#### Agent Organization

**File:** `src/agent_organization/base.py`

**Purpose:** Abstract coordination patterns between agents. Controls how observations are distributed and actions are aggregated.

**Key Methods:**

- `distribute_observation(env_obs)`: Map environment observation to agent-specific observations
- `collect_actions(agent_actions)`: Aggregate agent actions for environment
- `get_agents()`: Return all agents in the organization

**Formal definition** (Kim et al. 2025 [FVFQ73RF]):

An agent system is defined as **S = (A, E, C, Ω)** where:
- **A = {a₁, …, aₙ}** — set of agents (n ≥ 1); each aᵢ = (Φᵢ, Aᵢ, Mᵢ, πᵢ) with reasoning policy Φᵢ (LLM or planner), action space Aᵢ, internal memory Mᵢ, decision function πᵢ: H → Aᵢ
- **E** — shared environment (satellite constellation simulation)
- **C** — communication topology (defines information flow between agents)
- **Ω** — orchestration policy (how sub-agent outputs are aggregated, whether overrides are possible, termination conditions)

When |A| = 1 → **Single-Agent System (SAS)**; |A| > 1 → **Multi-Agent System (MAS)**.

**Topology mapping to AUTOPS Organization dimension:**

| Topology (Kim et al.) | AUTOPS config value | Naming suffix | C definition | Ω policy | Complexity |
|---|---|---|---|---|---|
| Single-Agent            | `sas`               | `sas`  | — (one reasoning locus)    | direct                  | O(k) |
| Centralized MAS         | `centralized_mas`   | `cmas` | orchestrator → sub-agents  | hierarchical            | O(rnk) |
| Decentralized MAS       | `decentralized_mas` | `dmas` | all-to-all peer exchange   | consensus               | O(dnk) |
| Independent MAS         | `independent_mas`   | `imas` | agent-to-aggregator only   | synthesis_only          | O(nk) |
| Hybrid MAS              | `hybrid_mas`        | `hmas` | star + peer edges          | hierarchical + lateral  | O(rnk + pn) |

The left column ("AUTOPS config value") is the canonical name accepted by
`src/orchestration/config_loader.py`; the "Naming suffix" column is the abbreviation used in
experiment filenames (`eventsat_<suffix>_...`). `dmas`, `imas`, and `hmas` are registered
in the validator but their implementations raise `NotImplementedError` — they are deferred
to constellation scenarios (N ≥ 3, Flamingo onwards), see §Implementations below.

where k = reasoning iterations, r = orchestration rounds, d = debate rounds, n = agents.

**Empirical scaling effects** (Kim et al. 2025, 180-configuration controlled study):

1. **Capability saturation** (β̂ = −0.404, p<0.001): Once single-agent baseline exceeds ~45% task accuracy, adding agents yields diminishing or *negative* returns. Coordination overhead exceeds improvement potential. For AUTOPS: if symbolic/hybrid representations already perform well, distributed org may degrade utility.

2. **Topology-dependent error amplification**: Independent agents amplify errors 17.2× through unchecked propagation; centralized coordination contains this to 4.4× via validation bottlenecks. Prediction for AUTOPS: centralized org is safer for anomaly-handling.

3. **Task-type dependency — critical for satellite ops**: Coordination benefits are task-contingent:
   - Parallelisable tasks (financial reasoning): Centralized +80.8%
   - Dynamic exploration (web navigation): Decentralized +9.2%
   - **Sequential constraint satisfaction (planning):** *Every* multi-agent variant degraded performance −39% to −70%

   Satellite mode selection is a **sequential constraint satisfaction task** (hard ordering: charge → observe → compress → detect → send → communicate). This predicts **centralized org will outperform distributed** for EventSat — a testable hypothesis for RQ2.

4. **Intelligence-coordination alignment**: Higher-capability representations (Phase 4a/4b/4c LLMs) need the *right* topology to benefit. Wrong org structure negates capability gains. This makes the **Organization × Representation interaction a first-class RQ2 research question**.

**Architecture selection rule** (87% accuracy on held-out configurations): Optimal topology depends on measurable task properties — decomposability, tool complexity, sequential depth — not simply on "more agents". AUTOPS can derive analogous selection heuristics from its experimental matrix.

**Implementations:**

Full Kim et al. (2025) [FVFQ73RF] taxonomy — "Towards a Science of Scaling Agent Systems":

- `SingleAgentSystem` (SAS): |A|=1, C undefined, Ω direct. 36 `eventsat_sas_*` configs covering 3 loops × 4 representations × 3 ops paradigms.
- `CentralizedMAS`: Orchestrator + local agents, C = star, Ω = hierarchical. 12 `eventsat_cmas_*_ah.yaml` configs (3 loops × 4 representations, AH only — CG/AG degenerate at N=1 since ground already acts as the strategic layer).
- `DecentralizedMAS`: All-to-all peer exchange, C = all-to-all, Ω = consensus. Stub — degenerate at N=1; reserved for constellation scenarios (N≥3).
- `IndependentMAS`: No inter-agent communication, C = ∅. Stub — meaningful only with subsystem-level agents (ADCS/payload/comms) or N≥3 satellites.
- `HybridMAS`: Heterogeneous mixed topology. Stub — reserved for complex multi-cluster constellations.

***

#### Decision Loop Engine

**File:** `src/decision_loop/base.py`

**Purpose:** Abstract decision-making pattern defining temporal control flow. Each loop type follows specific research papers.

**Key Methods:**

- `process(observation, memory)`: Main decision cycle, returns (action, updated_memory)
- `get_metrics()`: Return decision loop metrics

**⚠️ Critical:** Decision loop implementations must strictly follow scientific papers. Do not predefine specific steps—implementations will be created step-by-step following literature.

**Implemented loops:**

- **SDA**: Linear reactive pattern — single-pass sense-decide-act, no iteration.
- **OODA**: Fixed four-phase structure (Observe-Orient-Decide-Act) with situation
  classification, Case-Based Reasoning, urgency scoring, and feedback loops.
- **ReAct**: Iterative Thought-Action-Observation cycle (Yao et al. 2023). Adds
  explicit reasoning traces via `representation.reason()` and grounding validation
  before executing each action. Converges or falls back to charging.
  Note: Deployed frontier systems (Claude, GPT-4o) standardize on ReAct-style
  reason-act-observe cycles in their orchestration layers.

**Note on CoALA (Sumers et al. 2024):** CoALA ("Cognitive Architectures for Language
Agents") is an **architecture blueprint**, not a decision loop. It defines an LLM-orchestrated
agentic system with memory architecture (working, episodic, semantic, procedural) and
action space decomposition that combines decision loop + representation concerns. In this
framework, CoALA is implemented as a hybrid **representation type** (`agentic_eventsat`)
that internalizes its own reasoning cycle, rather than as a separate decision loop.

***

#### Representation Module

**File:** `src/representation/base.py`

**Purpose:** How knowledge and decisions are represented. This is what fills the decision loop pattern.

**Key Methods:**

- `encode_observation(obs)`: Transform observation to internal representation
- `select_action(state, memory)`: Core decision-making logic
- Additional methods as required by specific decision loops

**Cognitive paradigms** (Brooks 1991, Colelough & Regli 2025, Navarro 2025):

- **Symbolic**: Explicit declarative knowledge — rules, planners, constraint solvers. Knowledge lives in explicit world models, rules, ontologies.
- **Subsymbolic**: Implicit learned representations from raw data — RL policies, DNNs, base LLMs (without symbolic layer). Knowledge lives in network weights and embeddings.
- **Hybrid**: Integration of symbolic + subsymbolic (Kahneman System 1/2). Includes LLM + tools/memory, DNN + logic constraints, agentic systems. An LLM alone is subsymbolic; an LLM combined with tools, memory structures, or symbolic constraints becomes hybrid.

**Representation taxonomy — paradigm to implementation mapping:**

| Paradigm | Where Knowledge Lives | Reasoning Structure | Examples | Our Implementations |
|----------|----------------------|---------------------|----------|---------------------|
| Symbolic | Explicit rules, ontologies, world models | Deductive, constraint-based | Expert systems, planners, production rules | `rule_based_eventsat`, `schedule_based_eventsat`, `conventional_schedule_eventsat` |
| Subsymbolic | Network weights, embeddings | Statistical, distributed | RL policies, DNNs, base LLMs (no symbolic layer) | `subsymbolic_eventsat` (Phase 4b) |
| Hybrid | Both explicit + implicit | Combines fast intuition (System 1) with deliberate reasoning (System 2) | LLM + tools/memory, DNN + logic, agentic systems | `llm_eventsat` (Phase 4a), `agentic_eventsat` (Phase 4c) |

**Key rule for implementations:** An LLM alone is subsymbolic (implicit distributed representations in weights). An LLM combined with tools, memory structures, or symbolic constraints becomes hybrid (explicit symbolic layer on top of subsymbolic core). This follows Kahneman's System 1/2 and the neuro-symbolic AI literature (Colelough & Regli 2025). The agentic pattern (LLM + memory + tools + reasoning loop) is an **implementation of the hybrid paradigm**, not a new paradigm.

**Representation subtypes** (e.g., `llm_eventsat`, `agentic_eventsat`, `subsymbolic_eventsat`) are design choices *within* paradigms, specified via `representation_config.type` in YAML. The top-level `representation` field remains one of: `symbolic`, `subsymbolic`, `hybrid`.

**Phase 3 orthogonality observation:** With deterministic symbolic representations, some loop × representation combinations (e.g., SDA vs ReAct with `rule_based_eventsat`) produce identical decisions because the rules always return the same output regardless of reasoning iteration. The ReAct loop's reason→act→observe cycle converges in one iteration since there is no "reasoning" to iterate on. With hybrid/subsymbolic representations, these combinations **will** produce meaningfully different outcomes: LLMs generate different reasoning traces across iterations, and RL policies have stochastic exploration. This is expected behaviour, not a design flaw — it validates that the loop dimension becomes load-bearing when the representation has non-deterministic reasoning.

**Note:** Same representation can work with different decision loops. The representation provides the "what," the decision loop provides the "when/how."

***

#### Memory System

**File:** `src/memory/fixed_memory.py`

**Purpose:** Fixed memory design accessible by all cognitive architectures. All experimental variants have access to the same information—only the representation differs.

**Design:** Single unified memory structure providing:

- Current constellation state
- Historical information (sliding window)
- Task queue and completion history
- Resource budgets

**Note:** Memory structure is fixed across all experiments to ensure fair comparison. Representation modules determine how to use this information.

***

#### Emergence Controller

**File:** `src/emergence/controller.py`

**Purpose:** Controls whether decision-making logic is hand-designed or learned from experience.

**Key Method:**

- `get_representation(repr_type, decision_loop_type)`: Factory method returning configured representation

**Modes:**

- **Hand-designed**: Logic designed by human experts (rules, prompts, models)
- **Learned**: Logic learned from training data (RL policies, learned heuristics)

***

#### Operations Paradigm

**File:** `src/operations/base.py`

**Purpose:** 5th morphological matrix dimension. Controls how human-machine operations are structured — who sees what information, when actions can be applied, and how authority is split between ground and onboard systems. The paradigm sits between the agent organization and the environment, filtering observations and gating actions.

**Key Methods:**

- `filter_observation(full_observation, step)`: What the agent is allowed to see (full state vs. stale ground knowledge)
- `can_act(step, ground_pass_active)`: Whether the agent can issue commands at this step
- `process_action(action, step, ground_pass_active)`: Buffer, delay, or pass through actions
- `update_ground_knowledge(full_observation, step)`: Update ground state after telemetry downlink

**Implementations:**

- `AutonomousHybrid`: Dual onboard/ground operations. Onboard autonomy (rules, small DNNs) operates with full real-time state every step. Ground systems (LLM, planners) prepare detailed analysis and plans between passes using last-received telemetry (Rossi et al. 2023: tactical planning cycle; Sellmaier et al. 2022 §16.4: offline preparation), uplinked during next contact. Onboard can override ground plans for fault detection or opportunistic events. In EventSat, no LLM runs onboard — only small DNNs are feasible.
- `AutonomousGround`: Algorithmic ground-based operations. Ground only sees downlinked telemetry (stale between passes). Algorithmic scheduler prepares optimal schedules **between passes** using last-received telemetry; schedule uplinked during next contact. Satellite executes schedule between passes with zero onboard autonomy. No planning delay or cognitive constraints — algorithmic ideal of ground ops.
- `ConventionalGround`: Realistic human flight dynamics team operations. Same information constraints as AutonomousGround, but with a **one-pass planning delay**: schedule planned after pass N is uploaded at pass N+1. Models the real planning cycle where operators analyse telemetry and plan schedules between passes (not during them). Paired with `ConventionalScheduleEventSat` representation which adds human cognitive constraints (conservative margins, limited horizon, shift handovers). Two-buffer design: `_active_schedule` (executing) + `_planned_schedule` (waiting for next pass upload).

**Supporting Data:**

- `GroundKnowledge`: Dataclass representing what ground operators know — battery SoC, stored data, mode, health status, staleness counter. Updated only during communication passes.

***

### 4.2 Experiment Orchestration

**File:** `src/orchestration/experiment_runner.py`

**Purpose:** Configuration-driven experiment execution with comprehensive logging and reproducibility.

**Key Features:**

- Load YAML configuration
- Initialize all components based on config
- Execute episodes with metrics collection
- Save results with full provenance
- Support batch experiment execution

**Critical:** All experimental choices must be configurable via YAML—no hardcoded decisions.

***

### 4.3 Matrix Coverage and Comparison Scope

The 5D morphological matrix admits many theoretically possible cells, but not
every cell is scientifically meaningful. Phase 5 instantiates **84 EventSat
configs** (48 hand-designed `_hd_` + 36 learned `_le_ / _lep_ / _lec_`). The
remaining cells are excluded by principle, not by oversight:

| Exclusion | Principle | Consequence |
|---|---|---|
| `symb_le` / `symb_lep` / `symb_lec` | Symbolic representations are deterministic rule chains; they have no learnable parameters or adjustable prompts. | Symbolic appears only with `_hd_`. |
| `subm_hd` | PPO-trained policies are *learned by construction*; a hand-designed subsymbolic agent is not coherent in this framework. | Subsymbolic appears only with `_le_` (PPO). |
| `hybr_lec` / `subm_lep` / `subm_lec` | Writable CoALA (`_lec_`) presupposes *agentic self-reflection* over semantic/episodic stores; plain LLM-only (`hybr`) does not assume agentic behavior, and PPO has no prompts/memories to accrete into. | `_lec_` appears only with `agnt`; `_lep_` only with `hybr` and `agnt`. |
| `cmas × {ag, cg}` | Centralized MAS coordination is degenerate at N=1 when ground acts as the strategic layer — the orchestrator has no sub-agents to coordinate. | CMAS only appears with `ah`. |
| `dmas` / `imas` / `hmas` (any) | Peer-to-peer, independent, and hybrid MAS topologies require N ≥ 3 satellites to be non-trivial. | Deferred to constellation scenarios (Flamingo and later). |

**Comparison scope.** Because each representation family has a *different*
learning mechanism (PPO for subsymbolic, prompt optimization for LLM/hybrid,
writable-CoALA for agentic), **emergence is compared within a representation
family, not across**. Concretely:

- *Within-family comparisons* (supported): `symb_hd` vs `subm_le`; `agnt_hd`
  vs `agnt_lep` vs `agnt_lec`; `hybr_hd` vs `hybr_lep`.
- *Across-family comparisons* (supported): `symb_hd_ah` vs `hybr_hd_ah` vs
  `agnt_hd_ah` vs `subm_le_ah` — holding emergence as "representation-appropriate
  default" rather than a shared learning mechanism.
- *Not supported by this matrix*: a single "learned-everywhere" axis (because
  no unified emergence mechanism exists across representation families).

The **memory invariant** reinforces this scoping: `FixedMemory` for all `_hd_`
and non-CoALA `_le_/_lep_` variants; `WritableMemory` only for `_lec_`, which
is compared against its `_agnt_hd_` counterpart (same representation, same
loop, same ops paradigm — emergence is the only varied axis).

***

## 5. Configuration System

### Configuration Template

**File:** `configs/experiments/template.yaml`

```yaml
# Experiment Identification
experiment_id: "exp_XXX_description"
description: "Brief description of experimental configuration"

# Reproducibility
seed: 42

# Morphological Matrix Dimensions
agent_organization: "sas"          # sas | centralized_mas | decentralized_mas | independent_mas | hybrid_mas
decision_loop: "sda"               # sda | ooda | react
representation: "symbolic"         # symbolic | subsymbolic | hybrid   (paradigm; nested `representation_config.type` picks the implementation)
emergence_mode: "hand_designed"    # hand_designed | learned            (if `learned`, set `emergence_config.mechanism`)
operations_paradigm: "autonomous_hybrid"  # autonomous_hybrid | autonomous_ground | conventional_ground

# Configuration for each component
agent_organization_config:
  # Specific parameters for chosen organization

decision_loop_config:
  # Specific parameters for chosen decision loop

representation_config:
  # Specific parameters for chosen representation

emergence_config:
  # Parameters for loading/initializing representation

operations_paradigm_config:
  # Parameters for chosen operations paradigm
  # For conventional_ground: default_mode, command_sequence_horizon

# Environment Configuration
environment:
  constellation_size: 5
  timestep_seconds: 60
  scenario: "to_be_defined"  # space_data_centers | communications | ssa
  scenario_config: {}

# Memory Configuration (fixed across all experiments)
memory_config:
  # Parameters for fixed memory design

# Execution Parameters
num_episodes: 100
max_steps: 1440  # 24 hours at 1-minute timesteps

# Metrics Configuration
metrics:
  enabled:
    - utility
    - latency
    - robustness
    - resource_efficiency
    - operator_load
    - explainability
    - scale_complexity        # tracked as a function of constellation_size × complexity_index
  collection_frequency: "per_step"  # per_step | per_episode

# Output Configuration
output_dir: "data/results/${experiment_id}"
save_checkpoints: false
log_level: "INFO"
```


### Configuration Validation

All configurations must be validated on load:

- Required fields present
- Valid choices for morphological dimensions
- Constellation size within limits
- Scenario configuration complete

***

## 6. Metrics Framework

### 6.1 Core Metrics

The following metrics must be collected, but **specific implementations require deeper study**:

#### 1. Utility

**Definition:** Weighted composite of observation and downlink achievement, penalised by anomaly rate.

**Formula (EventSat):** `u = w_obs × (obs_hours / scaled_obs_target) + w_dl × (dl_mb / scaled_dl_target) - w_anomaly × anomaly_rate`

Targets are scaled from the 90-day mission duration to the episode length. Weights are configurable via experiment YAML (`utility_weights` section).

**Rationale:** Primary performance metric—does the system accomplish its mission?

***

#### 2. Latency

**Definition:** Decision-making computational time

**Rationale:** Real-time constraints in space operations—decisions must be timely

**Measurement:** Wall-clock time per decision cycle

***

#### 3. Robustness

**Definition:** Consistency of mission performance across varying initial conditions (orbit insertion geometry, launch lottery RAAN/ArgP/TA randomisation). Measured as the coefficient of variation of utility across episodes:

> CV = σ(utility) / μ(utility)

A lower CV means the architecture delivers reliably similar results regardless of the specific orbit geometry in a given episode, indicating true robustness to insertion uncertainty.

**Rationale:** An architecture that scores high utility on average but collapses under unfavourable orbital geometries is not robust. The launch lottery in the EventSat scenario explicitly randomises insertion parameters per episode so that Monte Carlo averaging over episodes captures rideshare uncertainty — CV across those episodes is therefore a direct measure of robustness.

**Measurement:** `robustness_cv = std(utility_per_episode) / mean(utility_per_episode)`. Computed at experiment level; lower is better.

**Note:** Within-episode anomaly recovery time (`robustness_mean_recovery_steps`) is tracked separately as a secondary diagnostic metric but is not the primary robustness indicator.

***

#### 4. Resource Efficiency

**Definition:** Mission utility per Watt-hour consumed: `utility / total_energy_consumed_wh`.

Energy is estimated from battery state-of-charge deltas (`energy_consumed_wh = max(0, soc_delta × battery_capacity_wh)` per step, summed across the episode).

**Rationale:** Satellites have limited power, data bandwidth, computation

***

#### 5. Operator Load

**Definition:** Fraction of decision steps requiring an environment safety override: `safety_overrides / n_steps`.

A `safety_override` tracks environment-enforced safety interventions (forced mode transitions). Value in [0, 1]; lower is better.

**Rationale:** Autonomy goal is reducing operator burden

***

#### 6. Scale & Complexity

**Definition:** Performance degradation across the 2D scalability space: *constellation size* (number of satellites) × *structural complexity* (topology, from centralized to fully distributed)

**Rationale:** RQ3 directly addresses how architectures degrade or adapt along both axes. Structural complexity follows super-linear effort scaling distinct from raw satellite count.

**Measurement:** Track all metrics as a function of both `constellation_size` and a `complexity_index` capturing topology and inter-agent coordination overhead. Derive joint scaling laws from the resulting surfaces.

***

#### 7. Data Downlink Efficiency

**Definition:** Fraction of the maximum achievable downlink capacity actually used in an episode:

> data_downlink_efficiency = downlinked_mb / max_achievable_downlink_mb

**Rationale:** An architecture may achieve high utility (observations taken) but fail to downlink the data due to poor scheduling of communication passes. This metric separates observation performance from ground-contact exploitation and directly addresses the data pipeline bottleneck described in Proposal Section 6.1.

**Measurement:** Computed per episode from total data downlinked vs. the maximum achievable given the ground passes that occurred. Value in [0, 1]; higher is better.

***

#### 8. Explainability

**Definition:** Fraction of decision steps accompanied by a rationale string: `decisions_with_rationale / n_steps`.

**Rationale:** Mission safety requirements and human-machine trust demand that operators understand why the system acted as it did, not just what it did. RQ2 specifically asks how architecture choice determines the type and degree of explainability available.

**Measurement:** The `has_rationale` flag is set per decision step when the representation produces a rationale string. Symbolic representations (`rule_based_eventsat`) yield 1.0; subsymbolic and hybrid representations yield lower values depending on their reasoning trace coverage.

**Note:** The qualitative distinction between intrinsic explainability (symbolic rule traces), post-hoc interpretability (learned policy inspection), and emergent explainability (LLM reasoning chains) remains relevant for interpretation, but the quantitative metric is the simple ratio above.

***

### 6.2 Metrics Collection Interface

**File:** `src/orchestration/metrics_collector.py`

```python
class MetricsCollector(ABC):
    """Abstract metrics collection framework"""
    
    @abstractmethod
    def collect_step_metrics(self, env_state, actions, rewards, info) -> StepMetrics:
        """Collect metrics for single timestep"""
        pass
    
    @abstractmethod
    def aggregate_episode_metrics(self, step_metrics) -> EpisodeMetrics:
        """Aggregate step metrics into episode summary"""
        pass
    
    @abstractmethod
    def compute_statistics(self, episode_metrics) -> Statistics:
        """Compute statistical measures across episodes"""
        pass
```

**Note:** Specific metric implementations will be developed following literature review and theoretical justification. This is a PhD-level research contribution.

***

## 7. Implementation Phases

### Phase 1: Foundation (Weeks 1-4)

**Goal:** Abstract interfaces and minimal working system

**Deliverables:**

1. All abstract base classes defined
2. Configuration system operational
3. Experiment orchestrator skeleton
4. Environment base class (no specific scenario yet)
5. Test framework established

**Validation:** Can load config, instantiate abstract classes, run empty experiment loop

***

### Phase 2: First Complete Path (Weeks 5-8)

**Goal:** One fully working configuration (simplest case)

**Deliverables:**

1. Choose operational scenario (researcher decision)
2. Implement scenario-specific environment
3. Implement SingleAgentSystem (SAS)
4. Implement one decision loop (researcher chooses which)
5. Implement one representation (researcher chooses which)
6. End-to-end experiment execution

**Validation:** Complete experiment produces valid metrics for chosen configuration

***

### Phase 3: Morphological Expansion (Weeks 9-16)

**Goal:** Implement alternative configurations systematically

**Implemented (Phase 3):**

- **Decision loops**: OODA (Boyd/Miller/Hartmann), ReAct (Yao et al. 2023, Li 2025)
- **Operations paradigms**: Three-tier taxonomy — AutonomousHybrid (existing),
  AutonomousGround (renamed, algorithmic scheduler), ConventionalGround (new,
  human-realistic with one-pass delay per Sellmaier et al. 2022, ECSS-E-ST-70C)
- **Representations**: ScheduleBasedEventSat (Phase 2 → upgraded OODA + ReAct-capable),
  ConventionalScheduleEventSat (new, human cognitive constraints via Endsley 1995)
- **Reasoning interface**: `Representation.reason()` optional method added to base;
  overridden by RuleBasedEventSat and ScheduleBasedEventSat for ReAct Thought step
- **Experiment configs**: 9 configs covering all SDA/OODA/ReAct × AH/AG/CG combinations
- **Tests**: 299 tests total (added 140 new tests for new components)

**Planned (next phases):**

- Scale constellation size: 1 → 5 → 20 → 100

**Note:** Each new component requires theoretical justification and validation against baselines

***

### Phase 4: Hybrid & Subsymbolic Representations + Learned Emergence

**Goal:** Expand representation dimension (hybrid, subsymbolic) and emergence mode (learned)

**Completed:**

1. Phase 4a: LLM hybrid representation — `llm_eventsat`, 9 configs, 48 tests (Rodriguez-Fernandez et al. 2024, Li 2025)
2. Phase 4b: Subsymbolic/RL representation — `subsymbolic_eventsat` with PPO training pipeline, Gymnasium wrapper, 25D obs space, MultiDiscrete action space, 9 configs, 67 tests (Oliver et al. EUCASS 2025, Hamilton et al. 2025, BSK-RL)

3. Phase 4c: Agentic hybrid representation — `agentic_eventsat`, CoALA-style multi-step reasoning with 6 domain tools, 9 configs, 76 tests (Sumers et al. 2024, Sapkota et al. 2026, Li 2025)
4. Phase 4d: Inference gating by operations paradigm — AG/CG skip LLM inference between passes (Rossi et al. 2023, Sellmaier et al. 2022). LLM cache stores prompts. Decision trace JSONL (DEBUG mode). Cross-cutting architecture documentation in `implementations.md`. 493 total tests.

### Phase 5: Kim et al. 2025 Taxonomy + Learned-Emergence for LLM Representations

**Goal:** Align agent-organization taxonomy with literature; define and implement learned-emergence for all LLM-based representations.

**Completed:**

1. **Taxonomy rename** (Kim et al. 2025 [FVFQ73RF] "Towards a Science of Scaling Agent Systems"):
   - SAS (|A|=1, former "centralized") — `SingleAgentSystem`
   - CentralizedMAS (star topology, former "hierarchical") — `CentralizedMAS`
   - DecentralizedMAS (all-to-all, former "distributed") — stub
   - IndependentMAS (C=∅) — stub (future constellation scenarios)
   - HybridMAS (heterogeneous) — stub (future constellation scenarios)
   - 36 SAS `eventsat_sas_*` + 12 CMAS `eventsat_cmas_*` configs.

2. **`emergence_config.mechanism` field** in `ExperimentConfig` with cross-field validators:
   - `ppo` → requires `representation=subsymbolic`
   - `prompt_optimized` → requires `representation=hybrid`
   - `writable_coala` → requires `representation=hybrid` + `representation_config.type=agentic_eventsat`

3. **WritableMemory** (`src/memory/writable_memory.py`) — CoALA §3 writable semantic + episodic stores on top of FixedMemory. Used exclusively by `_lec_` configs. Persistence via JSON. `reset()` preserves stores across episodes by design.

4. **CoALA memory-write tools** in `agentic_tools.py`: `memory_write_rule`, `memory_write_episode` — injected into the agentic tool schema only for `writable_coala` configs.

5. **AgenticEventSat learned-emergence branching**: reads `mechanism` at `__init__`:
   - `hand_designed` → unchanged
   - `writable_coala` → WritableMemory + extended system prompt + writable tools
   - `prompt_optimized` → loads `data/trained_prompts/<id>/prompt.txt` with fallback

6. **PromptOptimizer** (`src/emergence/prompt_optimizer.py`) — bootstrap few-shot prompt optimization (Khattab et al. 2023 [DSPy]): loads high-utility trajectories, generates few-shot-augmented candidates, scores on held-out split, writes `prompt.txt` + `metadata.json`. No DSPy runtime dependency.

7. **LLMEventSat `prompt_optimized` wiring** — reads mechanism and loads trained prompt at `__init__`.

8. **`autops train` CLI command** — dispatches: PPO → PPOTrainer; prompt_optimized → PromptOptimizer; writable_coala → prints guidance (online accretion, no pre-training).

9. **36 new learned-emergence configs**: 12 `*_agnt_lep_*`, 12 `*_agnt_lec_*`, 12 `*_hybr_lep_*`. Grand total: 84 experiment configs.

10. 552 tests across 21 modules. Trunk-based commits (no feature branches).

***

## 8. Development Standards

### Technology Stack

- **Python**: 3.11+
- **Dependency Management**: `uv` (existing)
- **Testing**: pytest
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style
- **Formatting**: black, isort, ruff
- **Configuration**: YAML via PyYAML


### uv Configuration

**File:** `pyproject.toml` (extend existing)

```toml
[project]
name = "autops-demo"
version = "0.2.0"
description = "Cognitive Architecture Experiments for Satellite Constellation Autonomy"
requires-python = ">=3.11"

dependencies = [
    # Existing dependencies from current autops-demo
    "flask",
    "openai",
    "ollama",
    "requests",
    "numpy",
    "geopy",
    "aiohttp",
    "toon-format",
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "psycopg2-binary",
    "apscheduler",
    "orekit-jpype",
    "orekitdata",
    
    # New dependencies for experiments
    "pyyaml",           # Configuration
    "pytest",           # Testing
    "pytest-cov",       # Coverage
    "pydantic",         # Data validation
    "networkx",         # Graph topologies for distributed org
    "pandas",           # Results analysis
    "matplotlib",       # Visualization
    "seaborn",          # Statistical plots
    "scipy",            # Statistical tests
]

[project.optional-dependencies]
dev = [
    "black",
    "isort",
    "ruff",
    "mypy",
    "ipython",
    "jupyter",
]

rl = [
    "torch",           # For subsymbolic representations (optional)
    "gymnasium",       # Standard RL interface (optional)
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "--cov=src --cov-report=html --cov-report=term"

[tool.black]
line-length = 100

[tool.isort]
profile = "black"
line_length = 100

[tool.ruff]
line-length = 100
```


### Code Style Example

```python
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


class DecisionLoop(ABC):
    """
    Abstract base class for decision-making patterns.
    
    Decision loops define the temporal control flow of agent reasoning.
    Specific implementations must follow established research papers.
    
    Attributes:
        representation: The representation module providing decision logic
    """
    
    def __init__(self, representation: 'Representation'):
        """
        Initialize decision loop.
        
        Args:
            representation: Representation module to use for decision-making
        """
        self.representation = representation
    
    @abstractmethod
    def process(
        self, 
        observation: 'AgentObservation', 
        memory: 'Memory'
    ) -> Tuple['Action', 'Memory']:
        """
        Execute one decision cycle.
        
        Args:
            observation: Current observation for this agent
            memory: Agent's memory state from previous step
        
        Returns:
            action: Selected action to execute
            memory: Updated memory state
        
        Raises:
            ValueError: If observation format is invalid
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """
        Return decision loop performance metrics.
        
        Returns:
            Dictionary of metric names to values (e.g., latency, iterations)
        """
        pass
```


***

## 9. Testing Strategy

### Unit Tests

- Each abstract base class has test suite
- Each concrete implementation has test suite
- Mock objects for dependencies
- Aim for >80% coverage


### Integration Tests

- Full experiment execution
- Configuration validation
- Metrics collection pipeline
- Results saving/loading


### Validation Tests

- Reproducibility (same seed → same results)
- Scaling (small constellations run correctly)
- Component swapping (different loops with same representation)

**File:** `tests/test_reproducibility.py`

```python
def test_experiment_reproducibility():
    """Same configuration and seed produces identical results"""
    runner1 = ExperimentRunner("configs/experiments/test.yaml")
    runner2 = ExperimentRunner("configs/experiments/test.yaml")
    
    results1 = runner1.run_experiment()
    results2 = runner2.run_experiment()
    
    assert results1.metrics == results2.metrics
```


***

## 10. Documentation Requirements

### docs/architecture.md

Detailed explanation of system architecture, design decisions, and component interactions.

### docs/metrics.md

Theoretical foundation for each metric:

- Definition
- Rationale
- Measurement approach
- Literature justification
- Implementation notes


### docs/scenarios.md

Operational scenario definitions:

- Mission objectives
- Task types
- Constraints
- Reward structure
- Real-world examples


### docs/implementation_guide.md

Step-by-step guide for implementing new components:

- Decision loops: How to follow research papers
- Representations: Guidelines for each type
- Agent organizations: Coordination patterns
- Validation: Testing new components

***

## 11. Operational Scenarios

Three initial concrete scenarios have been selected, ordered by scale and complexity. They provide the progression from single-satellite to large constellation, covering the 2D scalability space of RQ3.

### Scenario 1: EventSat Mission (Phase 2 Starting Point)

**Scale:** 1 satellite | **Complexity:** minimal (single-agent, centralized)

**Description:** TUM's own EventSat satellite. Full access to subsystem models and mission data enables high-fidelity environment modeling. This is the baseline scenario where the cognitive architecture comparison begins.

**Tasks:** Observation scheduling, onboard resource management (power, data, thermal), anomaly response.
**Constraints:** Orbit-dependent visibility windows, power budget, downlink capacity, onboard storage.
**Metrics:** Observation utility, resource efficiency, decision latency, explainability.

**Data Sources:** Internal TUM/AUTOPS mission data — high confidence in accurate modeling.

**Implementation Path:** `src/environment/scenarios/eventsat.py`

---

### Scenario 2: Vyoma Flamingo Constellation (Medium Scale)

**Scale:** Up to 12 satellites (as planned) | **Complexity:** medium (multi-agent, hierarchical/distributed topologies)

**Description:** AUTOPS project partners' Flamingo constellation. Planned to reach 12 satellites, making it a natural medium-scale use case. Collaboration with Vyoma provides realistic mission parameters.

**Tasks:** Space Situational Awareness (SSA) — observation scheduling, sensor tasking across the constellation, coverage optimization, data fusion.
**Constraints:** Inter-satellite link availability, revisit time requirements, sensor FOV, downlink budget.
**Metrics:** Target coverage rate, revisit time, detection probability, coordination overhead, explainability.

**Data Sources:** AUTOPS project collaboration with Vyoma — high confidence in useful modeling data.

**Implementation Path:** `src/environment/scenarios/flamingo.py`

---

### Scenario 3: Large-Scale Constellation

**Scale:** 100+ satellites | **Complexity:** high (fully distributed, heterogeneous)

**Description:** A large-scale constellation scenario at the high end of the scalability spectrum. Represents the high-scale, high-complexity endpoint of the scalability study. The specific mission type will be determined during Phase 2; less mission-specific data is available at this stage, so the scenario will be modeled using published literature and synthetic parameters.

**Tasks:** To be defined based on selected mission type (candidates include resource scheduling, coordination, coverage optimisation).
**Constraints:** Power budget, ISL bandwidth, orbital position affecting latency to ground.
**Metrics:** Mission utility, resource efficiency, coordination overhead, explainability.

**Data Sources:** Literature-based modeling (lower fidelity than Scenarios 1–2); to be refined as the field matures.

**Implementation Path:** TBD (Phase 2)

---

### Progression Strategy

Scenarios are implemented sequentially:
1. **EventSat** → baseline single-satellite cognitive architecture comparison
2. **Flamingo** → medium-scale multi-agent organization comparison
3. **Space Data Centers** → large-scale scalability laws and composability limits

 This progression directly maps to the RQ3 scalability study across both constellation size and structural complexity.

***

## 12. Next Steps for AI Code Agents

### Immediate Actions:

1. **Review existing autops-demo structure** to understand current implementation
2. **Create directory structure** as specified above
3. **Define abstract base classes** for all components (no implementations yet)
4. **Set up testing framework** with pytest
5. **Implement configuration system** with YAML loading and validation
6. **Create documentation templates** in `docs/`

### Awaiting Researcher Input:

1. ~~**Operational scenario selection**~~ → **decided**: EventSat, Flamingo, Space Data Centers
2. **First decision loop choice** (which to implement first for EventSat?)
3. **First representation choice** (symbolic | hybrid | subsymbolic?)
4. **Hand-designed logic specifications** (rules, prompts, etc.)
5. **EventSat-specific constraints** (visibility windows, power model, downlink budget)

### Do NOT Implement Yet:

- Specific decision loop implementations (wait for paper-following instructions)
- Representation implementations (wait for specifications)
- Metrics formulas (require theoretical development)
- Environment scenarios (wait for scenario selection)
- Reward functions (scenario-dependent)

***

## 13. Key Principles for Implementation

### 1. Abstract Before Concrete

Define all interfaces before any implementation. Type hints and docstrings are mandatory.

### 2. Configuration Over Code

Every experimental choice must be in YAML configuration files. No hardcoded assumptions.

### 3. Test-Driven Development

Write tests alongside code. Validate each component independently before integration.

### 4. Scientific Rigor

Implementations of cognitive architectures must strictly follow published research papers. Do not invent steps.

### 5. Incremental Complexity

Start with simplest case (1 satellite, 1 decision loop, hand-designed). Add complexity systematically.

### 6. Documentation First

Document design decisions before implementation. Every component needs rationale.

### 7. Reproducibility

Every experiment must be fully reproducible from configuration file and random seed.

***

## 14. References for Implementation

### Existing Codebase

- **autops-demo repository**: [https://github.com/clemenjuan/autops-demo](https://github.com/clemenjuan/autops-demo)
- Reuse: Orekit integration, tool interfaces, data pipeline concepts


### Scientific Papers

**Cognitive paradigm taxonomy:**
- Brooks (1991) "Intelligence Without Representation", Artificial Intelligence 47(1)
- Colelough & Regli (2025) "Neuro-Symbolic AI in 2024: A Systematic Review"
- Navarro (2025) "Enhancing Cognitive Functions in LLMs Towards AGI"

**Decision loops:**
- ReAct: Yao et al. (2023) "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR
- Tree of Thoughts: Yao et al. (2023) "Tree of Thoughts: Deliberate Problem Solving with LLMs"

**Representations:**
- CoALA (architecture blueprint): Sumers et al. (2024) "Cognitive Architectures for Language Agents", TMLR
- LLM for sat ops: Rodriguez-Fernandez et al. (2024) "Language Models are Spacecraft Operators"
- LLM agent: Li (2025) "Developing AI Agents for Satellite Operations"
- RL for sat scheduling: Wang et al. (2022) "DRL-based Autonomous Mission Planning for AEOSs"

**Agent organization & multi-agent scaling:**
- Kim et al. (2025) "Towards a Science of Scaling Agent Systems" [FVFQ73RF] — formal topology taxonomy (S = (A,E,C,Ω)), 180-config controlled study, 3 scaling effects, architecture selection rules (87% accuracy)

**Agentic AI:**
- Sapkota et al. (2026) "AI Agents vs. Agentic AI: A Conceptual Taxonomy"
- Masterman et al. (2024) "The Landscape of Emerging AI Agent Architectures"
- V et al. (2026) "Agentic AI: Architectures, Taxonomies, and Evaluation of LLM Agents"

**Note:** See Zotero library for full references. Each implementation must cite its source paper.

***

## Contact and Coordination

**Researcher:** Clemente J. Juan Oliver
**Institution:** TUM Chair of Spacecraft Systems
**Email:** clemente.juan@tum.de

**For AI Agents:**

- Await explicit instructions before implementing decision loops or representations
- Ask clarifying questions about operational scenarios
- Request specifications for hand-designed logic
- Follow scientific papers strictly when referenced
- Document all design decisions

***

**END OF FOUNDATION SPECIFICATION**

This document provides the architectural foundation. Specific implementations will follow step-by-step with researcher guidance and scientific paper references.


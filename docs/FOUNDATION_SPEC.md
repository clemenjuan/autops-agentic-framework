# Implementation Foundation for PhD Experimental Framework

**Project:** Custom Modular Architecture for Cognitive Satellite Constellation Autonomy Research
**Researcher:** Clemente J. Juan Oliver, TUM Chair of Spacecraft Systems
**Repository:** autops-demo
**Date:** February 12, 2026

***

## 1. Project Overview

### Objective

Build a modular experimental framework to systematically compare cognitive architectures for autonomous satellite constellation management. The framework must support testing combinations of:

- **Agent Organizations**: Centralized, Hierarchical, Distributed
- **Decision Loops**: SDA (Sense-Decide-Act), OODA, CoALA, and others
- **Representations**: Symbolic, Hybrid/Neuro-symbolic, Neural
- **Emergence Modes**: Hand-designed, Learned
- **Constellation Sizes**: 1, 5, 20-30, 100+ satellites


### Research Questions
The first iteration of research questions I came up with are:
- **RQ1a**: How do representation, emergence variations, decision-making loops and module design affect performance metrics?
- **RQ1b**: analysis Pareto Frontier of trade-offs
 - **RQ1c**: Which cognitive architectures suit which operational areas?
- **RQ2**: How do agent organizations impact performance and robustness?
- **RQ3**: How do architectures scale with constellation size (5-500 satellites)?


### Key Design Principles

1. **Orthogonality**: Each dimension (organization, loop, representation, emergence) is independent
2. **Modularity**: Components can be swapped without affecting others
3. **Reproducibility**: Configuration-driven experiments with seed control
4. **Fair Comparison**: Same environment and metrics for all variants
5. **Scientific Rigor**: Implementations follow established research papers

***

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Experiment Orchestration Layer                      │
│  - Configuration Management (YAML/JSON per experiment)          │
│  - Metrics Collection (utility, latency, robustness, op load)   │
│  - Reproducibility (seed control, logging, checkpointing)       │
│  - Statistical Analysis & Pareto Frontier Computation           │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌────────────────┐  ┌────────────────┐
│  Agent Org    │  │  Decision Loop │  │ Representation │
│  Controller   │  │  Engine        │  │ Module         │
│               │  │                │  │                │
│ - Centralized │  │ - SDA          │  │ - Symbolic     │
│ - Hierarchical│  │ - OODA         │  │ - Hybrid       │
│ - Distributed │  │ - CoALA        │  │ - Neural       │
└───────────────┘  └────────────────┘  └────────────────┘
                            │
                    ┌───────┴───────┐
                    ▼               ▼
            ┌──────────────┐ ┌──────────────┐
            │  Emergence   │ │   Memory     │
            │  Controller  │ │   System     │
            │              │ │   (Fixed)    │
            │ - Hand-coded │ │              │
            │ - Learned    │ │ All designs  │
            └──────────────┘ │ access same  │
                            │ information  │
                            └──────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
        ┌────────────────┐      ┌────────────────┐
        │   Satellite    │      │  Tool/Action   │
        │  Environment   │◄─────┤   Interface    │
        │                │      │                │
        │ - Orbital sim  │      │ - Operational  │
        │ - Task gen     │      │   scenario     │
        │ - Constraints  │      │   specific     │
        │ - Scalable     │      │   (TBD)        │
        └────────────────┘      └────────────────┘
```


***

## 3. Directory Structure

```
autops-demo/
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── satellite_env.py         # Core environment (abstract)
│   │   ├── orbital_mechanics.py     # Orekit integration
│   │   └── scenarios/               # Operational scenarios (TBD)
│   │       ├── __init__.py
│   │       └── README.md            # Scenario definitions go here
│   ├── agent_organization/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract AgentOrganization
│   │   ├── centralized.py
│   │   ├── hierarchical.py
│   │   └── distributed.py
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
│   │   ├── README.md                # Experiment design documentation
│   │   └── template.yaml            # Configuration template
│   └── scenarios/
│       └── README.md                # Scenario-specific configs
├── tests/
│   ├── test_environment.py
│   ├── test_agent_organization.py
│   ├── test_decision_loops.py
│   ├── test_representations.py
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

**Note:** Specific task types, rewards, and constraints depend on chosen operational scenario (space-based data centers, communications, or SSA). These will be defined in scenario-specific subclasses.

***

#### Agent Organization

**File:** `src/agent_organization/base.py`

**Purpose:** Abstract coordination patterns between agents. Controls how observations are distributed and actions are aggregated.

**Key Methods:**

- `distribute_observation(env_obs)`: Map environment observation to agent-specific observations
- `collect_actions(agent_actions)`: Aggregate agent actions for environment
- `get_agents()`: Return all agents in the organization

**Implementations:**

- `CentralizedOrganization`: Single agent controls entire constellation
- `HierarchicalOrganization`: Mission manager + local satellite agents
- `DistributedOrganization`: Peer-to-peer multi-agent with communication topology

***

#### Decision Loop Engine

**File:** `src/decision_loop/base.py`

**Purpose:** Abstract decision-making pattern defining temporal control flow. Each loop type follows specific research papers (e.g., CoALA paper for CoALA implementation).

**Key Methods:**

- `process(observation, memory)`: Main decision cycle, returns (action, updated_memory)
- `get_metrics()`: Return decision loop metrics

**⚠️ Critical:** Decision loop implementations must strictly follow scientific papers. Do not predefine specific steps—implementations will be created step-by-step following literature.

**Examples:**

- SDA: Linear reactive pattern
- OODA: Orient-heavy deliberation
- CoALA: Follow Sumers et al. (2023) "Cognitive Architectures for Language Agents"

***

#### Representation Module

**File:** `src/representation/base.py`

**Purpose:** How knowledge and decisions are represented. This is what fills the decision loop pattern.

**Key Methods:**

- `encode_observation(obs)`: Transform observation to internal representation
- `select_action(state, memory)`: Core decision-making logic
- Additional methods as required by specific decision loops

**Types:**

- **Symbolic**: Rules, planners, constraints (hand-designed logic)
- **Hybrid/Neuro-symbolic**: LLM reasoning + symbolic tools + MARL-networks
- **Neural**: Learned policies (RL-trained networks)

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
agent_organization: "centralized"  # centralized | hierarchical | distributed
decision_loop: "sda"               # sda | ooda | coala | [custom]
representation: "symbolic"          # symbolic | hybrid | neural
emergence_mode: "hand_designed"    # hand_designed | learned

# Configuration for each component
agent_organization_config:
  # Specific parameters for chosen organization

decision_loop_config:
  # Specific parameters for chosen decision loop

representation_config:
  # Specific parameters for chosen representation

emergence_config:
  # Parameters for loading/initializing representation

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

**Definition:** Total value achieved from completed tasks/objectives

**Rationale:** Primary performance metric—does the system accomplish its mission?

**Note:** Exact formula depends on operational scenario (reward function definition TBD)

***

#### 2. Latency

**Definition:** Decision-making computational time

**Rationale:** Real-time constraints in space operations—decisions must be timely

**Measurement:** Wall-clock time per decision cycle

***

#### 3. Robustness

**Definition:** Performance stability under perturbations and uncertainty

**Rationale:** Space environment is unpredictable—architectures must handle failures

**Note:** Specific robustness metrics require theoretical development (e.g., variance analysis, failure recovery metrics)

***

#### 4. Resource Efficiency

**Definition:** Achieved utility per unit resource consumed

**Rationale:** Satellites have limited power, data bandwidth, computation

**Note:** Resource model depends on operational scenario

***

#### 5. Operator Load

**Definition:** Required human intervention frequency

**Rationale:** Autonomy goal is reducing operator burden

**Note:** Operationalization requires defining what constitutes "intervention" (constraint violations, failed actions, manual overrides)

***

#### 6. Scalability

**Definition:** Performance degradation as constellation size increases

**Rationale:** Research question RQ3 directly addresses scaling

**Measurement:** Track all metrics as function of constellation_size

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
3. Implement CentralizedOrganization
4. Implement one decision loop (researcher chooses which)
5. Implement one representation (researcher chooses which)
6. End-to-end experiment execution

**Validation:** Complete experiment produces valid metrics for chosen configuration

***

### Phase 3: Morphological Expansion (Weeks 9-16)

**Goal:** Implement alternative configurations systematically

**Approach:**

- Add decision loops one at a time, following scientific papers
- Add representations following established methods
- Add agent organizations with validation
- Scale constellation size incrementally (1 → 5 → 20 → 100)

**Note:** Each new component requires theoretical justification and validation against baselines

***

### Phase 4: Learned Variants (Weeks 17-20)

**Goal:** Implement emergence mode "learned"

**Deliverables:**

1. RL training pipeline for neural representations
2. Learned policy integration
3. Comparison: hand-designed vs learned for same decision loop
4. Analysis of emergence vs non-emergence trade-offs

***

## 8. Development Standards

### Technology Stack

- **Python**: 3.11+
- **Dependency Management**: `uv` (existing in autops-demo)
- **Testing**: pytest
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style
- **Formatting**: black, isort, ruff
- **Configuration**: YAML via PyYAML or OmegaConf


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
    "torch",           # For neural representations (optional)
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

## 11. Operational Scenario Selection (TBD)

Before Phase 2, choose one of:

### Option 1: Space-Based Data Centers

**Tasks:** Computational job scheduling, thermal management, resource allocation
**Constraints:** Power, cooling, inter-satellite links
**Metrics:** Job completion rate, energy efficiency

### Option 2: Communications Constellation

**Tasks:** Ground contact scheduling, data routing, handoff coordination
**Constraints:** Bandwidth, visibility windows, latency requirements
**Metrics:** Data throughput, latency, coverage

### Option 3: Space Situational Awareness (SSA)

**Tasks:** Observation scheduling, sensor tasking, anomaly detection
**Constraints:** Sensor FOV, power budget, revisit requirements
**Metrics:** Target coverage, detection rate, revisit time

**Decision Point:** Researcher must select scenario based on:

- AUTOPS project relevance
- Data availability
- Complexity appropriate for PhD scope
- Vincenzo's research input

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

1. **Operational scenario selection** (space data centers | communications | SSA)
2. **First decision loop choice** (which to implement first?)
3. **First representation choice** (symbolic | hybrid | neural?)
4. **Hand-designed logic specifications** (rules, prompts, etc.)
5. **Scenario-specific constraints** (after scenario selected)

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


### Scientific Papers (examples for decision loops)

- CoALA: Sumers et al. (2023) "Cognitive Architectures for Language Agents", TMLR
- ReAct: Yao et al. (2023) "ReAct: Synergizing Reasoning and Acting in Language Models"
- Tree of Thoughts: Yao et al. (2023) "Tree of Thoughts: Deliberate Problem Solving with LLMs"

**Note:** Specific papers to follow will be provided per component during implementation.

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


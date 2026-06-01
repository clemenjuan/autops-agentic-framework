# Implementation Guide

Step-by-step guide for implementing new components in the experimental framework.

---

## General Principles

1. **Abstract Before Concrete** — Define interfaces first; implement after.
2. **Configuration Over Code** — Every experimental choice goes in YAML.
3. **Test-Driven Development** — Write tests alongside code.
4. **Scientific Rigor** — Follow published papers; document deviations.
5. **Incremental Complexity** — Start simple, add complexity systematically.

---

## Adding a New Decision Loop

*(The "Decision Loop" code module is the **Decision Procedure** axis; see [`FOUNDATION_SPEC.md` §3](FOUNDATION_SPEC.md#3-morphological-matrix-structure).)*

### Prerequisites
- Read and understand the source paper thoroughly.
- Identify the algorithm steps (reference specific sections/figures).

### Steps

1. **Create the module** at `src/decision_procedure/<loop_name>.py`.

2. **Cite the paper** in the module docstring:
   ```python
   """
   <Loop Name> Decision Loop.

   Implementation following:
       <Authors> (<Year>). "<Title>". <Venue>.
       Section X.Y, Algorithm Z.
   """
   ```

3. **Subclass `DecisionProcedure`** from `src/decision_procedure/base.py`.

4. **Implement `process()`**: Map the paper's algorithm steps into the
   `process(observation, memory) → (action, memory)` signature.

5. **Implement `get_metrics()`**: Return relevant metrics (latency, iterations, etc.).

6. **Add tests** in `tests/test_decision_loops.py`:
   - Test with a dummy representation.
   - Test that the algorithm steps execute in the correct order.
   - Test edge cases (empty observation, etc.).

7. **Register in configuration**: Ensure the experiment runner's factory
   can instantiate the new loop from YAML config.

8. **Document**: Update `src/decision_procedure/README.md`.

---

## Adding a New Representation

*Representation = **substrate** only (symbolic/subsymbolic/hybrid). Two things are **not** new
representations: (1) action-space richness — reactive vs agentic is a hybrid-only flavor; (2)
learned behaviour — `ppo`/`prompt_optimized`/`writable_coala` are the **Behaviour** overlay, wired
via `behaviour_config`, not separate classes. See [`FOUNDATION_SPEC.md` §3](FOUNDATION_SPEC.md#3-morphological-matrix-structure).*

### Steps

1. **Create the module** at `src/representation/<repr_name>.py`.

2. **Subclass `Representation`** from `src/representation/base.py`.

3. **Implement `encode_observation()`**: Transform raw observations into
   the representation's internal format.

4. **Implement `select_action()`**: Core decision logic.

5. **For learned variants**: Also implement `update()` for training.

6. **Register** with the `BehaviourController` using the `@register()` decorator:
   ```python
   from src.emergence.controller import register

   @register("my_symbolic_rules")
   class MySymbolicRepresentation(Representation):
       ...
   ```

7. **Add tests** in `tests/test_representations.py`.

8. **Document**: Update `src/representation/README.md`.

---

## Adding a New Agent Organization

### Steps

1. **Create the module** at `src/agent_organization/<org_name>.py`.

2. **Subclass `AgentOrganization`** from `src/agent_organization/base.py`.

3. **Implement**:
   - `initialize()`: Set up agents for given constellation size.
   - `distribute_observation()`: Map environment obs → per-agent obs.
   - `collect_actions()`: Aggregate agent actions → environment actions.
   - `get_agents()`: List all agent IDs.

4. **Add tests** in `tests/test_agent_organization.py`.

5. **Register** in the experiment runner's factory.

---

## Adding a New Operations Paradigm

### Steps

1. **Create the module** at `src/operations/<paradigm_name>.py`.

2. **Subclass `OperationsParadigm`** from `src/operations/base.py`.

3. **Implement**:
   - `filter_observation(full_observation, step)`: What the agent sees (full state, stale data, partial view, etc.).
   - `can_act(step, ground_pass_active)`: Whether the agent can issue commands at this step.
   - `process_action(action, step, ground_pass_active)`: Buffer, delay, transform, or pass through actions.
   - `get_name()`: Return a unique string identifier.

4. **Register** in `src/orchestration/config_loader.py`:
   - Add the paradigm name to `VALID_OPERATIONS_PARADIGMS`.

5. **Register** in `src/orchestration/experiment_runner.py`:
   - Add an import and case in `_create_operations_paradigm()`.

6. **Add tests** in `tests/test_operations_paradigm.py`.

### Reference implementations

- `AutonomousHybrid` (`src/operations/autonomous_hybrid.py`): Pass-through paradigm. Full real-time state, immediate actions every step.
- `ConventionalGround` (`src/operations/conventional_ground.py`): Stale telemetry, uplink-gated actions during ground passes only.

---

## Adding a New Operational Scenario

### Steps

1. **Create the scenario** at `src/environment/scenarios/<scenario_name>.py`.

2. **Subclass `SatelliteEnvironment`** from `src/environment/satellite_env.py`.

3. **Implement**:
   - `reset()`: Initialise the constellation for this scenario.
   - `step()`: Execute one timestep with scenario-specific physics/tasks.
   - `get_observation()`: Return scenario-specific observation.
   - `get_metrics()`: Return scenario-specific performance metrics.

4. **Create scenario config** in `configs/scenarios/<scenario_name>.yaml`.

5. **Implement scenario-specific `MetricsCollector`** subclass.

6. **Add tests** in a new test file (e.g., `tests/test_scenario_<name>.py`).

7. **Document** in `docs/scenarios.md`.

---

## Running Experiments

### Single Experiment
```bash
uv run autops run configs/experiments/my_experiment.yaml --episodes 1 --steps 100
```

### Training Learned-Emergence Variants
```bash
uv run autops train configs/experiments/eventsat_sas_sda_subm_le_ah.yaml    # PPO
uv run autops train configs/experiments/eventsat_sas_sda_hyre_lep_ah.yaml   # prompt-opt
uv run autops train configs/experiments/eventsat_sas_sda_hyag_lec_ah.yaml   # writable CoALA
```

### Batch Experiments
```bash
uv run autops generate --template configs/experiments/template.yaml
uv run autops batch configs/experiments/generated/
```

### Analysing Results
```bash
uv run autops analyze data/results/my_experiment/
jupyter notebook notebooks/analysis.ipynb
```

---

## Validation Checklist

Before committing a new component:

- [ ] All tests pass (`pytest tests/`)
- [ ] Type hints on all public APIs
- [ ] Google-style docstrings
- [ ] Configuration-driven (no hardcoded choices)
- [ ] Documented in relevant README or docs/
- [ ] Scientific paper cited (for decision loops, representations)
- [ ] Reproducible (seed-controlled)

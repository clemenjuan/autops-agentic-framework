# Implementation Guide

Step-by-step guide for implementing new components in the experimental framework.

---

## General Principles

1. **No Trash Files** — Add only necessary source, configs, tests, and canonical docs. Prefer one existing file over several new ones; keep generated artifacts ignored and cleaned up.
2. **Abstract Before Concrete** — Define interfaces first; implement after.
3. **Configuration Over Code** — Every experimental choice goes in YAML.
4. **Test-Driven Development** — Write tests alongside code.
5. **Scientific Rigor** — Follow published papers; document deviations.
6. **Incremental Complexity** — Start simple, add complexity systematically.

---

## Decision Procedure Policy

The current EventSat and Flamingo benchmarks hold the decision driver fixed to SDA. Do not add new files under `src/core/decision_procedure/` unless the benchmark design itself changes. Agentic behaviour belongs in the representation layer (`agentic_eventsat` / scheduler variants), not in a separate ReAct-style loop.

## Adding a New Representation

*Representation = **substrate** only (symbolic/subsymbolic/hybrid). Two things are **not** new
representations: (1) action-space richness — single-shot vs agentic exists only on LLM-bearing cells; (2)
learned/online behaviour — `ppo` (RL training) and `writable_coala` (agentic online learning) are
wired via `behaviour_config`, not separate classes. See [`morphological_matrix.md`](morphological_matrix.md).*

### Steps

1. **Create the module** in the owning scenario package, e.g. `src/eventsat/<repr_name>.py` or `src/flamingo/<repr_name>.py`.

2. **Subclass `Representation`** from `src/core/representation.py`.

3. **Implement `encode_observation()`**: Transform raw observations into
   the representation's internal format.

4. **Implement `select_action()`**: Core decision logic.

5. **For learned variants**: Also implement `update()` for training.

6. **Register** with the `BehaviourController` using the `@register()` decorator:
   ```python
   from src.core.behaviour.controller import register

   @register("my_symbolic_rules")
   class MySymbolicRepresentation(Representation):
       ...
   ```

7. **Add tests** in `tests/test_representations.py`.

8. **Document**: Update `docs/implementations.md` when the representation changes.

---

## Adding a New Agent Organization

### Steps

1. **Create the module** at `src/core/organization/<org_name>.py`.

2. **Subclass `AgentOrganization`** from `src/core/organization/base.py`.

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

1. **Create the module** at `src/core/operations/<paradigm_name>.py`.

2. **Subclass `OperationsParadigm`** from `src/core/operations/base.py`.

3. **Implement**:
   - `filter_observation(full_observation, step)`: What the agent sees (full state, stale data, partial view, etc.).
   - `can_act(step, ground_pass_active)`: Whether the agent can issue commands at this step.
   - `process_action(action, step, ground_pass_active)`: Buffer, delay, transform, or pass through actions.
   - `get_name()`: Return a unique string identifier.

4. **Register** in `src/core/config_loader.py`:
   - Add the paradigm name to `VALID_OPERATIONS_PARADIGMS`.

5. **Register** in `src/core/experiment_runner.py`:
   - Add an import and case in `_create_operations_paradigm()`.

6. **Add tests** in `tests/test_operations_paradigm.py`.

### Reference implementations

- `AutonomousHybrid` (`src/core/operations/autonomous_hybrid.py`): Pass-through paradigm. Full real-time state, immediate actions every step.
- `ConventionalGround` (`src/core/operations/conventional_ground.py`): Stale telemetry, uplink-gated actions during ground passes only.

---

## Adding a New Operational Scenario

### Steps

1. **Create the scenario package** at `src/<scenario_name>/` with an `env.py`.

2. **Subclass `SatelliteEnvironment`** from `src/core/satellite_env.py`.

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
uv run autops train configs/experiments/eventsat_sas_ao_rl.yaml      # PPO (RL onboard)
uv run autops train configs/experiments/eventsat_sas_ag_llm-a.yaml   # writable CoALA (agentic online learning)
```

### Batch Experiments
```bash
uv run autops generate --template configs/experiments/template.yaml
uv run autops batch configs/experiments/generated/
```

### Analysing Results
```bash
uv run python scripts/refresh_board.py
```

Open `data/figures/index.html` for the local board launcher.

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

# Operational Scenarios

Scenario-specific environment implementations go here.

Each scenario subclasses `SatelliteEnvironment` and defines:

- **Mission objectives**: What the constellation is trying to achieve
- **Task types**: Specific tasks generated for agents
- **Constraints**: Physical and operational limitations
- **Reward structure**: How performance is quantified

## Candidate Scenarios (select one for Phase 2)

### Option 1: Large-Scale Constellation
- Tasks: Resource scheduling, coordination, coverage optimisation
- Constraints: Power, ISL bandwidth, orbital geometry

### Option 2: Communications Constellation
- Tasks: Ground contact scheduling, data routing, handoff coordination
- Constraints: Bandwidth, visibility windows, latency requirements

### Option 3: Space Situational Awareness (SSA)
- Tasks: Observation scheduling, sensor tasking, anomaly detection
- Constraints: Sensor FOV, power budget, revisit requirements

## Implementation Guidelines

1. Subclass `SatelliteEnvironment` from `src/environment/satellite_env.py`
2. Implement all abstract methods (`reset`, `step`, `get_observation`, `get_metrics`)
3. Define scenario-specific configuration in YAML under `configs/scenarios/`
4. Write comprehensive tests in `tests/`

# ADCS Simulation

A modular and configurable Attitude Determination and Control System (ADCS) simulation for the **EventSat 6U CubeSat**, designed as an RL training/evaluation environment and reconfigurable for other CubeSat missions.

**Status Update 03.06.2026:** Skeleton is complete. All of the modules are created, where each of them is filled with dummy functions. The idea is to lock all the interfaces between modules from the start. Functions currenty return zeros/identity. The end-to-end loop is running, verified by the 'adcs_test'.

## Project Goals

1. Provide a complete ADCS simulation environment for EventSat, integrating with [OreKit](https://www.orekit.org/) for orbit propagation and environmental data.
2. Serve as the training environment for a reinforcement learning agent that autonomously controls satellite attitude across multiple operational modes.
3. Provide modular design, such that the same simulation can be reconfigured for other CubeSat missions by changing the mission parameters.

## File structure

```
src/environment/orbital/
├── propagator.py            # OreKit wrapper; EnvironmentData + get_environment()
└── adcs/
    ├── __init__.py
    ├── state.py             # SatState — true state vector
    ├── configs.py           # configuration: *Config + SensorSuite/ActuatorSuite
    ├── eventsat.py          # EventSat instances (sensors, actuators)
    ├── sensors.py           # read_*() + SensorMeasurements
    ├── actuators.py         # apply_*() + ControlCommand
    ├── estimator.py         # EstimatorState + update_estimator() (MEKF)
    ├── control.py           # Setpoint + compute_control()
    ├── dynamics.py          # integrate() + disturbance_torque()
    └── simulation.py        # step() + run()
tests/
└── test_adcs.py             # end-to-end simulation test
```
### Data flow

At each timestep, `simulation.step()` performs the following sequence:

1. **Environment:** query OreKit at `t + dt` for orbit state (`r_eci`, `v_eci`) and environmental quantities (B-field, sun vector, eclipse flag, atmospheric density), bundled into an `EnvironmentData` object.
2. **Sensors:** call each sensor's `read_*` function with the true state and its config. Outputs bundled into a `SensorMeasurements` object.
3. **Estimator:** `update_estimator(estimator, measurements, dt)` advances the MEKF and returns a new `EstimatorState`.
4. **Controller:** `compute_control(estimator, setpoint, actuators, dt)` produces a `ControlCommand` (one torque command per wheel, one dipole per magnetorquer).
5. **Actuators:** each command goes through its corresponding `apply_*` function, returning the body-frame torque it produces. The contributions are summed.
6. **Disturbance:** `disturbance_torque(state, env)` returns the body-frame disturbance torque from gravity gradient, drag, SRP, and residual dipole.
7. **Integration:** `integrate(state, total_torque, dt)` advances attitude / angular velocity / wheel speeds. The orbit fields are then overwritten from `env` to stay consistent with OreKit's propagation.

The data flow is closed-loop: the controller acts on the estimator's view, never the truth; the integrator updates the truth; the new truth is what the sensors measure on the next step.

- **Truth vs. estimate are separate:** `SatState` (true) and `EstimatorState`
  (the filter's view) are distinct; the controller and the RL agent only see the estimate.
- **Configurability:** The simulation physics is written once and can work with different mission and config file.
- **OreKit is isolated:** All of the information from OreKit goes through `propagator.py`, thus nothing in `adcs/` imports directly from OreKit.

## Conventions

### Frames

- Attitude is the ECI→body rotation, stored as a scalar-first quaternion `[w, x, y, z]`.
- Angular velocity is expressed in the body frame.
- Position and velocity are expressed in ECI.

### Time

- The loop runs in integer step indices: `start_step` (inclusive) to `end_step` (exclusive), with a fixed step length `step_s` [s].
- Continuous time in seconds is carried in `SatState.t` (computed as `step_index * step_s` where the physics needs it).

### Imports

- Absolute, rooted at `src` — e.g. `from src.environment.orbital.adcs.state import SatState`.

### Typing

- Use `Dict` / `List` / `Optional` / `Tuple` from `typing`, not the lowercase builtin generics.

### Logging

- One logger per module: `logger = logging.getLogger(__name__)`.

## Running

The OreKit zipped file: orekit-data.zip, needs to be added to root and then:


```bash
uv sync --extra dev --extra orbital
uv run python -c "from src.environment.orbital.adcs.eventsat import sensors, actuators; from src.environment.orbital.adcs.simulation import run; h = run(sensors, actuators, step_s=1.0, start_step=0, end_step=10); print(len(h), h[0].t, h[-1].t)"
uv run pytest tests/test_adcs.py -v
```

## EventSat configuration

**Sensors:**
- **3 magnetometer readings:** deployable A, deployable B, compact
- **2 fine sun sensors:** fss_a, fss_b
- **1 coarse sun sensor array:** with 10 photodiodes
- **1 earth horizon sensor**

**Actuators:**
- **4 reaction wheels:** in pyramid configuration
- **3 magnetorquers:** one per body axis

Satellite physical parameters (mass, inertia tensor, COM offset), orbit configuration, and full mission setup will be added later.

## Another mission

Add `othersat.py` building `sensors`/`actuators` from the same `configs.py` file with that satellite's instances, and pass them to `run()`. Configuration and physics stay
untouched. (A future YAML loader will replace the hand-written instances.)

## Roadmap

- [x] ADCS simulation full skeleton
- [x] Simulation skeleton test
- [ ] Satellite physical parameters (mass, inertia tensor, ...)
- [ ] Full mission config (orbit, simulation parameters, ...)
- [ ] Real OreKit integration
- [ ] Replace dynamics dummy with real physics
- [ ] Replace sensor dummy with real measurement models
- [ ] Replace actuator dummy with real actuator dynamics
- [ ] Replace MEKF dummy with real Kalman filter
- [ ] Replace control sdummy with real control laws
- [ ] Gymnasium wrapper for RL training

## References

# ADCS Simulation

A modular and configurable Attitude Determination and Control System (ADCS) simulation for the **EventSat 6U CubeSat**, designed as an RL training/evaluation environment and reconfigurable for other CubeSat missions.

**Status Update 04.09.2026:** Earth horizon sensor implemented in `sensors.py`.
The boresight is mounted ~69 deg off nadir so it lands on the limb, modeled after the 
information from the ADCS PD and EHS PD, but not confirmed for eventsat. Field of view 
was chosen to be 90x72 and per ADCS PD and not EHS PD, which stated 90x80, since both documents
state the the half verical fov is 36 deg. FoV roll is also implemented.

**Status Update 02.09.2026:** Coarse sun sensor implemented in `sensors.py`.
Each cell returns a current. `noise_std` is a placeholder, as well as the surface normals.

**Status Update 02.09.2026:** Physical constants collected into
`adcs/constants.py`: Earth's radius, gravitational parameter and rotation rate,
solar radiation pressure, and Earth's Bond albedo. The rule is that physical
constants live here.

**Status Update 31.08.2026:** Fine sun sensor implemented in `sensors.py`. 
The noise and bias are applied to `(α, β)` rather than to the output vector. 
`read_fine_sun_sensor` returns `Optional[np.ndarray]`: `None` when there is no 
measurement (eclipse, out offield of view, above the 70 °/s slew cutoff, or 
when noise pushes a reported angle outside the instrument's physical range). 
`fov_half_angle` is now validated to lie in (0, π/2], since
the pinhole parameterisation has no rear hemisphere. 

**Status Update 29.08.2026:** Rate gyro measurement model implemented in
`sensors.py`: `ω_meas = R_S·ω_body + b + n_v`, Farrenkopf (1978) two-term
model with the per-sample noise scaled as `arw/√dt` (Woodman 2007 §3.2.2
Eq. 5). EventSat models GYR0 only; GYR1 stays defined but out of the suite.
All gyro parameters remain placeholders. `RateGyroConfig` gains
`update_rate_hz`; `MagnetometerConfig.measurement_range` renamed to
`max_field`, following the `max_<quantity>` convention.

**Status Update 28.08.2026:** Magnetometer measurement model implemented in
`sensors.py`: `B_meas = R_S·C(q)·B_eci + b₀ + n`, additive structure after
Alonso & Shuster (2002), with per-axis Gaussian noise and saturation clipping.
Models the *calibrated* sensor output, so scale-factor and non-orthogonality
errors are excluded as removed on board. `MagnetometerConfig` gains
`update_rate_hz` and `measurement_range`. `eventsat.py` now carries three
magnetometers (deployable primary, deployable secondary, compact) — the
CubeMag Deployable reports two live chips, not one. Noise corrected: the
CubeMag PD quotes 50 nT at 3σ and `noise_std` is 1σ. Nine magnetometer tests
added.

**Status Update 26.08.2026:** Worked on `sensors.py`. Rate gyros are  to the sensors, placeholder values are filled in `eventsat.py`. New data class for gyro bias: `SensorState`, it will hold error states, that carry memory across steps. Seeding is added. Functions for initializing and propagating the sensor state were added: `initial_sensor_state` and `propagate_sensor_state`

**Status Update 25.08.2026:** `actuators.py` is fully implemented, which includes the reaction wheel and the magnetorquers models.

`apply_reaction_wheel()` returns the achieved scalar motor torque. Commands are clamped to `max_torque`. Coulomb and viscous friction (Paluszek Eq. 10.14) are implemented. Momentum is bounded to `±max_momentum` at the end of the step. 

`apply_magnetorquer()` returns τ = m × B with the field rotated into the body frame. 

**Status Update 14.07.2026:** `eventsat.py` is updated with the correct values for the physical parameters of the satellite, reaction wheels, magnetorquers and Orbit. Some values that are still uncertain, are marked as such.

**Status Update 28.06.2026:** `dynamics.py` is fully implemented. `integrate()` is a full reaction-wheel gyrostat. `disturbance_torque()` sums gravity gradient, residual magnetic dipole, aerodynamic drag, and solar radiation pressure.

**Status Update 23.06.2026:** `propagator.py` is fully implemented - real OreKit integration replacing the zero stub. `get_environment()` now returns a fully populated `EnvironmentData` (orbit state, geomagnetic field, Sun vector, eclipse flag, atmospheric density), all in ECI frame (GCRF) and SI units.

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
    ├── configs.py           # configuration: *Config + SensorSuite/ActuatorSuite + SatelliteConfig + OrbitConfig
    ├── eventsat.py          # EventSat instances (sensors, actuators, satellite, orbit)
    ├── sensors.py           # read_*() + SensorMeasurements
    ├── actuators.py         # apply_*() + ControlCommand
    ├── estimator.py         # EstimatorState + update_estimator() (MEKF)
    ├── control.py           # Setpoint + compute_control()
    ├── dynamics.py          # integrate() + disturbance_torque()
    └── simulation.py        # step() + run()
tests/
└── test_adcs.py             # end-to-end simulation test + disturbance & propagator physics tests

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

The OreKit zipped file: orekit-data.zip, needs to be added to root. The standard orekit-data archive does not include the geomagnetic model files. For a non-zero B-field, IGRF.COF must be added manually from the the NOAA Geomag 7.0 package.

Then:

```bash
uv sync --extra dev --extra orbital
uv run python -c "
import numpy as np
from src.environment.orbital.adcs.eventsat import sensors, orbit
from src.environment.orbital.adcs.simulation import initial_state
from src.environment.orbital.adcs.sensors import read_magnetometer
from src.environment.orbital import propagator as P
P.configure(orbit)
env = P.get_environment(0.0)
state = initial_state(0.0, 4)
rng = np.random.default_rng(0)
for m in sensors.magnetometers:
    b = read_magnetometer(state, env, m, rng)
    print(m.name, np.linalg.norm(b))
"
uv run pytest tests/test_adcs.py -v
```

## EventSat configuration

**Sensors:**
- **3 magnetometer readings:** deployable primary, deployable secondary, compact
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
- [x] Satellite physical parameters (mass, inertia tensor, ...)
- [x] Real OreKit integration
- [x] Replace dynamics dummy with real physics
- [x] Replace actuator dummy with real actuator dynamics
- [ ] Replace sensor dummy with real measurement models
- [ ] Full mission config (orbit, simulation parameters, ...)
- [ ] Replace MEKF dummy with real Kalman filter
- [ ] Replace control sdummy with real control laws
- [ ] Gymnasium wrapper for RL training

## References
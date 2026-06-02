"""Satellite true state.

Defines SatState, the true state vector.
- The dynamics integrator changes it
- The sensors measure it
- Then the estimator approximates it based on measurement
- The controller never sees SatState


Frame conventions: 
Each quantity is stored in the frame where its governing equation is simplest 
    * Attitude is the rotation from the ECI (inertial) frame to the body
      frame, stored as a scalar-first quaternion [w, x, y, z].
    * Angular velocity is expressed in the body frame.
    * Position and velocity are expressed in the ECI frame.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SatState:
    """True satellite state at one instant.

    Attributes:
        t: Time since the simulation epoch [s].
        q_eci_body: Attitude quaternion, ECI to body, scalar-first
            [w, x, y, z], shape (4,).
        omega_body: Angular velocity in the body frame [rad/s], shape (3,).
        wheel_speeds: Reaction wheel speeds [rad/s], one per wheel,
            shape (n_wheels,). EventSat has four.
        r_eci: Position in the ECI frame [m], shape (3,).
        v_eci: Velocity in the ECI frame [m/s], shape (3,).
    """

    t: float
    q_eci_body: np.ndarray
    omega_body: np.ndarray
    wheel_speeds: np.ndarray
    r_eci: np.ndarray
    v_eci: np.ndarray
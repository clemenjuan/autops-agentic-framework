"""Rigid-body attitude dynamics and environmental disturbances.

integrate is the only writer of the true rotational state: given the net torque
on the body, it advances attitude, angular velocity, and wheel speeds over one
timestep.

disturbance_torque returns the net environmental torque that feeds into it.

All functions are currently place-holders, so the loop runs.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.propagator import EnvironmentData
from src.environment.orbital.adcs.configs import SatelliteConfig
from src.environment.orbital.adcs.constants import MU_EARTH, OMEGA_EARTH, SOLAR_PRESSURE

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product ``q1 ⊗ q2`` (scalar-first ``[w, x, y, z]``).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_kinematics(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion time-derivative ``q̇`` for a body-frame angular velocity.

    Locked convention (Solà, arXiv:1711.02508): ``q`` is the scalar-first
    ECI->body attitude quaternion, ``omega`` is the body-frame angular
    velocity [rad/s], and

        q̇ = ½ q ⊗ [0, ω].

    ``omega`` is promoted to the pure quaternion ``[0, ωx, ωy, ωz]`` before the
    product. This is the kinematics half of the convention; the matching
    ECI->body rotation matrix (built from the conjugate of ``q``) arrives with
    the gravity-gradient disturbance, and a single-axis spin test will lock the
    two halves together.
    """
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    return 0.5 * quat_multiply(q, omega_quat)

def dcm_eci_to_body(q: np.ndarray) -> np.ndarray:
    """Direction-cosine matrix C with ``v_body = C @ v_eci``.

    Convention partner of ``quat_kinematics``. With the scalar-first ECI->body
    quaternion and kinematics q̇ = ½ q⊗[0,ω], the standard Hamilton rotation
    matrix R(q) is the body->ECI rotation, so the ECI->body transform is its
    transpose, C = R(q)ᵀ (equivalently a rotation by the conjugate of q). A
    single-axis spin test confirms this agrees with the kinematics.
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z),       2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),   1 - 2*(x*x + y*y)],
    ])
    return R.T

def _omega_dot(
    omega: np.ndarray,
    wheel_speeds: np.ndarray,
    body_torque: np.ndarray,
    wheel_torque: np.ndarray,
    params: SatelliteConfig,
) -> np.ndarray:
    """Body angular acceleration ω̇ for the full reaction-wheel gyrostat.

    Spec body equation, reduced-inertia form:

        h  = I·ω + W·J_w·(Ω + Wᵀ·ω)      # total system momentum (body frame)
        ω̇ = I⁻¹·[ τ_ext − ω×h − W·u_w ]

    with I the reduced inertia (cached on ``params``), τ_ext = ``body_torque``
    (magnetorquer + disturbances), and u_w = ``wheel_torque`` (per-wheel motor
    torque). The ``Wᵀ·ω`` inside h is each wheel's *absolute* spin — its speed
    relative to the body plus the body's own rotation along that wheel's axis —
    and ``W·u_w`` is the motor reaction on the body.
    """
    W = params.wheel_axes            # (3,4)
    Jw = params.wheel_inertia_mat     # (4,4)
    h = params.inertia @ omega + W @ Jw @ (wheel_speeds + W.T @ omega)
    return params.inertia_inv @ (body_torque - np.cross(omega, h) - W @ wheel_torque)

def _wheel_dot(
    omega_dot: np.ndarray,
    wheel_torque: np.ndarray,
    params: SatelliteConfig,
) -> np.ndarray:
    """Wheel angular accelerations Ω̇ (per wheel, relative to the body).

    Spec wheel equation:

        Ω̇ = J_w⁻¹·u_w − Wᵀ·ω̇

    The ``−Wᵀ·ω̇`` term is the body's angular acceleration felt along each
    wheel's spin axis: wheel speed is defined *relative to the body*, so when the
    body accelerates the relative speed changes even at constant motor torque.
    Must be called after ``_omega_dot`` — it consumes ω̇.
    """
    return params.wheel_inertia_inv @ wheel_torque - params.wheel_axes.T @ omega_dot

def _state_derivative(
    x: np.ndarray,
    body_torque: np.ndarray,
    wheel_torque: np.ndarray,
    params: SatelliteConfig,
) -> np.ndarray:
    """Derivative ẋ of the stacked attitude state x = [q(4), ω(3), Ω(4)].

    Assembles the full gyrostat derivative in the spec's required order —
    kinematics, then ω̇, then Ω̇ (which depends on ω̇) — and returns ẋ in the
    same 11-vector layout. Inputs are held constant across an RK4 step
    (zero-order hold).
    """
    q = x[0:4]
    omega = x[4:7]
    wheel_speeds = x[7:11]
    q_dot = quat_kinematics(q, omega)
    omega_dot = _omega_dot(omega, wheel_speeds, body_torque, wheel_torque, params)
    wheel_dot = _wheel_dot(omega_dot, wheel_torque, params)
    return np.concatenate([q_dot, omega_dot, wheel_dot])


def integrate(
    state: SatState,
    body_torque: np.ndarray,
    wheel_torque: np.ndarray,
    params: SatelliteConfig,
    dt: float,
) -> SatState:
    """Advance the true rotational state one step dt with RK4 over the gyrostat.

    Args:
        state: current true ``SatState``.
        body_torque: (3,) external body torque tau_ext = magnetorquer + disturbances.
        wheel_torque: (4,) per-wheel motor torque u_w.
        params: ``SatelliteConfig`` with cached inertia and wheel geometry.
        dt: step length [s].

    Both torques are held constant across the step (zero-order hold).
    """
    # pack the rotational state into the stacked 11-vector [q(4), w(3), omega(4)]
    x = np.concatenate([state.q_eci_body, state.omega_body, state.wheel_speeds])

    # RK4 — torques held constant across all four stages
    k1 = _state_derivative(x, body_torque, wheel_torque, params)
    k2 = _state_derivative(x + 0.5 * dt * k1, body_torque, wheel_torque, params)
    k3 = _state_derivative(x + 0.5 * dt * k2, body_torque, wheel_torque, params)
    k4 = _state_derivative(x + dt * k3, body_torque, wheel_torque, params)
    x_new = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # single end-of-step quaternion renormalization
    q_new = x_new[0:4] / np.linalg.norm(x_new[0:4])

    return replace(
        state,
        t=state.t + dt,
        q_eci_body=q_new,
        omega_body=x_new[4:7].copy(),
        wheel_speeds=x_new[7:11].copy(),
    )

def _residual_dipole_torque(
    state: SatState, env: EnvironmentData, params: SatelliteConfig
) -> np.ndarray:
    """Residual magnetic dipole torque in the body frame [N·m], shape (3,).

        tau = m_res × B_body (Paluszek Eq. 8.31) 

        ``m_res`` is the spacecraft's leftover magnetic moment 
        ``params.residual_dipole`` [A·m²], fixed in the body frame
        B_body is ``env.b_field_eci`` [T] rotated into the body frame.
    """
    b_body = dcm_eci_to_body(state.q_eci_body) @ env.b_field_eci
    return np.cross(params.residual_dipole, b_body)

def _projected_area(direction_body: np.ndarray, dimensions: np.ndarray) -> float:
    """Projected area [m²] of the satellite to a flux arriving along
    ``direction_body`` - a unit vector in the body frame pointing from the
    spacecraft toward the source (velocity direction for drag, Sun direction for
    SRP).

    Sums ``A_i · max(0, n̂_i · d̂)`` over the six box faces: Paluszek Eq. 8.2 per
    face, with 8.3.1's rule that only faces turned toward the flux contribute.
    Deployables are not modeled.
    """
    Lx, Ly, Lz = dimensions
    normals = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    areas = np.array([Ly * Lz, Ly * Lz, Lx * Lz, Lx * Lz, Lx * Ly, Lx * Ly])
    return float(np.sum(areas * np.maximum(0.0, normals @ direction_body)))


def _aerodynamic_torque(
    state: SatState, env: EnvironmentData, params: SatelliteConfig
) -> np.ndarray:
    """Aerodynamic drag torque in the body frame [N·m], shape (3,).

        v_rel = v_eci − omega × r_eci          # relative to the co-rotating atmosphere
        F     = −1/2 ro C_D A_p |v_rel| v_rel   # drag opposing motion (Paluszek Eq. 8.1)
        tau     = (cop − com) × F

    A_p is the attitude-dependent projected area (six-face box model). The force
    acts at the center of pressure COP; the torque about the COM uses the lever arm
    ``cop_offset − com_offset``. Returns zero if the relative speed is zero.
    """
    omega_earth = np.array([0.0, 0.0, OMEGA_EARTH])
    v_rel_eci = env.v_eci - np.cross(omega_earth, env.r_eci)

    v_rel_body = dcm_eci_to_body(state.q_eci_body) @ v_rel_eci
    speed = np.linalg.norm(v_rel_body)
    if speed == 0.0:
        return np.zeros(3)

    area = _projected_area(v_rel_body / speed, params.dimensions)
    force = -0.5 * env.atmospheric_density * params.drag_coeff * area * speed * v_rel_body

    lever = params.cop_offset - params.com_offset
    return np.cross(lever, force)

def _srp_torque(
    state: SatState, env: EnvironmentData, params: SatelliteConfig
) -> np.ndarray:
    """Solar radiation pressure torque in the body frame [N·m], shape (3,).

        F = −P · C_r · A_p · s_hat        # photon pressure, pushing away from the Sun
        tau = (cop − com) × F

    Reduction of Paluszek Eq. 8.35 with a single reflectivity
    C_r = ``params.reflectivity``; P is the solar pressure at 1 AU
    (``SOLAR_PRESSURE``). s_hat is ``env.sun_vector_eci`` (unit, spacecraft→Sun)
    rotated into the body frame; A_p is the sunlit projected area (six-face box
    model). Gated to zero in eclipse (umbra). Eq. 8.35's per-face optical model
    is the upgrade path.
    """
    if env.eclipse:
        return np.zeros(3)
    sun_body = dcm_eci_to_body(state.q_eci_body) @ env.sun_vector_eci
    area = _projected_area(sun_body, params.dimensions)
    force = -SOLAR_PRESSURE * params.reflectivity * area * sun_body
    lever = params.cop_offset - params.com_offset
    return np.cross(lever, force)

def _gravity_gradient_torque(
    state: SatState, env: EnvironmentData, params: SatelliteConfig
) -> np.ndarray:
    """Gravity-gradient torque in the body frame [N·m], shape (3,).

        tau = (3*mu / r^3) · nn_hat × (I @ n_hat)

    where r = |r_eci|, n_hat is the unit nadir direction in the body frame, and I is
    the spacecraft inertia tensor. Uses ``inertia_full`` (the physical mass
    distribution), not the dynamics-reduced ``inertia``. The torque vanishes when
    a principal axis of I points along nadir; the sign of n̂ is irrelevant.
    """
    r_eci = env.r_eci
    r_norm = np.linalg.norm(r_eci)
    if r_norm == 0.0:
        return np.zeros(3)
    nadir_body = dcm_eci_to_body(state.q_eci_body) @ (r_eci / r_norm)
    inertia = params.inertia_full
    return (3.0 * MU_EARTH / r_norm**3) * np.cross(nadir_body, inertia @ nadir_body)

def disturbance_torque(
    state: SatState, env: EnvironmentData, params: SatelliteConfig
) -> np.ndarray:
    """Net environmental disturbance torque in the body frame [N·m], shape (3,).
    """
    return (
        _gravity_gradient_torque(state, env, params)
        + _residual_dipole_torque(state, env, params)
        + _aerodynamic_torque(state, env, params)
        + _srp_torque(state, env, params)
    )
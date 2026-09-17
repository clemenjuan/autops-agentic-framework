from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# Canonical order of the dense, per-step reward components. Used both to build
# the reward vector and to look up matching weights from the `weights` dict, so
# the two stay in sync.
REWARD_COMPONENTS = (
    "pointing_error_penalty",
    "boundary_layer_reward",
    "rw_saturation_reward",
    "slew_rate_boundary_reward",
)

# Sparse components, paid only on the step where the corresponding event fires.
# Unlike REWARD_COMPONENTS these are absolute reward amounts rather than weights
# multiplying a computed quantity, so they are summed in directly.
EVENT_COMPONENTS = (
    "target_cleared_bonus",
    "mission_done_bonus",
)


def get_total_reward(
    err_quat: np.ndarray,
    omega_wheels: np.ndarray,
    omega_wheels_max: float,
    omega_body: np.ndarray,
    omega_body_thresh: float,
    sigma: float,
    weights: Dict[str, float],
    target_cleared: bool = False,
    mission_done: bool = False,
    m_dipole: Optional[np.ndarray] = None,
    m_dipole_max: Optional[float] = None,
) -> Dict[str, float]:
    """
    Computes the total scalar reward -- dense shaping terms plus the sparse
    event bonuses -- returning both the total and the individual parts for
    analysis. This is the whole reward the environment hands to the agent; the
    env only decides *whether* the events fired, not what they are worth.

    weights: every reward magnitude, keyed by REWARD_COMPONENTS and
    EVENT_COMPONENTS. Required, with no default: the weights actually
    trained with live in the scenario config (eventsat._reward_weights),
    and a fallback here would shadow them silently whenever a caller
    forgot to pass them.

    target_cleared / mission_done: whether those events fired on this step.
    

    m_dipole / m_dipole_max are reserved for a future magnetorquer dipole
    saturation term and are currently unused.
    """
    e_current = float(np.linalg.norm(err_quat[1:4]))

    reward_vector = np.array(
        [
            pointing_error_penalty(e_current),
            boundary_layer_reward(e_current, sigma),
            rw_saturation_penalty(omega_wheels, omega_wheels_max),
            slew_rate_boundary_penalty(omega_body, omega_body_thresh),
        ],
        dtype=float,
    )
    weight_vector = np.array([weights.get(name, 0.0) for name in REWARD_COMPONENTS], dtype=float)

    reward = float(np.dot(weight_vector, reward_vector))

    reward_vector = np.multiply(weight_vector, reward_vector)

    events = event_rewards(
        target_cleared=target_cleared,
        mission_done=mission_done,
        weights=weights,
    )

    return {
        "reward": reward + sum(events.values()),
        **dict(zip(REWARD_COMPONENTS, reward_vector.tolist())),
        **events,
    }


def event_rewards(
    target_cleared: bool,
    mission_done: bool,
    weights: Dict[str, float],
) -> Dict[str, float]:
    """Sparse bonuses for this step, keyed by EVENT_COMPONENTS (0.0 when the
    corresponding event did not fire).
    """
    return {
        "target_cleared_bonus": float(weights.get("target_cleared_bonus", 0.0))
        if target_cleared
        else 0.0,
        "mission_done_bonus": float(weights.get("mission_done_bonus", 0.0))
        if mission_done
        else 0.0,
    }


def pointing_error_penalty(e_current: float) -> float:
    """Dense, monotone cost on the pointing error, in [-1, 0].

    This is the only term that has a gradient at large pointing errors. The
    differential term telescopes to ~0 over an episode (it sums to
    e_start - e_end, so it only shapes *local* progress), and the boundary-layer
    Gaussian is numerically flat beyond roughly 3*sigma. Without this term the
    reward landscape at a typical ~100deg initial error is essentially flat, so
    there is nothing for the policy to climb toward the target.

    e_current is the quaternion vector-part magnitude, sin(theta/2), which is
    monotone in theta over [0, 180deg] -- so this is 0 on target and -1 at
    180deg off.
    """
    return -e_current


def boundary_layer_reward(e_current: float, sigma: float) -> float:
    """
    Gives the agent a constant reward for holding within the desired
    accuracy level. This is necessary because the differential reward
    vanishes near the target and can't sustain fine-holding control on
    its own without it, the agent gets no positive reinforcement for
    staying on target.
    """
    return float(np.exp(-(e_current**2) / (2.0 * sigma**2)))


def rw_saturation_penalty(rw_speeds: np.ndarray, rw_max_speed: float) -> float:
    """Quadratic penalty that grows as reaction wheel speeds approach saturation."""
    return -float(np.sum((rw_speeds / rw_max_speed) ** 2))


def slew_rate_boundary_penalty(omega_body: np.ndarray, omega_thresh: float) -> float:
    """Quadratic penalty for body rates exceeding the slew-rate threshold, zero otherwise."""
    speed = float(np.linalg.norm(omega_body))
    return -(max(0.0, speed - omega_thresh) ** 2)

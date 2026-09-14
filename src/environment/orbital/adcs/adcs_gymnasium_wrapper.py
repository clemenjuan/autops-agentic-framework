"""
Gymnasium Wrapper for ADCS simulation Environment.

Wraps dynamics.py as a standard gymnasium.Env with:
- observation_space: Box(11,) 
- action_space: Box(7,) kept between [-1,1]
The action space consists of:  
- 4 torque commands for the reaction wheels 
- 3 dipole commands for the magnetorquers 

The observation vector is a 11D and normalized 
- 4D current quaternion error 
- 3D spacecraft angular velocity 
- 4D Reaction wheels speeds 
  """
from __future__ import annotations

import gymnasium as gym 
from gymnasium import spaces
from gymnasium.envs.registration import register

from typing import Any, Dict, Optional
import numpy as np
from dataclasses import replace

from src.environment.orbital.propagator import get_environment, configure

from src.environment.orbital.adcs.simulation import (step, initial_state)
from src.environment.orbital.adcs.eventsat import (_WHEEL_MAX_TORQUE, _MTQ_MAX_DIPOLE, _WHEEL_MAX_MOMENTUM, _WHEEL_INERTIA, satellite, sim, actuators, sensors, env, orbit)
from src.environment.orbital.adcs.estimator import initial_estimator_state
from src.environment.orbital.adcs.sensors import initial_sensor_state
from src.environment.orbital.adcs.actuators import ControlCommand
from src.environment.orbital.adcs.adcs_rewards import (
    EVENT_COMPONENTS,
    REWARD_COMPONENTS,
    get_total_reward,
)
from src.environment.orbital.adcs.configs import AdcsEnvConfig
from src.mission.slew_sequence_mission import SlewSequenceMission


use_state_flag = True


_WHEEL_MAX_SPEED = _WHEEL_MAX_MOMENTUM/_WHEEL_INERTIA #TODO adjust to fixed value if available

register(
    id="gymnasium_env/adcs_sim-v0",
    entry_point="src.environment.orbital.adcs.adcs_gymnasium_wrapper:EventSatEnv"
)


class EventSatEnv(gym.Env):

    def __init__(self, config:Optional[AdcsEnvConfig] = None)->None:
        self.cfg = config or env
        self.mission = SlewSequenceMission(self.cfg.mission)
        

        self.max_action = np.concatenate([np.repeat([_WHEEL_MAX_TORQUE], 4), 
                                          np.repeat([_MTQ_MAX_DIPOLE], 3)])

        self.satellite = satellite
        self.sim = replace(sim, seed= self.cfg.seed)
        self.actuators = actuators
        self.sensors = sensors
        configure(orbit)


        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(7,),
            dtype=np.float32
        )

        high=np.ones(11, dtype=np.float32)
        low = -high

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

        return None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        
        self._step_count = 0
       
        
        # initial_state leaves r_eci/v_eci zeroed -- only step() fills them, from
        # the propagator. Without this patch the state reports the satellite at
        # the centre of the Earth until the first step lands, which is wrong in
        # anything that reads the orbit off reset(): a recorded episode, a plot,
        # a ground track. simulation.run does the same thing for the same reason.
        t0 = self.cfg.start_step * self.sim.step_s
        env0 = get_environment(t0)
        self.state = replace(
            initial_state(t0, len(actuators.reaction_wheels)),
            r_eci=env0.r_eci,
            v_eci=env0.v_eci,
        )
        self.sensor_state = initial_sensor_state(self.sensors, self.np_random)
        self.estimator = initial_estimator_state()
        self.mission_state = self.mission.reset(self.state, self.np_random)
        
        # The opening pointing error, so reset() reports the same keys step()
        # does rather than a differently shaped info dict.
        true_error_deg = self._get_error_angle(
            self._compute_error_quat(self.state.q_eci_body, self.mission_state.setpoint)
        )

        obs = self._get_obs(self.state, self.estimator, self.mission_state.setpoint)
        info = self._get_info(true_error_deg)
        return obs, info

    def step(self, action):

        """
        Possible solution is to further optionally pass the action into the step function
        and maybe a flag if we are using rl at the moment according to this flag we use either 
        the provided input action or compute the action from the internal controller.
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        physical_torques = np.multiply(action, self.max_action)
        command = ControlCommand(wheel_commands=physical_torques[0:4],
                                mtq_commands=physical_torques[4:7],)

        self.state, self.sensor_state, self.estimator = step(self.state, self.sensor_state, self.estimator, self.sensors,
                                                              self.actuators,self.satellite, self.mission_state.setpoint, self.sim, self.np_random, 
                                                              command)
        
       
        true_error_quaternion = self._compute_error_quat(self.state.q_eci_body, self.mission_state.setpoint)
        true_error_deg = self._get_error_angle(true_error_quaternion)

        self.mission_state, mission_event= self.mission.update(
            ms= self.mission_state,
            state= self.state,
            env_data= get_environment(self.state.t),
            pointing_error_deg = true_error_deg,
            dt = self.sim.step_s
        )

        # This sets the estimates equal to the true state while the estimator 
        # is not yet implemented. TODO this can be removed once the estimator has a proper implementation
        if use_state_flag:
            self.estimator.q_estimate = self.state.q_eci_body
            self.estimator.omega_estimate = self.state.omega_body

        obs = self._get_obs(self.state, self.estimator, self.mission_state.setpoint)

        reward_info = get_total_reward(
            err_quat = true_error_quaternion,
            omega_wheels = self.state.wheel_speeds,
            omega_wheels_max = _WHEEL_MAX_SPEED,    
            omega_body = self.state.omega_body,
            omega_body_thresh = self.cfg.body_rate_thresh,
            sigma = self.cfg.mission.sigma,
            weights = self.cfg.reward_weights,
            target_cleared = mission_event.target_cleared,  
            mission_done = mission_event.mission_done
        )
        reward = reward_info['reward']

        info = self._get_info(true_error_deg, reward_info)

        self._step_count += 1

        # Mission success is the only termination: it is the one genuinely
        # absorbing state. There is deliberately NO tumble check, because the
        # satellite cannot tumble. Every actuator saturated for 4000 steps
        # reaches 0.29 rad/s and then stays flat -- the wheels can only trade
        # ~0.105 rad/s of momentum with the body, and the magnetorquers add
        # external momentum at ~1.5e-5 N*m against a 25 uT field. Disturbances
        # do not change this: they are ~1e-6 N*m and oscillatory rather than
        # secular (the rate envelope saturates near 0.009 rad/s over a full
        # orbit of free drift), so they contribute ~0.002 rad/s per episode.
        # cfg.max_body_rate is therefore an observation scale only, never a
        # limit -- nothing enforces it, and nothing should, at this value.
        #
        # This changes if episodes ever start from a randomised tumbling
        # attitude, or if the threshold drops into the reachable band. A rate
        # cutoff then has to be reported as `truncated`, not `terminated`: a
        # spinning satellite is not an absorbing state, and zeroing the return
        # instead of bootstrapping V(s) would make spinning up the cheapest way
        # to escape the pointing penalty.
        terminated = mission_event.mission_done
        truncated = self._check_truncated()

        return obs, reward, terminated, truncated, info
        

    def render(self):
        pass

    def _get_obs(self,state, estimator, setpoint):
        """The estimator's running estimate of the satellite state.
            Attributes:
                q_estimate: Estimated attitude quaternion (ECI to body), shape (4,),
                    scalar-first.
                omega_estimate: Estimated angular velocity in the body frame [rad/s],
                    shape (3,).
                bias_estimate: Estimated gyro bias [rad/s], shape (3,).
                covariance: Error-state covariance, shape (6, 6). The 6-dim error state
                    is [attitude error (3); gyro-bias error (3)].
            """
        q_error = self._compute_error_quat(estimator.q_estimate, setpoint) 
        omega_body = estimator.omega_estimate / self.cfg.max_body_rate
        wheel_speeds = state.wheel_speeds / _WHEEL_MAX_SPEED
        obs = np.concatenate([q_error, omega_body, wheel_speeds]).astype(np.float32)
        obs = np.clip(obs, -1.0, 1.0)
        return obs

    def _get_info(self, pointing_error_deg, reward_info=None):
        """Per-step diagnostics returned alongside the observation.

        Three groups, prefixed so they stay separable in a metrics backend:

        ``mission/`` whatever the mission reports about its own progress.
        ``adcs/``    physical quantities the reward acts on but never exposes:
                     how far off target, how fast the body is turning, and how
                     close the wheels are to saturation.
        ``reward/``  the individual weighted reward terms. The scalar reward
                     cannot say *which* term a policy is optimising -- one that
                     sits still to dodge the wheel-saturation penalty and one
                     that actually slews can score alike and look nothing like
                     each other here.

        Gymnasium expects reset() and step() to report the same keys, so on
        reset the reward terms are zero: nothing has been earned yet. The names
        come from adcs_rewards, so a component added there appears here without
        a change.

        Args:
            pointing_error_deg: Angle to the current target [deg], computed
                from the true attitude by the caller.
            reward_info: The dict from get_total_reward, or None at reset.

        Returns:
            A flat dict of scalars. No arrays: this is built every step, and
            metrics backends drop anything that is not a number anyway.
        """
        info = self.mission.info(self.mission_state)

        info.update({
            "adcs/pointing_error_deg": float(pointing_error_deg),
            "adcs/body_rate": float(np.linalg.norm(self.state.omega_body)),
            # Max rather than mean: one saturated wheel already costs an axis.
            "adcs/wheel_speed_frac": float(
                np.max(np.abs(self.state.wheel_speeds)) / _WHEEL_MAX_SPEED
            ),
        })

        components = REWARD_COMPONENTS + EVENT_COMPONENTS
        if reward_info is None:
            info.update({f"reward/{name}": 0.0 for name in components})
        else:
            # "reward" itself is left out: it is the step() return value, and a
            info.update({f"reward/{name}": float(reward_info[name]) for name in components})

        return info

    def _compute_error_quat(self, q_body, setpoint):
        q_w = q_body[0]
        quat_vect = q_body[1:4]

        target_scalar  = setpoint.target_q_eci_body[0]
        target_vector = setpoint.target_q_eci_body[1:4]

        # Scalar error
        quat_scalar_error = q_w * target_scalar + np.dot(quat_vect, target_vector)

        # Vector error
        quat_vector_error = q_w * target_vector - target_scalar * quat_vect - np.cross(quat_vect, target_vector)

        quat_error = np.concatenate([np.array([quat_scalar_error]), quat_vector_error])

        # A quaternion and its negation represent the same rotation; forcing a
        # positive scalar part picks the shortest-path (< 180 deg) representation.
        if quat_scalar_error < 0:
            quat_error = -quat_error
        return quat_error



    def _check_truncated(self):
        return (self._step_count >= self.cfg.mission.max_steps)


    def _get_error_angle(self,true_error_quaternion):
        #unpack quaternion error and calculate the angle error 
        pointing_error_scalar = true_error_quaternion[0]
        pointing_error_vector = true_error_quaternion[1:4]
        pointing_error_vector_mag = np.linalg.norm(pointing_error_vector)
        pointing_error_angle_rad = 2.0 * np.arctan2(pointing_error_vector_mag, np.abs(pointing_error_scalar))
        pointing_error_angle_deg = 180 * pointing_error_angle_rad/np.pi

        return pointing_error_angle_deg

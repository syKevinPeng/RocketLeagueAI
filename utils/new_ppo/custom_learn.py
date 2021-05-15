
from typing import Any, Dict, Optional, Type, Union

# from stable_baselines3.common import logger
# from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
# from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
# from stable_baselines3.common.utils import explained_variance, get_schedule_fn
# from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from stable_baselines3.common import logger
# from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback#, Schedule
from stable_baselines3.common.utils import safe_mean # obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

import time
import numpy as np
import torch as th
import gym


# def custom_collect_rollouts(
#         model,
#         env: VecEnv,
#         callback: BaseCallback,
#         rollout_buffer: RolloutBuffer,
#         n_rollout_steps: int,
#     ) -> bool:
#         """
#         Collect experiences using the current policy and fill a ``RolloutBuffer``.
#         The term rollout here refers to the model-free notion and should not
#         be used with the concept of rollout used in model-based RL or planning.

#         :param env: The training environment
#         :param callback: Callback that will be called at each step
#             (and at the beginning and end of the rollout)
#         :param rollout_buffer: Buffer to fill with rollouts
#         :param n_steps: Number of experiences to collect per environment
#         :return: True if function returned with at least `n_rollout_steps`
#             collected, False if callback terminated rollout prematurely.
#         """
#         assert model._last_obs is not None, "No previous observation was provided"
#         n_steps = 0
#         rollout_buffer.reset()
#         # Sample new weights for the state dependent exploration
#         if model.use_sde:
#             model.policy.reset_noise(env.num_envs)

#         callback.on_rollout_start()

#         while n_steps < n_rollout_steps:
#             if model.use_sde and model.sde_sample_freq > 0 and n_steps % model.sde_sample_freq == 0:
#                 # Sample a new noise matrix
#                 model.policy.reset_noise(env.num_envs)

#             with th.no_grad():
#                 # Convert to pytorch tensor or to TensorDict
#                 obs_tensor = obs_as_tensor(model._last_obs, model.device)
#                 actions, values, log_probs = model.policy.forward(obs_tensor)
#             actions = actions.cpu().numpy()

#             # Rescale and perform action
#             clipped_actions = actions
#             # Clip the actions to avoid out of bound error
#             if isinstance(model.action_space, gym.spaces.Box):
#                 clipped_actions = np.clip(actions, model.action_space.low, model.action_space.high)

#             new_obs, rewards, dones, infos = env.step(clipped_actions)

#             model.num_timesteps += env.num_envs

#             # Give access to local variables
#             callback.update_locals(locals())
#             if callback.on_step() is False:
#                 return False

#             model._update_info_buffer(infos)
#             n_steps += 1

#             if isinstance(model.action_space, gym.spaces.Discrete):
#                 # Reshape in case of discrete action
#                 actions = actions.reshape(-1, 1)
#             rollout_buffer.add(model._last_obs, actions, rewards, model._last_episode_starts, values, log_probs)
#             model._last_obs = new_obs
#             model._last_episode_starts = dones

#         with th.no_grad():
#             # Compute value for the last timestep
#             obs_tensor = obs_as_tensor(new_obs, model.device)
#             _, values, _ = model.policy.forward(obs_tensor)

#         rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

#         callback.on_rollout_end()

#         return True

def custom_collect_rollouts(
        model, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
    """
    Collect experiences using the current policy and fill a ``RolloutBuffer``.
    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.

    :param env: The training environment
    :param callback: Callback that will be called at each step
        (and at the beginning and end of the rollout)
    :param rollout_buffer: Buffer to fill with rollouts
    :param n_steps: Number of experiences to collect per environment
    :return: True if function returned with at least `n_rollout_steps`
        collected, False if callback terminated rollout prematurely.
    """
    assert model._last_obs is not None, "No previous observation was provided"
    n_steps = 0
    rollout_buffer.reset()
    # Sample new weights for the state dependent exploration
    if model.use_sde:
        model.policy.reset_noise(env.num_envs)

    callback.on_rollout_start()

    while n_steps < n_rollout_steps:
        if model.use_sde and model.sde_sample_freq > 0 and n_steps % model.sde_sample_freq == 0:
            # Sample a new noise matrix
            model.policy.reset_noise(env.num_envs)

        with th.no_grad():
            # Convert to pytorch tensor
            obs_tensor = th.as_tensor(model._last_obs).to(model.device)
            actions, values, log_probs = model.policy.forward(obs_tensor)
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(model.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, model.action_space.low, model.action_space.high)

        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------

        # print('env and env type: ',env,type(env))

        rot = 65536/2
        # new_obs, rewards, dones, infos = env.step(clipped_actions)
        new_obs, rewards, dones, (done_exer,exer_state), infos = env.step(clipped_actions, # action
                                                                   reset_at_term_exer=True,
                                                                   which_exer=0, 
                                                                   exercise_reset_states=[([0, 5000, 98],[400,4900,17],[0,rot,0])])

        if done_exer: new_obs = env.reset_to_exer_state(exer_state)

        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------

        model.num_timesteps += env.num_envs

        # Give access to local variables
        callback.update_locals(locals())
        if callback.on_step() is False:
            return False

        model._update_info_buffer(infos)
        n_steps += 1

        if isinstance(model.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)
        rollout_buffer.add(model._last_obs, actions, rewards, model._last_dones, values, log_probs)
        model._last_obs = new_obs
        model._last_dones = dones

    with th.no_grad():
        # Compute value for the last timestep
        obs_tensor = th.as_tensor(new_obs).to(model.device)
        _, values, _ = model.policy.forward(obs_tensor)

    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    callback.on_rollout_end()

    return True

def custom_learn(
        model,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True
    ):

    iteration = 0

    total_timesteps, callback = model._setup_learn(
        total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
    )

    callback.on_training_start(locals(), globals())

    while model.num_timesteps < total_timesteps:

        continue_training = custom_collect_rollouts(model, model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)

        if continue_training is False:
            break

        iteration += 1
        model._update_current_progress_remaining(model.num_timesteps, total_timesteps)

        # Display training infos
        if log_interval is not None and iteration % log_interval == 0:
            fps = int(model.num_timesteps / (time.time() - model.start_time))
            logger.record("time/iterations", iteration, exclude="tensorboard")
            if len(model.ep_info_buffer) > 0 and len(model.ep_info_buffer[0]) > 0:
                logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer]))
                logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in model.ep_info_buffer]))
            logger.record("time/fps", fps)
            logger.record("time/time_elapsed", int(time.time() - model.start_time), exclude="tensorboard")
            logger.record("time/total_timesteps", model.num_timesteps, exclude="tensorboard")
            logger.dump(step=model.num_timesteps)

        model.train()

    callback.on_training_end()

    return model

import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.reward_functions.common_rewards import *
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from stable_baselines3 import PPO # pip install stable-baselines3[extra]
import time

#The desired number of seconds we would like to wait before terminating an episode.
ep_len_seconds = 120

#By default, RLGym will repeat every action for 8 physics ticks before waiting for a new action from our agent.
default_tick_skip = 8

#The tick rate of the Rocket League physics engine.
physics_ticks_per_sec = 120
max_steps = int(round(ep_len_seconds * physics_ticks_per_sec / default_tick_skip))

timeout_condition = TimeoutCondition(max_steps)
reward_function = GoalReward
obs_builder = AdvancedObs()
terminal_conditions = [timeout_condition,]


#All we have to do now is pass our custom configuration objects to rlgym!
env = rlgym.make("default",
                 spawn_opponents=False,
                 # reward_fn=reward_function,
                 obs_builder=obs_builder,
                 terminal_conditions=terminal_conditions)

#Initialize PPO from stable_baselines3
model = PPO("MlpPolicy", env=env, verbose=1, device='cuda')

if __name__ == "__main__":
    model.learn(total_timesteps=int(1e6))
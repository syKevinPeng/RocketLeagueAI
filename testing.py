import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.reward_functions.common_rewards import GoalReward, MoveTowardsBallReward
from rlgym.utils.reward_functions.combined_reward import *
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from customized_reward import TimeReward, TouchBallReward
from cnnLstm_policy import CustomActorCriticPolicy

#The desired number of seconds we would like to wait before terminating an episode.
ep_len_seconds = 120

#By default, RLGym will repeat every action for 8 physics ticks before waiting for a new action from our agent.
default_tick_skip = 8

#The tick rate of the Rocket League physics engine.
physics_ticks_per_sec = 120
max_steps = int(round(ep_len_seconds * physics_ticks_per_sec / default_tick_skip))

timeout_condition = TimeoutCondition(max_steps)
reward_function = CombinedReward((TouchBallReward(), GoalReward(per_goal=5.0), MoveTowardsBallReward(), TimeReward()), (0.2, 1.0, 0.1, 0.1))
obs_builder = AdvancedObs()
terminal_conditions = [timeout_condition,]


#All we have to do now is pass our custom configuration objects to rlgym!
env = rlgym.make("default",
                 spawn_opponents=False,
                 game_speed=1,
                 reward_fn=reward_function,
                 obs_builder=obs_builder,
                 terminal_conditions=terminal_conditions)


if __name__ == "__main__":
    # Initialize PPO from stable_baselines3
    model = PPO('MlpPolicy', env=env, verbose=1, device='cuda')
    model.load("attack_agent.zip")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # Enjoy trained agent
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.reward_functions.common_rewards import GoalReward
from rlgym.utils.reward_functions.combined_reward import *
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs

from utils.new_ppo.custom_ppo  import PPO
from utils.customized_reward import TimeReward, TouchBallReward, LinearDistanceReward, MoveTowardsBallReward
from utils.cnnLstm_policy import CustomActorCriticPolicy
from utils.new_ppo.custom_learn import custom_learn
from utils.custom_terminal_conditions import ExerciseTimeoutCondition, ExerciseGoalScoredCondition

#The desired number of seconds we would like to wait before terminating an episode.
ep_len_seconds = 120
exer_len_seconds = 10

#By default, RLGym will repeat every action for 8 physics ticks before waiting for a new action from our agent.
default_tick_skip = 8

#The tick rate of the Rocket League physics engine.
physics_ticks_per_sec = 120
max_steps = int(round(ep_len_seconds * physics_ticks_per_sec / default_tick_skip))
exer_steps = int(round(exer_len_seconds * physics_ticks_per_sec / default_tick_skip))

exer_timeout_condition = ExerciseTimeoutCondition(max_steps=max_steps,       # num_steps before episode resets
                                                  max_exer_steps=exer_steps, # num_steps before exercise repeats
                                                  randomize_exer=False)

# TouchBallReward: get +1 reward when touch ball
# MoveTowardsBallReward: The projection of linear velocity over distance (maximum value 24.5)
# GoalReward: reward for each goal can be controlled via per_goal parameter
# MoveTowardsGoalReward: project of linear velocity over distance
# TimeReward: reward -1 for every second

reward_weights = {
'TouchBallReward': 50,
'LinearDistanceReward': 1/100000,
'GoalReward': 100,
'MoveTowardsBallReward':1/300,
'TimeReward':1
}
# reward_weights = {
# 'TouchBallReward': 1,
# 'LinearDistanceReward': 0,
# 'GoalReward': 5,
# 'MoveTowardsBallReward':0.1,
# 'TimeReward':0.1
# }

# timeout_condition = TimeoutCondition(max_steps)
reward_function = CombinedReward((TouchBallReward(),
                                  LinearDistanceReward(max_reward=2),
                                  GoalReward(),
                                  MoveTowardsBallReward(),
                                  TimeReward(per_sec=-1.0)),
                                 (reward_weights['TouchBallReward'],
                                  reward_weights['LinearDistanceReward'],
                                  reward_weights['GoalReward'],
                                  reward_weights['MoveTowardsBallReward'],
                                  reward_weights['TimeReward']))
obs_builder = AdvancedObs()
terminal_conditions = [exer_timeout_condition] # [GoalScoredCondition(),TimeoutCondition()]


#All we have to do now is pass our custom configuration objects to rlgym!
env = rlgym.make("default",
                 spawn_opponents=False,
                 game_speed=2,
                 reward_fn=reward_function,
                 obs_builder=obs_builder,
                 terminal_conditions=terminal_conditions,
                 random_resets= True)

#Initialize PPO from stable_baselines3
model = PPO('MlpPolicy', env=env, verbose=1, device='cuda', tensorboard_log="./logs/")
model.save("attack_agent")

if __name__ == "__main__":
    # model.learn(total_timesteps=int(1e6))
    custom_learn(model,total_timesteps=int(1e6))
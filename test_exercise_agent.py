import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.reward_functions.common_rewards import GoalReward, MoveTowardsBallReward
from rlgym.utils.reward_functions.combined_reward import *
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from stable_baselines3 import PPO
from utils.customized_reward import TimeReward, TouchBallReward, LinearDistanceReward
from stable_baselines3.common.evaluation import evaluate_policy
from utils.give_exercises import some_examples
from utils.give_exercises import convert_exercises
from utils.custom_terminal_conditions import ExerciseTimeoutCondition, ExerciseGoalScoredCondition

# reset states for exercises
exercises = some_examples(num_examples=4)
exercises.pop(2) # remove slightlyoffset_ball_car_vals
# exercises.pop(2) # remove farpostbounce_ball_car_vals
exercises = sum(exercises,[]) # flatten
exercise_reset_states = convert_exercises(exercises, flip_across_midline=False)
exercise_reset_states_flip = convert_exercises(exercises, flip_across_midline=True)
all_exercises = exercise_reset_states + exercise_reset_states_flip

# helper vars for resetting exercises
from math import inf
num_exer = len(exercise_reset_states)
max_goals_per_episode = inf
which_exer = 0

#The desired number of seconds for each exercise
exer_len_seconds = 10 # must be greater than 3, will not work during countdown

#The desired number of seconds we would like to wait before terminating an episode.
ep_len_seconds = exer_len_seconds*num_exer

#By default, RLGym will repeat every action for 8 physics ticks before waiting for a new action from our agent.
default_tick_skip = 8

#The tick rate of the Rocket League physics engine.
physics_ticks_per_sec = 120
max_steps = int(round(ep_len_seconds * physics_ticks_per_sec / default_tick_skip))
exer_steps = int(round(exer_len_seconds * physics_ticks_per_sec / default_tick_skip))

which_exer = 0
# timeout_condition = TimeoutCondition(max_steps=)
# goalscore_condition = GoalScoredCondition()

exer_timeout_condition = ExerciseTimeoutCondition(max_steps=max_steps,       # num_steps before episode resets
                                                  max_exer_steps=exer_steps, # num_steps before exercise repeats
                                                  randomize_exer=False)
exer_goalscored_condition = ExerciseGoalScoredCondition(max_episode_goals=max_goals_per_episode, 
                                                        randomize_exer=False)

# TouchBallReward: get +1 reward when touch ball
# MoveTowardsBallReward: The projection of linear velocity over distance (maximum value 24.5)
# GoalReward: reward for each goal can be controlled via per_goal parameter
# MoveTowardsGoalReward: ? not sure
# TimeReward: reward -1 for every second
reward_weights = {
'TouchBallReward': 50,
'LinearDistanceReward': 1/100000,
'GoalReward': 100,
'MoveTowardsBallReward':1/300,
'TimeReward':1
}

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
terminal_conditions = [exer_goalscored_condition,exer_timeout_condition] # [GoalScoredCondition(),TimeoutCondition()]

if __name__ == "__main__":
    env = rlgym.make("default",
                     spawn_opponents=False,
                     game_speed=1,
                     reward_fn=reward_function,
                     obs_builder=obs_builder,
                     terminal_conditions=terminal_conditions,
                     reset_at_term_exer=True,
                     exercise_reset_states=all_exercises)
    # Initialize PPO from stable_baselines3
    model = PPO('MlpPolicy', env=env, device='cuda')
    model.load("./models/attack_agent_Vlad_39e5.zip")
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # Enjoy trained agent
    obs = env.reset()
    for i in range(5000):
        action, _states = model.predict(obs, deterministic=True)
        print(action)
        obs, rewards, dones, info = env.step(action)
        # env.render()
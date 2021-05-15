import rlgym
import numpy as np
from math import inf

from rlgym.utils.gamestates import GameState, PlayerData, PhysicsObject
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym.utils.reward_functions.common_rewards import GoalReward, MoveTowardsBallReward
from rlgym.utils.reward_functions.combined_reward import *
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
# from customized_reward import TimeReward, TouchBallReward
from utils.cnnLstm_policy import CustomActorCriticPolicy

from utils.give_exercises import some_examples
from utils.give_exercises import convert_exercises
from utils.custom_terminal_conditions import ExerciseTimeoutCondition, ExerciseGoalScoredCondition

# reset states for exercises
exercises = some_examples(num_examples=4)
exercises.pop(2) # remove slightlyoffset_ball_car_vals
exercises = sum(exercises,[]) # flatten
exercise_reset_states = convert_exercises(exercises)

# helper vars for resetting exercises
num_exer = len(exercise_reset_states)
max_goals_per_episode = 9
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

reward_function = GoalReward()
                #   CombinedReward((#TouchBallReward(), 
                #                   GoalReward(per_goal=5.0), 
                #                   MoveTowardsBallReward(), 
                #                   TimeReward()), 
                #                   (0.2, 1.0, 0.1, 0.1))
obs_builder = AdvancedObs()
terminal_conditions = [exer_goalscored_condition,exer_timeout_condition] # [timeout_condition,goalscore_condition]


#All we have to do now is pass our custom configuration objects to rlgym!
env = rlgym.make("default",
                 spawn_opponents=False,
                 game_speed=3,
                 reward_fn=reward_function,
                 obs_builder=obs_builder,
                 terminal_conditions=terminal_conditions,
                 please_reset_at_end_of_exer=True,
                 num_exer=num_exer)


if __name__ == "__main__":
    # Initialize PPO from stable_baselines3
    obs = env.reset()
    model = PPO('MlpPolicy', env=env, verbose=1, device='cuda')
    model.load("attack_agent.zip")
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    
    # Enjoy trained agent
    empty_action = [0.]*8
    drive_forward = [1.] + [0.]*7
    for i in range(3200):
        # action, _states = model.predict(obs, deterministic=True)
        # obs, rewards, dones, info = env.step(empty_actions) # env.step(action)
        if i%250 == 0: print(i)
        action = drive_forward if i<1000 else empty_action

        obs, reward, done, (done_exer,exer_state), info = env.step(action, # action
                                                                   reset_at_term_exer=True,
                                                                   which_exer=which_exer, 
                                                                   exercise_reset_states=exercise_reset_states)
        if done_exer: 
            obs = env.reset_to_exer_state(exer_state)

        if done: # or which_exer == num_exer-1:
            print("Resetting Episode")
            obs = env.reset()
            # obs = env.reset_to_exer_state(exer_state)

        if done or done_exer:
            # increment which_exer
            if which_exer < num_exer-1: which_exer += 1
            else: which_exer = 0
        
        # env.render()
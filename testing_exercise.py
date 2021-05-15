import rlgym
import numpy as np

from rlgym.utils.gamestates import GameState, PlayerData, PhysicsObject
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.reward_functions.common_rewards import GoalReward, MoveTowardsBallReward
from rlgym.utils.reward_functions.combined_reward import *
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
# from customized_reward import TimeReward, TouchBallReward
from cnnLstm_policy import CustomActorCriticPolicy

#The desired number of seconds we would like to wait before terminating an episode.
ep_len_seconds = 120

#By default, RLGym will repeat every action for 8 physics ticks before waiting for a new action from our agent.
default_tick_skip = 8

#The tick rate of the Rocket League physics engine.
physics_ticks_per_sec = 120
max_steps = int(round(ep_len_seconds * physics_ticks_per_sec / default_tick_skip))

timeout_condition = TimeoutCondition(max_steps)
reward_function = GoalReward()
                #   CombinedReward((#TouchBallReward(), 
                #                   GoalReward(per_goal=5.0), 
                #                   MoveTowardsBallReward(), 
                #                   TimeReward()), 
                #                   (0.2, 1.0, 0.1, 0.1))
obs_builder = AdvancedObs()
terminal_conditions = [timeout_condition]


#All we have to do now is pass our custom configuration objects to rlgym!
env = rlgym.make("default",
                 spawn_opponents=False,
                 game_speed=1,
                 reward_fn=reward_function,
                 obs_builder=obs_builder,
                 terminal_conditions=terminal_conditions)

# car_obj
# inv_car_obj
# ball_obj
# inv_ball_obj
# player_obj
# exercise_state_obj

car_dict = \
    {'position': [0., 200.,    17.  ], 
    'quaternion': [ 0.853726  , -0.0024897 ,  0.00409195,  0.5207    ], 
    'linear_velocity':  [0., 0., 0.341],
    'angular_velocity': [0.,  0.     ,  0.     ], 
    '_euler_angles': np.array([0., 0., 0.]), 
    '_rotation_mtx': np.array([[0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.]]), 
    '_has_computed_rot_mtx': False, 
    '_has_computed_euler_angles': False}
inverted_car_dict = \
    {'position': [0., 200.,   17.  ], 
    'quaternion': [ 0.5207    ,  0.00409195,  0.0024897 , -0.853726  ], 
    'linear_velocity':  [0.   , -0.   ,  0.341], 
    'angular_velocity': [0., 0. , 0. ], 
    '_euler_angles': np.array([0., 0., 0.]), 
    '_rotation_mtx': np.array([[0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.]]), 
    '_has_computed_rot_mtx': False, 
    '_has_computed_euler_angles': False}
ball_dict = \
       {'position': [ 0.  ,  0.  , 92.74], 
        'quaternion': [1., 1., 1., 1.], 
        'linear_velocity': [0., 0., 0.], 
        'angular_velocity': [0., 0., 0.], 
        '_euler_angles': np.array([0., 0., 0.]), 
        '_rotation_mtx': np.array([[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.]]), 
        '_has_computed_rot_mtx': False, 
        '_has_computed_euler_angles': False}
inv_ball_dict = \
     {'position': [-0.  , -0.  , 92.74],
     'quaternion': [1., 1., 1., 1.],
     'linear_velocity': [-0., -0.,  0.],
     'angular_velocity': [-0., -0.,  0.],
     '_euler_angles': np.array([0., 0., 0.]),
     '_rotation_mtx': np.array([[0., 0., 0.],
                             [0., 0., 0.],
                             [0., 0., 0.]]),
     '_has_computed_rot_mtx': False,
     '_has_computed_euler_angles': False}

car_obj = PhysicsObject(position=car_dict['position'], 
                        quaternion=car_dict['quaternion'], 
                        linear_velocity=car_dict['linear_velocity'], 
                        angular_velocity=car_dict['angular_velocity'])
inv_car_obj = PhysicsObject(position=inverted_car_dict['position'], 
                            quaternion=inverted_car_dict['quaternion'], 
                            linear_velocity=inverted_car_dict['linear_velocity'], 
                            angular_velocity=inverted_car_dict['angular_velocity'])
ball_obj = PhysicsObject(position=ball_dict['position'], 
                        quaternion=ball_dict['quaternion'], 
                        linear_velocity=ball_dict['linear_velocity'], 
                        angular_velocity=ball_dict['angular_velocity'])
inv_ball_obj = PhysicsObject(position=inv_ball_dict['position'], 
                            quaternion=inv_ball_dict['quaternion'], 
                            linear_velocity=inv_ball_dict['linear_velocity'], 
                            angular_velocity=inv_ball_dict['angular_velocity'])
player_dict = \
           {'car_id': 1, 
            'team_num': 0, 
            'match_goals': 0, 
            'match_saves': 0, 
            'match_shots': 0, 
            'match_demolishes': 0, 
            'boost_pickups': 0, 
            'is_demoed': False, 
            'on_ground': True, 
            'ball_touched': False, 
            'has_flip': True, 
            'boost_amount': 0.33, 
            'car_data': car_obj, 
            'inverted_car_data': inv_car_obj}

player_obj = PlayerData()
player_obj.set_vals_from_dict(player_dict)

exercise_state_dict = {
    'game_type': 0, 
    'blue_score': 0, 
    'orange_score': 0, 
    'last_touch': -1, 
    'players': [player_obj], 
    'ball': ball_obj, 
    'inverted_ball': inv_ball_obj, 
    'boost_pads': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                         dtype=np.float32), 
    'inverted_boost_pads': np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                  dtype=np.float32)
}

exercise_state_obj = GameState()
exercise_state_obj.set_vals_from_dict(exercise_state_dict)

# from pprint import pprint
# print("car_obj vals: ")
# pprint(car_obj.__dict__)
# print("inv_car_obj vals: ")
# pprint(inv_car_obj.__dict__)
# print()
# print("ball_obj vals: ")
# pprint(ball_obj.__dict__)
# print("inv_ball_obj vals: ")
# pprint(inv_ball_obj.__dict__)
# print()
# print("exercise_state vals:")
# pprint(exercise_state_obj.__dict__)
from time import time 

if __name__ == "__main__":
    # Initialize PPO from stable_baselines3
    print(1)
    env.reset()
    print(2)
    model = PPO('MlpPolicy', env=env, verbose=1, device='cuda')
    print(3)
    model.load("attack_agent.zip")
    print(4)
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    
    # Enjoy trained agent
    print("BEFORE")
    obs = env.reset_to_state(exercise_state_obj)
    print("THIS IS OBS:",obs)
    t = time()
    for i in range(1000):
        print(i)
        # if round(t)%10 == 0:

        if i % 100 == 0:
            env.reset_to_state(exercise_state_obj)
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        # env.render()
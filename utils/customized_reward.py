import numpy as np

from rlgym.utils import math
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER
from rlgym.utils.gamestates import GameState, PlayerData
from stable_baselines3.common import logger

class TimeReward(RewardFunction):
    def __init__(self, per_sec = -0.1):
        super().__init__()
        self.pre_sec_reward = per_sec

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward_per_tick = self.pre_sec_reward / 120 # The tick rate of the Rocket League physics engine.
        return reward_per_tick

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

class TouchBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 1 if player.ball_touched else 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


"""
Custom linear distance reward function.
maximium reward is 0. The larger the distance between agent and player, the smaller the reward is.
"""
class LinearDistanceReward(RewardFunction):
    def __init__(self, max_reward):
        super().__init__()
        self.largest_dis = 0
        self.max_reward = max_reward

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        max_distance = 12000
        reward = max_distance - abs(np.linalg.norm(state.ball.position - player.car_data.position))
        if reward < 0:
            raise Exception(f"Linear Distance Rewrad is negative: {reward}")
        logger.record("reward/linear_distance_reward", reward)
        return self.max_reward*(reward/max_distance)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0
"""
Custom distance reward function.
maximium rewrad is 0. The larger the distance between agent and player, the smaller the reward is.

:param min: minimum reward of this function i.e. reward given at very large distance
:param last_layer_dim_pi: (int) number of units for the last layer of the policy network
:param last_layer_dim_vf: (int) number of units for the last layer of the value network
"""
class LogDistanceReward(RewardFunction):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return min(max(np.log(np.linalg.norm(state.ball.position - player.car_data.position)), self.min), self.max)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

class MoveTowardsBallReward(RewardFunction):
    def reset(self, initial_state: GameState, optional_data=None):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        inv_t = math.scalar_projection(player.car_data.linear_velocity, state.ball.position - player.car_data.position)
        logger.record("reward/move_to_ball_reward", inv_t)
        return inv_t

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        return 0
import numpy as np

from rlgym.utils import math
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER
from rlgym.utils.gamestates import GameState, PlayerData

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
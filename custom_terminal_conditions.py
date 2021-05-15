
from typing import List
import random

from rlgym.utils.terminal_conditions import ExerciseTerminalCondition
from rlgym.utils.gamestates import GameState

class ExerciseTimeoutCondition(ExerciseTerminalCondition):
    """
    A condition that will terminate an episode after some number of steps.
    """

    def __init__(self, max_steps: int, max_exer_steps: int, randomize_exer: bool):
        super().__init__()
        self.steps = 0
        self.max_steps = max_steps
        
        self.exer_steps = 0
        self.max_exer_steps = max_exer_steps
        
        # self.which_exer = 0
        self.randomize_exer = randomize_exer
        # self.all_exercises = exercises
        self.curr_exercise = None


    def reset(self, initial_state: GameState):
        """
        Reset the step counter.
        """
        self.steps = 0
        # self.which_exer = 0

    def to_next_exer_state(self, initial_state: GameState, which_exer: int, all_exercises:List[str]):
        """
        Reset the step counter.
        Provide reset_state
        """
        self.exer_steps = 0

        if self.randomize_exer:
            self.curr_exercise = all_exercises[random.randint(0, len(all_exercises)-1)]
        else:
            print("Choosing exer:", which_exer)
            self.curr_exercise = all_exercises[which_exer]

            # # increment which_exer
            # if which_exer < len(all_exercises): which_exer += 1
            # else: which_exer = 0

        return self.curr_exercise # which_exer, self.curr_exercise

    def is_terminal(self, current_state: GameState) -> bool:
        """
        Increment the step counter and return `True` if `max_steps` have passed.
        """

        self.steps += 1
        return self.steps >= self.max_steps

    def is_terminal_exer(self, current_state: GameState) -> bool:
        """
        Increment the exer_steps counter and return `True` if `max_exer_steps` have passed.
        """

        self.exer_steps += 1
        return self.exer_steps >= self.max_exer_steps


class ExerciseGoalScoredCondition(ExerciseTerminalCondition):
    """
    A condition that will terminate an episode after some number of steps.
    """

    def __init__(self, max_episode_goals: int, randomize_exer: bool):
        super().__init__()
        self.blue_score = 0
        self.orange_score = 0
        self.episode_goals = 0
        
        self.max_episode_goals = max_episode_goals
        self.randomize_exer = randomize_exer
        # self.all_exercises = exercises
        self.curr_exercise = None
        # self.which_exer = 0

    def reset(self, initial_state: GameState):
        """
        Reset the step counter.
        """
        # self.which_exer = 0
        self.episode_goals = 0

    def to_next_exer_state(self, initial_state: GameState, which_exer: int, all_exercises:List[str]):
        """
        Reset the step counter.
        Provide reset_state
        """

        if self.randomize_exer:
            self.curr_exercise = all_exercises[random.randint(0, len(all_exercises)-1)]
        else:
            print("Choosing exer:", which_exer)
            self.curr_exercise = all_exercises[which_exer]

            # # increment which_exer
            # if which_exer < len(all_exercises): which_exer += 1
            # else: which_exer = 0

        return self.curr_exercise # which_exer, self.curr_exercise

    def is_terminal(self, current_state: GameState) -> bool:
        """
        Increment the current step counter and return `True` if `max_steps` have passed.
        """
        # HUGE flaw, is_terminal and is_terminal_exer both try to update `current_state`
        # HUGE flaw, is_terminal and is_terminal_exer both try to update `current_state`
        # HUGE flaw, is_terminal and is_terminal_exer both try to update `current_state`
        if current_state.blue_score != self.blue_score or current_state.orange_score != self.orange_score:
            self.blue_score = current_state.blue_score
            self.orange_score = current_state.orange_score
            self.episode_goals += 1
            if self.episode_goals >= self.max_episode_goals:
                return True
        return False

    def is_terminal_exer(self, current_state: GameState) -> bool:
        """
        Check to see if the game score for either team has been changed. If either score has changed, update the current
        known scores for both teams and return `True`. Note that the known game scores are never reset for this object
        because the game score is not set to 0 for both teams at the beginning of an episode.
        """
        # HUGE flaw, is_terminal and is_terminal_exer both try to update `current_state`
        # HUGE flaw, is_terminal and is_terminal_exer both try to update `current_state`
        # HUGE flaw, is_terminal and is_terminal_exer both try to update `current_state`
        if current_state.blue_score != self.blue_score or current_state.orange_score != self.orange_score:
            # self.blue_score = current_state.blue_score
            # self.orange_score = current_state.orange_score
            return True

        return False
"""
    The Rocket League gym environment.
"""

from typing import List, Union, Tuple, Dict

import numpy as np
from gym import Env

from rlgym.gamelaunch import launch_rocket_league
from rlgym.communication import CommunicationHandler, Message

class Gym(Env):
    def __init__(self, match, pipe_id=0, path_to_rl=None, use_injector=False, reset_at_term_exer=False, exercise_reset_states=None):
        super().__init__()

        self._match = match
        self.observation_space = match.observation_space
        self.action_space = match.action_space

        self._path_to_rl = path_to_rl
        self._use_injector = use_injector

        self._comm_handler = CommunicationHandler()
        self._local_pipe_name = self._comm_handler.format_pipe_id(pipe_id)
        self._local_pipe_id = pipe_id

        self._game_process = None

        self._open_game()
        self._setup_plugin_connection()

        self._prev_state = None

        self.which_exer = 0
        self.reset_at_term_exer = reset_at_term_exer
        self.exercise_reset_states = exercise_reset_states

    def _open_game(self):
        print("Launching Rocket League, make sure bakkesmod is running.")
        # Game process is only set if launched with path_to_rl
        self._game_process = launch_rocket_league(self._local_pipe_name, self._path_to_rl, self._use_injector)

    def _setup_plugin_connection(self):
        self._comm_handler.open_pipe(self._local_pipe_name)
        self._comm_handler.send_message(header=Message.RLGYM_CONFIG_MESSAGE_HEADER, body=self._match.get_config())

    def reset(self) -> List:
        """
        The environment reset function. When called, this will reset the state of the environment and objects in the game.
        This should be called once when the environment is initialized, then every time the `done` flag from the `step()`
        function is `True`.
        """

        exception = self._comm_handler.send_message(header=Message.RLGYM_RESET_GAME_STATE_MESSAGE_HEADER, body=Message.RLGYM_NULL_MESSAGE_BODY)
        if exception is not None:
            self._attempt_recovery()
            exception = self._comm_handler.send_message(header=Message.RLGYM_RESET_GAME_STATE_MESSAGE_HEADER,
                                                        body=Message.RLGYM_NULL_MESSAGE_BODY)
            if exception is not None:
                import sys
                print("!UNABLE TO RECOVER ROCKET LEAGUE!\nEXITING")
                sys.exit(-1)

        state = self._receive_state()
        self._match.episode_reset(state)
        self._prev_state = state

        return self._match.build_observations(state)

    def reset_to_exer_state(self,to_state=None) -> List:
        """
        The exercise reset function. 
        
        When called, this will reset the state of the environment and objects in the game.
        This should be called once when the environment is initialized, then every time the `done` flag from the `step()`
        function is `True`.
        """

        if to_state is None:
            exception = self._comm_handler.send_message(header=Message.RLGYM_RESET_GAME_STATE_MESSAGE_HEADER,
                                                        body=Message.RLGYM_NULL_MESSAGE_BODY)
        else:
            exception = self._comm_handler.send_message(header=Message.RLGYM_RESET_TO_SPECIFIC_GAME_STATE_MESSAGE_HEADER,
                                                        body=to_state)

        if exception is not None:
            self._attempt_recovery()
            if to_state is None:
                exception = self._comm_handler.send_message(header=Message.RLGYM_RESET_GAME_STATE_MESSAGE_HEADER,
                                                            body=Message.RLGYM_NULL_MESSAGE_BODY)
            else:
                exception = self._comm_handler.send_message(
                    header=Message.RLGYM_RESET_TO_SPECIFIC_GAME_STATE_MESSAGE_HEADER,
                    body=to_state)

            if exception is not None:
                import sys
                print("!UNABLE TO RECOVER ROCKET LEAGUE!\nEXITING")
                sys.exit(-1)

        # we can assume step() has filled the values needed
        # Calling _match.episode_reset() is a MISTAKE, we don't want to reset the episode
        # -----------------------------------
        state = self._receive_state()
        # self._match.episode_reset(state)
        # self._prev_state = state

        return self._match.build_observations(state)

    def step(self, actions: Union[np.ndarray, List[np.ndarray], List[float]]) -> Tuple[List, List, bool, Dict]:
        """
        The step function will send the list of provided actions to the game, then advance the game forward by `tick_skip`
        physics ticks using that action. The game is then paused, and the current state is sent back to RLGym. This is
        decoded into a `GameState` object, which gets passed to the configuration objects to determine the rewards,
        next observation, and done signal.

        :param actions: A tuple containing N lists of actions, where N is the number of agents interacting with the game.
        :param reset_at_term: A boolean that tells step to reset_to_state() when 
        :return: A tuple containing (obs, rewards, done, info)
        """

        #TODO: This is a temporary solution to the action space problems in the current implementation.
        if len(actions) == 8:
            actions[5:] = [0 if x <= 0 else 1 for x in actions[5:]]
        actions_sent = self._send_actions(actions)

        received_state = self._receive_state()

        # print("recieved!!",received_state.__dict__)
        # print()
        # print("player        ",received_state.players[0].__dict__['car_data'].__dict__)
        # print("inv_player    ",received_state.players[0].__dict__['inverted_car_data'].__dict__)
        # print("ball      ",received_state.ball.__dict__)
        # print("inv_ball  ",received_state.inverted_ball.__dict__)

        #If, for any reason, the state is not successfully received, we do not want to just crash the API.
        #This will simply pretend that the state did not change and advance as though nothing went wrong.
        if received_state is None:
            state = self._prev_state
        else:
            state = received_state

        if self.reset_at_term_exer:
            done_exer, exer_state = self._match.is_done_exer(state, self.which_exer, self.exercise_reset_states) or received_state is None or not actions_sent
        
        done = self._match.is_done(state) or received_state is None or not actions_sent
        obs = self._match.build_observations(state)
        reward = self._match.get_rewards(state, done)
        self._prev_state = state

        info = {
            'state': state,
            'result': self._match.get_result(state)
        }

        if self.reset_at_term_exer:
            if done_exer: 
                obs = self.reset_to_exer_state(exer_state)
            if done or done_exer:
                # increment which_exer
                if self.which_exer < len(self.exercise_reset_states)-1: self.which_exer += 1
                else: self.which_exer = 0
            # return obs, reward, done, (done_exer,exer_state), info
        
        return obs, reward, done, info

    def close(self):
        """
        Disconnect communication with the Bakkesmod plugin and close the game. This should only be called if you are finished
        with your current RLGym environment instance.
        """
        self._comm_handler.close_pipe()
        if self._game_process is not None:
            self._game_process.terminate()

    def _receive_state(self):
        # print("Waiting for state...")
        message, exception = self._comm_handler.receive_message(header=Message.RLGYM_STATE_MESSAGE_HEADER)
        if exception is not None:
            self._attempt_recovery()
            return None

        if message is None:
            return None
        # print("GOT MESSAGE\n HEADER: '{}'\nBODY: '{}'\n".format(message.header, message.body))
        if message.body is None:
            return None

        return self._match.parse_state(message.body)

    def _send_actions(self, actions):
        action_string = self._match.format_actions(actions)
        exception = self._comm_handler.send_message(header=Message.RLGYM_AGENT_ACTION_IMMEDIATE_RESPONSE_MESSAGE_HEADER, body=action_string)
        if exception is not None:
            self._attempt_recovery()
            return False
        return True

    def _attempt_recovery(self):
        print("!ROCKET LEAGUE HAS CRASHED!\nATTEMPTING RECOVERY")
        import os
        import time
        self.close()
        proc_list = os.popen('wmic process get description, processid').read()
        num_instances = proc_list.count("RocketLeague.exe")
        wait_time = 2 * num_instances

        print("Discovered {} existing Rocket League processes. Waiting {} seconds before attempting to open "
              "a new one.".format(num_instances, wait_time))

        time.sleep(wait_time)
        self._open_game()
        self._setup_plugin_connection()

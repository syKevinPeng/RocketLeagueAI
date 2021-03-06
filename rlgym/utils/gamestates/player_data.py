"""
A class containing all data about a player in the game.
"""

from rlgym.utils.gamestates import PhysicsObject


class PlayerData(object):
    def __init__(self):
        self.car_id: int = -1
        self.team_num: int = -1
        self.match_goals: int = -1
        self.match_saves: int = -1
        self.match_shots: int = -1
        self.match_demolishes: int = -1
        self.boost_pickups: int = -1
        self.is_demoed: bool = False
        self.on_ground: bool = False
        self.ball_touched: bool = False
        self.has_flip: bool = False
        self.boost_amount: float = -1
        self.car_data: PhysicsObject = PhysicsObject()
        self.inverted_car_data: PhysicsObject = PhysicsObject()

    def set_vals_from_dict(self,player_dict):
        self.car_id =             player_dict['car_id']
        self.team_num =           player_dict['team_num']
        self.match_goals =        player_dict['match_goals']
        self.match_saves =        player_dict['match_saves']
        self.match_shots =        player_dict['match_shots']
        self.match_demolishes =   player_dict['match_demolishes']
        self.boost_pickups =      player_dict['boost_pickups']
        self.is_demoed =          player_dict['is_demoed']
        self.on_ground =          player_dict['on_ground']
        self.ball_touched =       player_dict['ball_touched']
        self.has_flip =           player_dict['has_flip']
        self.boost_amount =       player_dict['boost_amount']
        self.car_data =           player_dict['car_data']
        self.inverted_car_data =  player_dict['inverted_car_data']

    def __str__(self):
        output = "****PLAYER DATA OBJECT****\n" \
                 "Match Goals: {}\n" \
                 "Match Saves: {}\n" \
                 "Match Shots: {}\n" \
                 "Match Demolishes: {}\n" \
                 "Boost Pickups: {}\n" \
                 "Is Alive: {}\n" \
                 "On Ground: {}\n" \
                 "Ball Touched: {}\n" \
                 "Has Flip: {}\n" \
                 "Boost Amount: {}\n" \
                 "Car Data: {}\n" \
                 "Inverted Car Data: {}"\
            .format(self.match_goals,
                    self.match_saves,
                    self.match_shots,
                    self.match_demolishes,
                    self.boost_pickups,
                    self.is_demoed,
                    self.on_ground,
                    self.ball_touched,
                    self.has_flip,
                    self.boost_amount,
                    self.car_data,
                    self.inverted_car_data)
        return output
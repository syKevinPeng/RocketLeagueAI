from typing import List, Union, Tuple, Dict

rot = 65536/2 # pi=2^16?? crazy

# player faces FORWARD next to ball, with ball at spawn
# player_orientation = [0, rot/4, 0] 
# player_pos         = [0, -200, 17]
# ball_pos           = [0, 0, 98]
# ball_lin_vel       = [0, 0, 0]
# ball_ang_vel       = [0, 0, 0]
# player_lin_vel     = [0, 0, 0]
# player_ang_vel     = [0, 0, 0]

# reset_state = player_pos + player_lin_vel + player_ang_vel + player_orientation + ball_pos + ball_lin_vel + ball_ang_vel
# encoded_reset_state = " ".join([str(arg) for arg in reset_state])

# ------------------------------------------------------------------------------------------
# some example exercises ------------------------------------------------------------------- 
# ------------------------------------------------------------------------------------------
goalpost_ball_car_vals = [([900, 5000, 98],[1300,4900,17],[0,rot,0]),
                            ([900, 5000, 98],[1150,4700,17],[0,rot*(3/4),0]),
                            ([900, 5000, 98],[900 ,4600,17],[0,rot*(1/2),0])]

midgoal_ball_car_vals =  [([0, 5000, 98],[400,4900,17],[0,rot,0]),
                            ([0, 5000, 98],[250,4700,17],[0,rot*(3/4),0]),
                            ([0, 5000, 98],[0   ,4600,17],[0,rot*(1/2),0])]

slightlyoffset_ball_car_vals =  [ ( [bp[0], bp[1]-400, bp[2]],
                                    [cp[0], cp[1]-900, cp[2]],
                                    co) for bp,cp,co in midgoal_ball_car_vals]

farpostbounce_ball_car_vals =  [ ([500,  3500, 98+1200],[1220,2500,17],[0,rot*(0.70),0]),
                                ([1050, 4000, 98+1200],[2120,3250,17],[0,rot*(0.81),0]),
                                ([1450, 4600, 98+1200],[2650,4200,17],[0,rot*(0.91),0])]
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def flip_vals_across_midline_left2right(ball_pos,car_pos,car_orient):
    ball_pos[0]  *= -1
    car_pos[0]   *= -1
    car_orient[2] = rot/2 - car_orient[2]

    return ball_pos, car_pos, car_orient

def ball_car_vals_to_state(ball_car_set, ball_lin_vel=None, 
                                        ball_ang_vel=None, 
                                        player_lin_vel=None, 
                                        player_ang_vel=None):

    ball_lin_vel   = ball_lin_vel   if ball_lin_vel   else [0, 0, 0]
    ball_ang_vel   = ball_ang_vel   if ball_ang_vel   else [0, 0, 0]
    player_lin_vel = player_lin_vel if player_lin_vel else [0, 0, 0]
    player_ang_vel = player_ang_vel if player_ang_vel else [0, 0, 0]
    ball_pos           = ball_car_set[0]
    player_pos         = ball_car_set[1]
    player_orientation = ball_car_set[2]

    reset_state = player_pos + player_lin_vel + player_ang_vel + player_orientation + \
                ball_pos   + ball_lin_vel   + ball_ang_vel
    encoded_reset_state = " ".join([str(arg) for arg in reset_state])

    return encoded_reset_state

def some_examples(num_examples):
    examples = [goalpost_ball_car_vals, 
                midgoal_ball_car_vals, 
                slightlyoffset_ball_car_vals, 
                farpostbounce_ball_car_vals]

    return examples[0:num_examples] # filter, then sum(examples,[]) to flatten

def convert_exercises(exercises_list: List[Tuple[List]]):
    # ([ball_pos],[car_pos],[car_orient])
    exercise_reset_states = [ball_car_vals_to_state(x) for x in exercises_list] 

    return exercise_reset_states

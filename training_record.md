1. First time
- tensorboard: https://tensorboard.dev/experiment/PfRR3OVqRr6vYynwW3JWxg/#scalars
- problem: reward function is not ideal
- car not moving 
- reward_function = CombinedReward((TouchBallReward(), GoalReward(per_goal=5.0), MoveTowardsBallReward(), TimeReward()), (1, 1.0, 0.1, 0.1))
2. second time
- modification: add linear distance reward function
- car not moving
- problem reward for linear distance is too high

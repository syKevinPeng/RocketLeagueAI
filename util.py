from stable_baselines3.common.callbacks import BaseCallback


class tensorboard_for_rewards(BaseCallback):
    """
       Custom callback for plotting additional values in tensorboard.
       """

    def __init__(self, verbose=0):
        super(tensorboard_for_rewards, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('Goal Reward', value)
        return True

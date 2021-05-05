import rlgym
import time

def example():
    env = rlgym.make("Duel")
    # agent = RandomAgent()
    # print(env.action_space.shape, env.observation_space.shape)
    print("mark3")
    while True:
        obs = env.reset()
        done = False
        steps = 0
        ep_reward = 0
        t0 = time.time()
        while not done:
            actions = env.action_space.sample()  # agent.act(obs) | Your agent should go here
            new_obs, reward, done, state = env.step(actions)
            ep_reward += reward
            obs = new_obs
            steps += 1

        length = time.time() - t0
        print("Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(length / steps, length,ep_reward))

def self_example():
    env = rlgym.make("DuelSelf")
    print("Here1")
    while True:
        print("Here2")
        obs = env.reset()
        obs_1 = obs[0]
        obs_2 = obs[1]
        done = False
        steps = 0
        ep_reward = 0
        t0 = time.time()
        while not done:
            actions_1 = env.action_space.sample()
            actions_2 = env.action_space.sample()
            actions = [actions_1, actions_2]
            new_obs, reward, done, state = env.step(actions)
            ep_reward += reward[0]
            obs_1 = new_obs[0]
            obs_2 = new_obs[1]
            steps += 1

        length = time.time() - t0
    print("Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(length / steps, length, ep_reward))

if __name__ == "__main__":
    example()

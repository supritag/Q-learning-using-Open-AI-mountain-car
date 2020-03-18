import numpy as np
import gym
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    n_games = 50000
    rewards = np.zeros(n_games)
    for i in range(n_games):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            #env.render()
        rewards[i] = episode_reward

    plt.plot(rewards)
    plt.xlabel('Episode number')
    plt.ylabel('Score')
    plt.savefig('mountaincar_baseline.png')

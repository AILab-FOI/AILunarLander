import requests
import json

import numpy as np
import gym


if __name__ == "__main__":

    # Initializes the environment
    env = gym.make('LunarLander-v2')  # uzimanje enviromenta
    URL = 'http://127.0.0.1:8000/action'

    # Defines training related constants
    N_EPISODES = 5
    num_episode_steps = env.spec.max_episode_steps  # constant value
    action_size = env.action_space.n
    state_space = env.observation_space.shape[0]

    for episode in range(N_EPISODES):
        # Defines the total reward per episode
        total_reward = 0

        # Resets the environment
        observation = env.reset()

        # Gets the state
        state = np.reshape(observation, [1, state_space])

        for episode_step in range(num_episode_steps):
            # Renders the screen after new environment observation
            env.render(mode="human")

            # Gets a new action
            data = state.tolist()

            action = requests.post(URL, json=data)
            print(action)

            # Takes action and calculates the total reward
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            # Updates the state
            state = np.reshape(observation, [1, state_space])

            if done:
                print("Episode %d/%d finished after %d steps with total reward of %f."
                      % (episode + 1, N_EPISODES, episode_step + 1, total_reward))
                break

            elif episode_step >= num_episode_steps - 1:
                print("Episode %d/%d timed out at %d with total reward of %f."
                      % (episode + 1, N_EPISODES, episode_step + 1, total_reward))


import requests

import gym


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    URL = 'http://localhost:8007/action'

    N_EPISODES = 5
    num_episode_steps = env.spec.max_episode_steps  # constant value
    action_size = env.action_space.n
    state_space = env.observation_space.shape[0]

    for episode in range(N_EPISODES):
        total_reward = 0

        observation = env.reset()

        for episode_step in range(num_episode_steps):
            env.render(mode="human")

            data = {'variables': observation.tolist()}

            response = requests.post(URL, json=data)
            action = int(response.content)

            observation, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                print("Episode %d/%d finished after %d steps with total reward of %f."
                      % (episode + 1, N_EPISODES, episode_step + 1, total_reward))
                break

            elif episode_step >= num_episode_steps - 1:
                print("Episode %d/%d timed out at %d with total reward of %f."
                      % (episode + 1, N_EPISODES, episode_step + 1, total_reward))

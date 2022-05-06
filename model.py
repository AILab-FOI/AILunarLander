import pickle
import os
import random
from collections import deque

import gym
import numpy as np
from sqlalchemy import PickleType
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class Agent:
    def __init__(self, state_space, action_size, batch_size=64, memory_size=1000000):
        self.state_space = state_space
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.98  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.98
        self.learning_rate = 1e-3
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_space, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state)[0])

    def experience_replay(self):
        if self.batch_size > len(self.memory):
            return

        # Randomly sample a batch from the memory
        random_batch = random.sample(self.memory, self.batch_size)

        state = np.zeros((self.batch_size, self.state_space))
        next_state = np.zeros((self.batch_size, self.state_space))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = random_batch[i][0]
            action.append(random_batch[i][1])
            reward.append(random_batch[i][2])
            next_state[i] = random_batch[i][3]
            done.append(random_batch[i][4])

        # Batch prediction
        target = self.model.predict(state)
        target_next = self.model(next_state)

        for i in range(len(random_batch)):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(
            np.array(state),
            np.array(target),
            batch_size=self.batch_size,
            verbose=0
        )


    def load_weights(self, weights_file):
        self.epsilon = self.epsilon_min
        self.model.load_weights(weights_file, by_name=True)

    def save_weights(self, weights_file):
        self.model.save_weights(weights_file)

    def save_model(self, model_file):
        self.model.save(model_file)


if __name__ == "__main__":

    # Initializes the environment
    env = gym.make('LunarLander-v2') #uzimanje enviromenta

    # Defines training related constants
    n_episodes = 2
    num_episode_steps = env.spec.max_episode_steps  # constant value
    action_size = env.action_space.n
    state_space = env.observation_space.shape[0]
    max_reward = 0

    # Creates an agent
    agent = Agent(state_space=state_space, action_size=action_size)

    # Loads the weights
    #if os.path.isfile("C:/Users/lukav/OneDrive/Documents/Završni rad/LunarLanderGH/AILunarLander/lunar_lander.h5"):
    agent.load_weights("lunar_lander_weights.h5")

    for episode in range(n_episodes):
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
            action = agent.act(state)

            # Takes action and calculates the total reward
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            # Gets the next state
            next_state = np.reshape(observation, [1, state_space])

            # Memorizes the experience
            agent.memorize(state, action, reward, next_state, done)

            # Updates the network weights
            agent.experience_replay()

            # Updates the state
            state = next_state

            if done:
                print("Episode %d/%d finished after %d steps with total reward of %f."
                      % (episode + 1, n_episodes, episode_step + 1, total_reward))
                break

            elif episode_step >= num_episode_steps - 1:
                print("Episode %d/%d timed out at %d with total reward of %f."
                      % (episode + 1, n_episodes, episode_step + 1, total_reward))

        # Updates the epsilon value
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        #Saves the network weights
        if total_reward >= max_reward:
            #agent.save_weights("lunar_lander_weights.h5")
            max_reward = total_reward

    #save model
    agent.save_model('model.h5')

    #save model in pickle file
    #with open('model_pickle', 'wb') as f:
    #    pickle.dump(agent, f)

    # Closes the environment
    env.close()


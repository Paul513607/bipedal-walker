import gc

import tensorflow
from keras import Model, Input
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random

MAX_MEMORY = 1000000
BATCH_SIZE = 32


class DeepQNetwork:
    def __init__(self, env, epsion=0.7, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, alpha=0.01,
                 gamma=0.99):
        self.env = env
        self.observation_len = env.observation_space.shape[0]
        self.actions_len = env.action_space.shape[0]

        # Hyperparameters
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsion
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = self.create_model()
        self.curr_rewards = []

    def create_model(self):
        observation_input = Input(shape = self.env.observation_space.shape)
        hidden1 = Dense(512, activation='relu')(observation_input)
        hidden2 = Dense(512, activation='relu')(hidden1)
        mean = Dense(self.env.action_space.shape[0], activation='tanh')(hidden2)

        model = Model(inputs=observation_input, outputs=mean)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def get_action(self, observation):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        p = self.model.predict(observation, verbose=0)
        return p[0]

    def save_to_memory(self, transition):
        self.memory.append(transition)
        self.curr_rewards.append(transition[2])

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for observation, action, reward, new_observation, done in batch:
            target = reward if done else (1 - self.alpha) * reward + \
                                         self.alpha * (self.gamma * np.amax(
                self.model.predict(new_observation, verbose=0)[0]))

            old_target = self.model.predict(observation, verbose=0)
            old_target[0] = target
            self.model.fit(x=observation, y=old_target, epochs=1, verbose=0)
            self.curr_rewards = []

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def on_episode_end_clean(self):
        tensorflow.keras.backend.clear_session()
        gc.collect()

    def save_model(self, name):
        self.model.save(name)

    def load_model(self, name):
        self.model = load_model(name)

    def load_weights(self, name):
        self.model.load_weights(name)

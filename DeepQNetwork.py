import random
from collections import deque

import keras
import numpy as np
from keras import Sequential
from keras.layers import Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

GAMMA = 0.95
ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 16
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
MIN_REPLAY_SIZE = 100
EPISODE_MODEL_SAVE = 40

class DeepQNetwork:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.losses = []
        if os.path.exists(f"models/model_{EPISODE_MODEL_SAVE}.h5"):
            self.model = keras.models.load_model(f"models/model_{EPISODE_MODEL_SAVE}.h5")
            self.target_model = keras.models.load_model(f"models/model_{EPISODE_MODEL_SAVE}.h5")
            self.load_model(f"models/model_{EPISODE_MODEL_SAVE}.h5")
        else:
            self.model = self.create_model()
            self.target_model = self.create_model()
            self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer='rmsprop', metrics=['accuracy'])
        return model

    def update_replay_memory(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape), verbose=0)[0]

    def get_action(self, state, action_space):
        if np.random.random() < EPSILON:
            return action_space.sample()
        else:
            return self.get_qs(state)

    def train(self, terminal_state, step):
        if len(self.memory) < MIN_REPLAY_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states, verbose=0)

        next_states = np.array([transition[3] for transition in minibatch])
        next_qs_list = self.target_model.predict(next_states, verbose=0)

        x = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_next_q = reward + GAMMA * np.max(next_qs_list[index])
            else:
                max_next_q = reward

            current_qs = current_qs_list[index]
            action_idx = np.argmax(action)
            current_qs[action_idx] = max_next_q

            x.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(x), np.array(y), batch_size=BATCH_SIZE, verbose=0, shuffle=False, callbacks=None)

        self.target_update_counter += 1

        if self.target_update_counter > TARGET_UPDATE:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def update_epsilon(self):
        global EPSILON
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
        
    def get_epsilon(self):
        return EPSILON

    def save_model(self, name):
        self.model.save(name)

    def load_model(self, name):
        self.model = keras.models.load_model(name)
        self.target_model = keras.models.load_model(name)

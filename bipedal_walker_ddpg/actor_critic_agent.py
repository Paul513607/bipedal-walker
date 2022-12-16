import keras.models
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.layers.merging import Add, Concatenate
import keras.backend as K
import tensorflow as tf
import os

import random
from replay_memory import ReplayMemory


class ActorCriticAgent:
    def __init__(self, env, actor_h1_size=512, actor_h2_size=512, actor_h3_size=512, critic_h1_size=512, critic_h2_size=512,
                 critic_h3_size=512, lr_actor=0.001, lr_critic=0.002, discount=0.99, tau=0.005, noise=0.1,
                 checkpoint_directory='temp/tensorflow_ddpg', max_memory_size=100000, batch_size=64):
        self.env = env
        self.discount = discount
        self.tau = tau
        self.noise = noise
        self.replay_buffer = ReplayMemory(max_memory_size)
        self.batch_size = batch_size
        self.checkpoint_directory = checkpoint_directory
        self.actor = self.make_actor_model(actor_h1_size, actor_h2_size, actor_h3_size, lr_actor, 'actor')
        self.target_actor = self.make_actor_model(actor_h1_size, actor_h2_size, actor_h3_size, lr_actor, 'target_actor')

        self.critic = self.make_critic_model(
            critic_h1_size, critic_h2_size, critic_h3_size, lr_critic, 'critic')
        self.target_critic = self.make_critic_model(
            critic_h1_size, critic_h2_size, critic_h3_size, lr_critic, 'target_critic')

        self.update_targets(tau=1)

    def make_actor_model(self, h1_size, h2_size, h3_size, learning_rate, name):
        state_input = Input(shape=self.env.observation_space.shape)
        hidden1 = Dense(h1_size, activation='relu')(state_input)
        hidden2 = Dense(h2_size, activation='relu')(hidden1)
        # hidden3 = Dense(h2_size, activation='relu')(hidden2)
        mean = Dense(self.env.action_space.shape[0], activation='tanh')(hidden2)

        model = Model(inputs=state_input, outputs=mean, name=name)
        model.compile(optimizer=Adam(learning_rate=learning_rate))

        return model

    def make_critic_model(self, h1_size, h2_size, h3_size, learning_rate, name):
        state_input = Input(shape=self.env.observation_space.shape)
        action_input = Input(shape=self.env.action_space.shape)
        state_hidden1 = Dense(h1_size, activation='relu')(state_input)
        action_hidden1 = Dense(h1_size)(action_input)

        # state_hidden2 = Dense(h2_size, activation='relu')(state_hidden1)

        # action_hidden1 = Dense(h2_size)(action_input)

        merged_layers = Add()([state_hidden1, action_hidden1])
        merged_hidden1 = Dense(h2_size, activation='relu')(merged_layers)
        value = Dense(1, activation='linear')(merged_hidden1)

        model = Model(inputs=[state_input, action_input], outputs=value, name=name)
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

        return model

    def add_to_buffer(self, state, action, reward, new_state, done):
        self.replay_buffer.append((state, action, reward, new_state, done))

    def choose_action(self, env_state, testing=False):
        state = tf.convert_to_tensor([env_state], dtype=tf.float32)
        actions = self.actor(state)

        if not testing:
            actions += tf.random.normal(shape=[self.env.action_space.shape[0]], mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, self.env.action_space.low[0], self.env.action_space.high[0])

        # extracting the numpy array from the tensor
        return actions[0]

    def update_targets(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + self.target_actor.weights[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + self.target_critic.weights[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def save_model(self):
        self.actor.save(os.path.join(self.checkpoint_directory, self.actor.name + '.h5'))
        self.critic.save(os.path.join(self.checkpoint_directory, self.critic.name + '.h5'))
        self.target_actor.save(os.path.join(self.checkpoint_directory, self.target_actor.name + '.h5'))
        self.target_critic.save(os.path.join(self.checkpoint_directory, self.target_critic.name + '.h5'))

    def load_model(self):
        self.actor = keras.models.load_model(os.path.join(self.checkpoint_directory, self.actor.name + '.h5'))
        self.critic = keras.models.load_model(os.path.join(self.checkpoint_directory, self.critic.name + '.h5'))
        self.target_actor = keras.models.load_model(os.path.join(self.checkpoint_directory, self.target_actor.name + '.h5'))
        self.target_critic = keras.models.load_model(os.path.join(self.checkpoint_directory, self.target_critic.name + '.h5'))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        sample = self.replay_buffer.sample(self.batch_size)
        state, action, reward, new_state, done = zip(*sample)
        done = np.array(done)
        # convert to tensor so we can do tensor operations
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)

        # separation of gradient calculation from choose action
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            critic_value_ = tf.squeeze(self.target_critic(
                [new_states, target_actions]), 1)
            critic_value = tf.squeeze(self.critic([states, actions]), 1)
            target = rewards + self.discount * critic_value_ * (1 - done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)
        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            # practic delta
            actor_loss = -self.critic([states, new_policy_actions])
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_targets()

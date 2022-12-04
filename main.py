import random

import keras
import gym
import time
import numpy as np
import time

from DeepQNetwork import DeepQNetwork

EPIOSDES = 3000
STEPS = 1000
RENDER = False

if __name__ == '__main__':
    env = gym.make("BipedalWalker-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = DeepQNetwork(state_size, action_size)
    for episode in range(EPIOSDES):
        state = env.reset()
        score = 0
        for step in range(STEPS):
            if RENDER:
                env.render()
            action = agent.get_action(state, env.action_space)
            new_state, reward, done, _ = env.step(action)
            score += reward
            agent.update_replay_memory(state, action, reward, new_state, done)
            agent.train(done, step)
            state = new_state
            if step % 10 == 0:
                pass
            if done:
                break
        print(f"Episode: {episode}, Score: {score}, Epsilon: {agent.get_epsilon()}")
        agent.update_epsilon()
        if episode == 2000:
            RENDER = True
        if (episode + 1) % 200 == 0:
            agent.save_model(f"models/model_{episode + 1}.h5")


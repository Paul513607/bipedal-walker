import keras
import gym
import time
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make("BipedalWalker-v3")
    env.reset()

    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
    env.close()

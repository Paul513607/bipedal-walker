import numpy as np
import gym
import gym_quickcheck

from AugmentedRandomSearch import AugmentedRandomSearch
from Policy import Policy
from Normalizer import Normalizer

EPISODE_START = 0
EPISODES = 2000
MAX_STEPS = 4000
ENVIRONMENT = 'gym_quickcheck:BipedalWalker-v3'
RENDER = True
# RENDER = True
MODEL_FILE_NAME = 'bipedal-walker-ars.npy'
FAIL_SCORE = -300
TARGET_SCORE = 300
GAMMA = 0.9

if __name__ == '__main__':
    env = gym.make(ENVIRONMENT)
    env = env.unwrapped
    env = gym.wrappers.RecordVideo(env, 'videos', episode_trigger=lambda x: x % 200 == 0)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    agent = AugmentedRandomSearch(num_inputs, num_outputs, 0.02, 16, 16, 0.03)
    try:
        agent.load(MODEL_FILE_NAME)
        print("Loaded saved model")
    except Exception as e:
        print(e)
        print("No saved model found")

    agent.train(env, GAMMA, MAX_STEPS, EPISODES)

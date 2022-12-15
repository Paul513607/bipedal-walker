import gym
import numpy as np
import time

from DeepQNetwork import DeepQNetwork

ENV_NAME = 'gym_quickcheck:BipedalWalker-v3'
EPISODE_START = 0
EPISODES = 10000
RENDER = True
# RENDER = True
MODEL_FILE_NAME = 'bipedal-walker-model1.h5'
MODEL_FILE_NAME2 = 'bipedal-walker-model-perm1.h5'
FAIL_SCORE = -300
TIMEOUT = 20
SCORE_RENDER = 270
SCORE_TARGET = 300
MIN_EPISODE_RENDER = 2000

if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    agent = DeepQNetwork(env, epsion=0.01)
    try:
        agent.load_model(MODEL_FILE_NAME)
        # agent.load_weights(MODEL_FILE_NAME2)
        print("Loaded saved model")
    except Exception as e:
        print(e)
        print("No saved model found")

    wins = 0
    all_scores = []

    for ep in range(EPISODE_START, EPISODES):
        observation = env.reset()
        observation = observation.reshape(1, -1)
        start = time.time()
        while True:
            if RENDER:
                env.render()

            action = agent.get_action(observation)
            new_observation, reward, done, inf = env.step(action)
            new_observation = new_observation.reshape(1, -1)
            agent.save_to_memory((observation, action, reward, new_observation, done))
            observation = new_observation
            score = sum(agent.curr_rewards)

            end = time.time()
            duration = end - start
            if duration > TIMEOUT or score < FAIL_SCORE:
                done = True

            if done:
                all_scores.append(score)
                max_reward = np.max(all_scores)
                episode_of_max_reward = np.argmax(all_scores)
                if score >= SCORE_TARGET:
                    wins += 1
                print('Episode: {}, Score: {}, Epsilon: {}, Max score: {}, Episode max: {}, Wins: {}'
                      .format(ep, score, agent.epsilon, max_reward, episode_of_max_reward, wins))

                agent.train()

                if int(score) > SCORE_RENDER and ep > MIN_EPISODE_RENDER:
                    render = True
                break
        agent.on_episode_end_clean()
        if (ep + 1) % 10 == 0:
            agent.save_model('bipedal-walker-model2.h5')

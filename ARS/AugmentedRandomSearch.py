import numpy as np

from ARS.Normalizer import Normalizer
from ARS.Policy import Policy


class AugmentedRandomSearch:
    num_inputs: int
    num_outputs: int
    learning_rate: float
    num_deltas: int
    num_best_deltas: int
    noise: float

    def __init__(self, num_inputs, num_outputs, learning_rate, num_deltas, num_best_deltas, noise):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.num_deltas = num_deltas
        self.num_best_deltas = num_best_deltas
        self.noise = noise
        self.normalizer = Normalizer(num_inputs)
        self.policy = Policy(num_inputs, num_outputs, learning_rate, num_deltas, num_best_deltas, noise)

    def explore(self, env, max_episodes, direction=None, delta=None, render=False):
        state = env.reset()
        done = False
        num_plays = 0.
        sum_rewards = 0
        while not done and num_plays < max_episodes:
            if render:
                env.render()
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)
            action = self.policy.evaluate(state, delta, direction)
            state, reward, done, _ = env.step(action)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            num_plays += 1
        return sum_rewards

    def train(self, env, gamma, max_steps, max_episodes, print_every=10):
        win = 0
        for step in range(0, max_steps):
            deltas = self.policy.sample_deltas()
            positive_rewards = np.zeros(self.num_deltas)
            negative_rewards = np.zeros(self.num_deltas)

            for k in range(self.num_deltas):
                positive_rewards[k] = self.explore(env, max_episodes, direction="+", delta=deltas[k])
                negative_rewards[k] = self.explore(env, max_episodes, direction="-", delta=deltas[k])

            all_rewards = np.array(positive_rewards + negative_rewards)
            sigma_r = all_rewards.std()

            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.num_best_deltas]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

            self.policy.update(rollouts, sigma_r)

            reward_evaluation = self.explore(env, max_episodes, render=True)
            if reward_evaluation > 300:
                win += 1
            if step % print_every == 0:
                print('Step:', step, 'Reward:', reward_evaluation, 'Win:', win)
                self.save("bipedal-walker-ars.npy")

    def play(self, env, render=True):
        state = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            state = self.normalizer.normalize(state)
            action = self.policy.evaluate(state)
            state, reward, done, _ = env.step(action)

    def save(self, filename):
        self.policy.save(filename)

    def load(self, filename):
        self.policy.load(filename)


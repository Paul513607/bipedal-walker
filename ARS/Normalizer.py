import numpy as np


class Normalizer:
    n: np.ndarray
    mean: np.ndarray
    mean_diff: np.ndarray
    variance: np.ndarray

    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.variance = np.zeros(num_inputs)

    def observe(self, observation):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (observation - self.mean) / self.n
        self.mean_diff += (observation - last_mean) * (observation - self.mean)
        self.variance = self.mean_diff / self.n
        self.variance[self.variance < 1e-2] = 1e-2

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.variance)
        return (inputs - obs_mean) / obs_std

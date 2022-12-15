import numpy as np


class Policy:
    theta: np.ndarray
    learning_rate: float
    num_deltas: int
    num_best_deltas: int
    noise: float

    def __init__(self, num_inputs, num_outputs, learning_rate, num_deltas, num_best_deltas, noise):
        self.theta = np.zeros((num_outputs, num_inputs))
        self.learning_rate = learning_rate
        self.num_deltas = num_deltas
        self.num_best_deltas = num_best_deltas
        self.noise = noise

    def evaluate(self, input_data, delta=None, direction=None):
        if direction is None:
            return self.theta.dot(input_data)
        elif direction == "+":
            return (self.theta + self.noise * delta).dot(input_data)
        elif direction == "-":
            return (self.theta - self.noise * delta).dot(input_data)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.num_deltas)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        self.theta += self.learning_rate / (self.num_best_deltas * sigma_r) * step

    def save(self, filename):
        np.save(filename, self.theta)

    def load(self, filename):
        self.theta = np.load(filename)

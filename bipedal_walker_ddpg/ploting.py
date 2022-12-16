import numpy as np
import matplotlib.pyplot as plt


def plot_agent_learning(score_history, plot_path):
    indexes = list(range(1, len(score_history) + 1))
    running_avg = np.zeros(len(score_history))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(score_history[max(0, i - 100):(i + 1)])
    plt.plot(indexes, running_avg)
    plt.plot(indexes, score_history)
    plt.title('Running average for previous 100 scores')
    plt.savefig(plot_path)

from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


class BanditTask:
    def __init__(self, mean_0: float, mean_1: float, std: float, n_trials: int):
        self.mean_0 = mean_0
        self.mean_1 = mean_1
        self.std = std
        self.n_trials = n_trials
        self.means, self.rewards = self.generate_trajectories()

    def generate_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        # Your code here
        means = np.repeat(np.array([[self.mean_0, self.mean_1]]), self.n_trials, axis=0)
        rewards = np.clip(np.random.normal(means, self.std), 0, 1)
        return means, rewards

    def plot(self):
        for i_arm in range(2):
            mean_line = plt.plot(self.means[:, i_arm], label=f'Mean arm {i_arm}')
            reward_line = plt.plot(self.rewards[:, i_arm], label=f'Reward arm {i_arm}', linestyle='--', color=mean_line[0].get_color())
        plt.legend()
        plt.xlabel('Trial')
        plt.ylabel('Reward')
        plt.title(self)

    def reset(self):
        self.means, self.rewards = self.generate_trajectories()
        return self

    def __iter__(self):
        for rewards in self.rewards:
            yield rewards

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std:.2f})"


class ReversalBanditTask(BanditTask):
    def generate_trajectories(self):
        reversal_point = self.n_trials // 2
        means = np.repeat(np.array([[self.mean_0, self.mean_1]]), self.n_trials, axis=0)
        means[reversal_point:] = np.repeat(np.array([[self.mean_1, self.mean_0]]), self.n_trials - reversal_point, axis=0)
        rewards = np.clip(np.random.normal(means, self.std, (self.n_trials, 2)), 0, 1)
        return means, rewards

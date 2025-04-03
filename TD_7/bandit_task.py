from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt


class BanditTask:
    def __init__(self, init_mean_0: float, mean_1: float, std: float, n_trials: int):
        self.mean_0 = init_mean_0
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


class RestlessBanditTask(BanditTask):
    def __init__(self, init_mean_0: float, mean_1: float, std: float, drift: float, n_trials: int):
        self.init_0 = init_mean_0
        self.init_1 = mean_1
        self.std = std
        self.drift = drift
        self.n_trials = n_trials
        self.means, self.rewards = self.generate_trajectories()

    def generate_trajectories(self):
        means = np.zeros((self.n_trials, 2))
        rewards = np.zeros((self.n_trials, 2))

        # Generating drifting mean
        means[0] = np.array([self.init_0, self.init_1])
        for i_trial in range(1, self.n_trials):
            means[i_trial] =np.clip(means[i_trial - 1] + np.random.normal(0, self.drift, 2), 0, 1)
        # Adding noise to the rewards
        for i_trial in range(self.n_trials):
            rewards[i_trial] = np.clip(means[i_trial] + np.random.normal(0, self.std, 2), 0, 1)
        return means, rewards

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std:.2f}, drift={self.drift:.2f})"

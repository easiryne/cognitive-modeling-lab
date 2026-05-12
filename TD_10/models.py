from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy
from matplotlib import pyplot as plt
from pybads import OptimizeResult, BADS


@dataclass
class Parameter:
    name: str
    initial_value: float
    distribution: scipy.stats.rv_continuous
    bounds: Tuple[float, float]
    plausible_bounds: Tuple[float, float]

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

class Model(ABC):
    evidence = None

    def simulate(self, task, plot=False):
        self.reset(task.n_blocks)
        evidences = np.zeros((task.n_blocks, task.n_trials+1))
        evidences[:, 0] = sigmoid(self.evidence.squeeze())

        # Loop through the stimuli (the blocks are batched)
        for j, (stimulus, decide) in enumerate(task):
            self.update(stimulus)
            evidences[:, j+1] = sigmoid(self.evidence.squeeze())

            # On the last stimulus, the model chooses a category
            if decide:
                prob = self.policy()
                choice = np.random.binomial(1, prob)

        if plot:
            fig, axes = task.plot(show=False)
            for i in range(task.n_blocks):
                ax = axes[i]
                ax.plot(evidences[i, :], linestyle='-', color='k')
                correct = task.ground_truth[i][0] == choice[i][0]
                choice_circle = plt.Circle((task.n_trials + 3, 0.5), radius=0.3, linewidth=3,
                                           edgecolor='green' if correct else 'red',
                                           facecolor='orange' if choice[i][0] == 1 else 'blue')
                ax.add_artist(choice_circle)
                ax.set_xlim(0, task.n_trials + 4)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['B', 'O'], fontsize=10)
                for label, color in zip(ax.get_yticklabels(), ['blue', 'orange']):
                    label.set_color(color)
                ax.axis('on')
                ax.set_ylabel('evidence', fontsize=6)
            axes[0].text(task.n_trials + 2.5, 1, "choice", fontsize=8, ha='left', va='bottom')
            fig.suptitle(self)
            plt.tight_layout()
            plt.show()

        return choice

    def log_likelihood(self, task,  actions: np.ndarray[int]) -> float:
        self.reset(task.n_blocks)
        for stimulus, decide in task:
            self.update(stimulus)
            if decide:
                prob_orange = self.policy()
        prob_given_choice = prob_orange * actions + (1 - prob_orange) * (1 - actions)
        prob_given_choice = 1e-6 + (1 - 1e-6 * 2) * prob_given_choice  # Little trick to avoid numerical errors
        log_likelihood = np.sum(np.log(prob_given_choice))
        return log_likelihood

    @property
    @abstractmethod
    def parameters(self) -> List[Parameter]:
        raise NotImplementedError()

    @classmethod
    def fit(cls, task, choices: np.ndarray[int], verbose: bool = False):
        def cost_function(params):
            model = cls(*params)
            log_likelihood = model.log_likelihood(task, choices)
            return -log_likelihood

        optimizer = BADS(
            fun=cost_function,
            x0=[param.initial_value for param in cls.parameters],
            lower_bounds=[param.bounds[0] for param in cls.parameters],
            upper_bounds=[param.bounds[1] for param in cls.parameters],
            plausible_lower_bounds=[param.plausible_bounds[0] for param in cls.parameters],
            plausible_upper_bounds=[param.plausible_bounds[1] for param in cls.parameters],
            options={
                'display': 'iter' if verbose else 'off',
            }
        )
        result: OptimizeResult = optimizer.optimize()
        optimal_params: np.ndarray = result.x
        return cls(*optimal_params)

    @abstractmethod
    def update(self, stimulus):
        raise NotImplementedError

    @abstractmethod
    def policy(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self, n_blocks: int):
        raise NotImplementedError

    def plot_param_specs(self):
        fig, axs = plt.subplots(1, len(self.parameters), figsize=(3 * len(self.parameters), 2))
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        for i, parameter in enumerate(self.parameters):
            x = np.linspace(parameter.bounds[0], parameter.bounds[1], 100)
            y = parameter.distribution.pdf(x)
            axs[i].plot(x, y, label='prior')
            axs[i].set_title(parameter.name)
            axs[i].axvline(parameter.initial_value, color='red', linestyle='--', label='initial value')
            axs[i].axvline(parameter.plausible_bounds[0], color='green', linestyle='--', label='plausible bounds')
            axs[i].axvline(parameter.plausible_bounds[1], color='green', linestyle='--')
            axs[i].set_xlabel('value')
            axs[i].set_ylabel('density')
            axs[i].legend()
        plt.show()

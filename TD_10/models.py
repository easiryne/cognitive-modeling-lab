from dataclasses import dataclass
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import scipy
from matplotlib import pyplot as plt
from pybads import OptimizeResult, BADS

from TD_10.sequential_perception_task import SequentialPerceptionTask
from sequential_perception_task import dist_orange, dist_blue


@dataclass
class Parameter:
    name: str
    initial_value: float
    distribution: scipy.stats.rv_continuous
    bounds: Tuple[float, float]
    plausible_bounds: Tuple[float, float]


class Model(ABC):
    evidence = None

    def simulate(self, task, plot=False):
        self.reset(task.n_blocks)
        probs = np.zeros((task.n_blocks, task.n_trials+1))
        probs[:, 0] = self.policy().squeeze()

        # Loop through the stimuli (the blocks are batched)
        for j, (stimulus, decide) in enumerate(task):
            self.update(stimulus)
            probs[:, j+1] = self.policy().squeeze()

            # On the last stimulus, the model chooses a category
            if decide:
                prob = self.policy()
                choice = np.random.binomial(1, prob)

        if plot:
            fig, axes = task.plot(show=False)
            for i in range(task.n_blocks):
                ax = axes[i]
                ax.plot(probs[i, :], linestyle='-', color='k')
                correct = task.ground_truth[i][0] == choice[i][0]
                choice_circle = plt.Circle((task.n_trials + 3, 0.5), radius=0.3, linewidth=3,
                                           edgecolor='green' if correct else 'red',
                                           facecolor='orange' if choice[i][0] == 1 else 'blue')
                ax.add_artist(choice_circle)
                ax.set_xlim(0, task.n_trials + 4)
                ax.axis('on')
                ax.set_ylabel('P(orange)', fontsize=8)
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


class ExtremumDetectionModel(Model):
    parameters = [
        Parameter(
            name='epsilon',
            initial_value=0.1,
            distribution=scipy.stats.uniform(0, 1),
            bounds=(0, 1),
            plausible_bounds=(0.1, 0.4)
        )
    ]
    def __init__(self, epsilon: float):
        self.epsilon: float = epsilon
        self.n_blocks: Optional[int] = None
        self.best_evidence: Optional[np.ndarray[float]] = None

    def update(self, stimulus):
        update_condition = np.absolute(stimulus) > np.absolute(self.best_evidence)
        self.best_evidence = np.where(update_condition, stimulus, self.best_evidence)

    def policy(self):
        greedy_decision = (self.best_evidence > 0).astype(int)
        p_orange = greedy_decision * (1 - self.epsilon) + (1 - greedy_decision) * self.epsilon
        return p_orange

    def reset(self, n_blocks: int):
        self.n_blocks = n_blocks
        self.best_evidence = np.zeros((n_blocks, 1), dtype=float)

    def __repr__(self):
        return f'ExtremumDetectionModel(epsilon={self.epsilon: .2f})'


class IntegrationModel(Model):
    parameters = [
        Parameter(
            name='leak',
            initial_value=0.1,
            distribution=scipy.stats.uniform(0, 1),
            bounds=(0, 1),
            plausible_bounds=(0.05, 0.8)
        ),
        Parameter(
            name='temperature',
            initial_value=0.1,
            distribution=scipy.stats.expon(scale=0.1),
            bounds=(1e-6, 1),
            plausible_bounds=(0.01, 0.5)
        )
    ]

    def __init__(self, leak: float, temperature: float):
        self.leak: float = leak
        self.temperature: float = temperature
        self.n_blocks: Optional[int] = None
        self.evidence: Optional[np.ndarray[float]] = None

    def update(self, stimulus):
        # Update the evidence with the leak
        p_orange = dist_orange.pdf(stimulus)
        p_blue = dist_blue.pdf(stimulus)
        log_odds_ratio = np.log(p_orange) - np.log(p_blue) # Todo: on appelle bien ca l'evidence?
        self.evidence = self.evidence * (1 - self.leak) + log_odds_ratio

    def policy(self):
        # Softmax policy to compute probability of playing action 1
        np.seterr(over='ignore')
        prob_choose_orange = 1 / (1 + np.exp(-self.evidence / self.temperature))
        np.seterr(over='warn')
        return prob_choose_orange

    def reset(self, n_blocks: int):
        self.n_blocks = n_blocks
        self.evidence = np.zeros((n_blocks, 1))

    def __repr__(self):
        return f"IntegrationModel(lk={self.leak: .2f}, t={self.temperature: .2f})"

if __name__ == '__main__':
    demo_task = SequentialPerceptionTask(8, 1)
    model = IntegrationModel(leak=0.1, temperature=0.001)
    model.simulate(demo_task, plot=True)
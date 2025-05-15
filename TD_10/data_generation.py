from typing import Optional

import numpy as np
import pandas as pd
import scipy

from models import Model, Parameter
from sequential_perception_task import SequentialPerceptionTask, dist_orange, dist_blue


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
        log_odds_ratio = np.log(p_orange) - np.log(p_blue)
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
        self.evidence: Optional[np.ndarray[float]] = None

    def update(self, stimulus):
        update_condition = np.absolute(stimulus) > np.absolute(self.evidence)
        self.evidence = np.where(update_condition, stimulus, self.evidence)

    def policy(self):
        greedy_decision = (self.evidence > 0).astype(int)
        p_orange = greedy_decision * (1 - self.epsilon) + (1 - greedy_decision) * self.epsilon
        return p_orange.astype(float)

    def reset(self, n_blocks: int):
        self.n_blocks = n_blocks
        self.evidence = np.zeros((n_blocks, 1))

    def __repr__(self):
        return f'ExtremumDetectionModel(epsilon={self.epsilon: .2f})'


def generate_dataset(models: list, task, seed = None):
    np.random.seed(seed)
    models_choices = []
    for model in models:
        model.reset(task.n_blocks)
        choices = model.simulate(task)
        models_choices.append(choices)
    return np.hstack(models_choices).T


if __name__ == '__main__':
    n_blocks = 200
    seed=42

    models = [
        IntegrationModel(0.1, 0.5),
        IntegrationModel(0.1, 0.1),
        ExtremumDetectionModel(0.3),
        IntegrationModel(0.8, 0.5),
        ExtremumDetectionModel(0.1),
        IntegrationModel(0.8, 0.1),
        IntegrationModel(0.5, 0.5),
        IntegrationModel(0.5, 0.1),
        ExtremumDetectionModel(0.5),
    ]
    task = SequentialPerceptionTask(n_trials=8, n_blocks=n_blocks, seed=seed)
    choices = generate_dataset(models, task, seed=seed)
    data = pd.DataFrame(choices, index=pd.Index([f"S{i+1:02d}" for i in range(len(models))], name='subject'))
    data.to_csv('data.csv', index=True, header=True)




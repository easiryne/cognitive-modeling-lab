from typing import Optional

import numpy as np
from pybads import OptimizeResult, BADS



class RLModel:
    n_params = 2
    def __init__(self, learning_rate: float, temperature: float):
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.n_blocks = None
        self.q_values = None

    def simulate(self, task, plot=False):
        raise NotImplementedError("We don't need the simulate() method in this TD.")

    def log_likelihood(self, actions: np.ndarray, rewards: np.ndarray) -> float:
        self._check_inputs(actions, rewards)
        n_blocks, n_trials = actions.shape
        self.reset(n_blocks)
        block_loglikelihood = np.zeros((n_blocks, 1))
        for trial in range(n_trials):
            action = actions[:, trial].reshape(n_blocks, 1)
            reward = rewards[:, trial].reshape(n_blocks, 1)
            prob_a1 = self.policy()
            prob_chosen_action = prob_a1 * action + (1 - prob_a1) * (1 - action)
            prob_chosen_action = 1e-6 + (1 - 1e-6 * 2) * prob_chosen_action ## Little trick to avoid numerical errors
            block_loglikelihood += np.log(prob_chosen_action)
            self.update(action, reward)
        log_likelihood = block_loglikelihood.sum()
        return log_likelihood

    def policy(self):
        q0 = self.q_values[:,0].reshape(self.n_blocks, 1)
        q1 = self.q_values[:,1].reshape(self.n_blocks, 1)
        np.seterr(over='ignore')  # Silence the warning for overflow
        prob = 1 / (1 + np.exp(-(q1 - q0) / self.temperature))
        np.seterr(over='warn')
        return prob

    def update(self, action: int, reward: float):
        update_idx = (np.arange(self.n_blocks), action.reshape(-1))
        td_error = reward.reshape(-1) - self.q_values[update_idx]
        self.q_values[update_idx] += self.learning_rate * td_error

    def reset(self, n_blocks: int):
        self.n_blocks = n_blocks
        self.q_values = np.full((n_blocks, 2), 0.5)

    def __repr__(self):
        return f'RLModel(lr={self.learning_rate: .3f}, t={self.temperature: .3f})'

    def _check_inputs(self, actions: np.ndarray, rewards: np.ndarray):
        if actions.ndim != 2 or rewards.ndim != 2:
            raise ValueError("'actions' and 'rewards' must be 2-dimensional arrays.")
        if actions.shape[0] != rewards.shape[0]:
            raise ValueError("'actions' and 'rewards' must have the same number of rows (blocks)")
        if actions.shape[1] != rewards.shape[1]:
            raise ValueError("'actions and 'rewards' must have the same number of columns (trials)")

    @classmethod
    def fit(cls, actions: np.ndarray[int], rewards: np.ndarray[float], verbose: bool = True):
        def cost_function(params):
            model = cls(*params)
            log_likelihood = model.log_likelihood(actions, rewards)
            return -log_likelihood

        optimizer = BADS(
            fun=cost_function,
            x0=[0.5, 0.1],
            lower_bounds=[0, 0.001],
            upper_bounds=[1, 1],
            plausible_lower_bounds=[0.2, 0.01],
            plausible_upper_bounds=[0.9, 0.5],
            options={
                'display': 'iter' if verbose else 'off'
            }
        )
        result: OptimizeResult = optimizer.optimize()
        optimal_params: np.ndarray = result.x
        return cls(*optimal_params)


class BiasRLModel(RLModel):
    n_params = 3
    def __init__(self, learning_rate: float, temperature: float, bias: float):
        super().__init__(learning_rate, temperature)
        self.bias = bias

    def policy(self):
        q0 = self.q_values[:,0].reshape(self.n_blocks, 1)
        q1 = self.q_values[:,1].reshape(self.n_blocks, 1)
        np.seterr(over='ignore')
        prob = 1 / (1 + np.exp(-(q1 - q0 + self.bias) / self.temperature))
        np.seterr(over='warn')
        return prob

    def __repr__(self):
        return f'BiasRLModel(lr={self.learning_rate: .3f}, t={self.temperature: .3f}, b={self.bias: .3f})'

    @classmethod
    def fit(cls, actions: np.ndarray[int], rewards: np.ndarray[float], verbose: bool = True):
        def cost_function(params):
            model = cls(*params)
            log_likelihood = model.log_likelihood(actions, rewards)
            return -log_likelihood

        optimizer = BADS(
            fun=cost_function,
            x0=[0.5, 0.1, 0],
            lower_bounds=[0, 0.001, -0.2],
            upper_bounds=[1, 1, 0.2],
            plausible_lower_bounds=[0.2, 0.01, -0.1],
            plausible_upper_bounds=[0.9, 0.5, 0.1],
            options={
                'display': 'iter' if verbose else 'off'
            }
        )
        result: OptimizeResult = optimizer.optimize()
        optimal_params: np.ndarray = result.x
        return cls(*optimal_params)


class RepeatRLModel(RLModel):
    n_params = 3
    def __init__(self, learning_rate: float, temperature: float, repeat: float):
        super().__init__(learning_rate, temperature)
        self.repeat = repeat
        self.last_action: Optional[np.ndarray] = None

    def policy(self):
        q0 = self.q_values[:,0].reshape(self.n_blocks, 1)
        q1 = self.q_values[:,1].reshape(self.n_blocks, 1)
        if self.last_action is None:
            repeat = np.zeros((self.n_blocks, 1))
        else:
            repeat = -self.repeat + 2 * self.repeat * self.last_action
        np.seterr(over='ignore')
        prob = 1 / (1 + np.exp(-(q1 - q0 + repeat) / self.temperature))
        np.seterr(over='warn')
        return prob

    def update(self, action: int, reward: float):
       super().update(action, reward)
       self.last_action = action

    def __repr__(self):
        return f'RepeatRLModel(lr={self.learning_rate: .3f}, t={self.temperature: .3f}, r={self.repeat: .3f})'

    @classmethod
    def fit(cls, actions: np.ndarray[int], rewards: np.ndarray[float], verbose: bool = True):
        def cost_function(params):
            model = cls(*params)
            log_likelihood = model.log_likelihood(actions, rewards)
            return -log_likelihood

        optimizer = BADS(
            fun=cost_function,
            x0=[0.5, 0.1, 0],
            lower_bounds=[0, 0.001, -0.2],
            upper_bounds=[1, 1, 0.2],
            plausible_lower_bounds=[0.2, 0.01, -0.1],
            plausible_upper_bounds=[0.9, 0.5, 0.1],
            options={
                'display': 'iter' if verbose else 'off'
            }
        )
        result: OptimizeResult = optimizer.optimize()
        optimal_params: np.ndarray = result.x
        return cls(*optimal_params)

if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv('OLD_experimental_data.csv')
    def extract_subject_data(id: int, data=data):
        subj_data = data[data['subject_id'] == id]
        block_actions = []
        block_rewards = []
        for block in [1, 2, 3, 4, 5, 6, 7, 8]:
            block_actions.append(subj_data[subj_data['block'] == block]['action'])
            block_rewards.append(subj_data[subj_data['block'] == block]['reward'])
        actions = np.stack(block_actions, axis=0)
        rewards = np.stack(block_rewards, axis=0)
        return actions, rewards

    actions, rewards = extract_subject_data(1, data)
    val_block = 0
    fit_actions = np.delete(actions, val_block, axis=0)
    fit_rewards = np.delete(rewards, val_block, axis=0)

    fitted_model = RLModel.fit(fit_actions, fit_rewards, verbose=True)
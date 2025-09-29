from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from pybads import OptimizeResult, BADS


class RLModel:
    n_params = 2
    def __init__(self, learning_rate: float, temperature: float):
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.q_values = np.array([0.5, 0.5])

    def simulate(self, task, plot=False):
        self.reset()
        actions = []
        probs = []
        rewards = []
        for trial_rewards in task:
            prob = self.policy()
            action = np.random.binomial(1, prob)
            reward = trial_rewards[action]
            self.update(action, reward)
            actions.append(action)
            probs.append(prob)
            rewards.append(reward)
        if plot:
            task.plot()
            plt.plot(probs, label='P(action=1)', color='green')
            plt.scatter(list(range(len(actions))), actions, label='actions', color='red')
            plt.legend()
            plt.title(f"{plt.gca().get_title()}\nwith {self}\ntotal reward = {sum(rewards):.2f}")
        return np.array(actions, dtype=int), np.array(probs, dtype=float), np.array(rewards, dtype=float)

    def log_likelihood(self, actions: np.ndarray, rewards: np.ndarray) -> float:
        self.reset()
        loglikelihood = 0
        for action, reward in zip(actions, rewards):
            prob_a1 = self.policy()
            prob_chosen_action = prob_a1 if action == 1 else 1 - prob_a1
            prob_chosen_action = np.max([prob_chosen_action, 1e-6]) ## Little trick to avoid numerical errors
            loglikelihood += np.log(prob_chosen_action)
            self.update(action, reward)
        return loglikelihood

    def policy(self):
        np.seterr(over='ignore')  # Silence the warning for overflow
        prob = 1 / (1 + np.exp(-(self.q_values[1] - self.q_values[0]) / self.temperature))
        np.seterr(over='warn')
        return prob

    def update(self, action: int, reward: float):
        td_error = reward - self.q_values[action]
        self.q_values[action] += self.learning_rate * td_error

    def reset(self):
        self.q_values = np.array([0.5, 0.5])

    def __repr__(self):
        return f'RLModel(lr={self.learning_rate: .3f}, t={self.temperature: .3f})'

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
        np.seterr(over='ignore')
        prob = 1 / (1 + np.exp(-(self.q_values[1] - self.q_values[0] + self.bias) / self.temperature))
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
        self.last_action: Optional[int] = None

    def policy(self):
        if self.last_action is None:
            repeat = 0
        elif self.last_action == 1:
            repeat = self.repeat
        elif self.last_action == 0:
            repeat = -self.repeat
        np.seterr(over='ignore')
        prob = 1 / (1 + np.exp(-(self.q_values[1] - self.q_values[0] + repeat) / self.temperature))
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

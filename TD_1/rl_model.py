import numpy as np
from pybads import BADS

EPSILON = 1e-8

class RLModel:
    def __init__(self, learning_rate: float, temperature: float):
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.q_values = np.array([0.5, 0.5])


    def simulate(self, bandit_data: np.ndarray):
        assert bandit_data.ndim == 2 and bandit_data.shape[0] == 2, \
            "You must pass ONE bandit trajectory at a time. Input to simulate() should have shape (2, n_trials)"
        actions = []
        probs = []
        # Loop through the trials
        for i_trial in range(bandit_data.shape[1]):
            # The policy gives the probability of choosing the right arm
            prob: float = self.softmax_policy(self.q_values)
            # Action is 0 if the left arm is chosen, 1 if the right arm is chosen
            action: int = np.random.binomial(1, prob)

            probs.append(prob)
            actions.append(action)

            # The bandit trajectory contains how much reward is given for the chosen action
            reward: float = bandit_data[action, i_trial]
            # The q-value of the chosen action is updated according to the temporal-difference learning rule
            reward_prediction_error = reward - self.q_values[action]
            self.q_values[action] += self.learning_rate * reward_prediction_error

        return np.array(actions), np.array(probs)

    def softmax_policy(self, q_values):
        if self.temperature == 0:
            return float(q_values[1] >= q_values[0])
        else:
            np.seterr('ignore')
            prob = 1 / (1 + np.exp(-np.array(q_values[1] - q_values[0])/self.temperature))
            np.seterr('warn')
            return prob

    @classmethod
    def fit(cls, your_actions, your_rewards):
        def negative_log_likelihood_objective_function(params):
            model = cls(*params)
            log_likelihood = model.log_likelihood(your_actions, your_rewards)
            if not np.isfinite(log_likelihood):
                return -np.finfo(np.float64).max
            else:
                return -log_likelihood

        optimizer = BADS(
            fun=negative_log_likelihood_objective_function,
            x0=np.array([0.5, 0.1]),
            lower_bounds=np.array([0, 0]),
            upper_bounds=np.array([1, 1]),
            plausible_lower_bounds=np.array([0.1, 0.01]),
            plausible_upper_bounds=np.array([0.9, 0.5]),
            options={
                'display': 'iter'
            }
        )
        result = optimizer.optimize()
        optimal_params = {'learning_rate': result.x[0],
                          'temperature': result.x[1]}

        return cls(**optimal_params)

    def log_likelihood(self, your_actions, your_rewards):
        sum_log_probs = 0
        for i_trial in range(len(your_actions)):
            action = your_actions[i_trial]
            reward = your_rewards[i_trial]

            # Get probability of action 1 from the model
            prob = self.softmax_policy(self.q_values)
            prob = EPSILON + (1 - EPSILON * 2) * prob
            # Update log_prob sum with the action you chose
            if action == 1:
                sum_log_probs += np.log(prob)
            else:
                sum_log_probs += np.log(1-prob)

            # Update the q-values of the model with the reward you got
            reward_prediction_error = reward - self.q_values[action]
            self.q_values[action] += self.learning_rate * reward_prediction_error
        return sum_log_probs




import numpy as np


class SimplifiedRLModel:
    def __init__(self, temperature: float):
        self.temperature = temperature
        self.q_values = np.array([0.5, 0.5]) # At t=0, both arms are assumed to have a value of 0.5

    def set_q_values(self, q_0: float, q_1: float):
        self.q_values = np.array([q_0, q_1])
        return self

    def policy(self):
        # Complete this code
        prob = 1 / (1 + np.exp(-(self.q_values[1] - self.q_values[0]) / self.temperature))
        return prob


def test_rl_model_policy(rl_model_class: SimplifiedRLModel):
    try:
        # Test basic computation without temperature
        model = rl_model_class(1)
        prob = model.policy()
        assert isinstance(prob, float), "❌ policy() should return a float"
        assert prob == 0.5, "❌ policy() should return 0.5 on first trial"

        model.q_values = np.array([0.4, 0.6])
        prob = model.policy()
        expected_prob = 0.5498339973124778
        assert np.isclose(prob, expected_prob), "❌ incorrect softmax rule: it does not use the q_values correctly"

        # Test computation with temperature
        temp = 0.2
        model = rl_model_class(temp)
        model.q_values = np.array([0.4, 0.6])
        prob = model.policy()
        expected_prob = 0.7310585786300049
        assert np.isclose(prob, expected_prob), "❌ incorrect softmax rule: it does not use the temperature correctly"
        print("✅ OK 👌")
    except AssertionError as e:
        print(e)


def test_likelihood_for_a_trajectory(likelihood_func):
    neutral_model = SimplifiedRLModel(1)
    actions = np.array([0, 0, 0, 1, 1])
    try:
        assert isinstance(likelihood_func(neutral_model, actions), float), "❌ likelihood_func should return a float"
        assert 1 >= likelihood_func(neutral_model, actions) >= 0, "❌ likelihood should be 0 <= likelihood <= 1"
        assert likelihood_func(neutral_model, actions) == 0.5**5, "❌ Incorrect calculation of the likelihood"
        print("✅ OK 👌")
    except AssertionError as e:
        print(e)


def test_log_likelihood_for_a_trajectory(log_likelihood_func):
    try:
        neutral_model = SimplifiedRLModel(1)
        actions = np.array([0, 0, 0, 1, 1])

        assert isinstance(log_likelihood_func(neutral_model, actions), float), "❌ likelihood_func should return a float"
        assert log_likelihood_func(neutral_model, actions) < 0, "❌ log likelihood should be negative"
        assert log_likelihood_func(neutral_model, actions) == np.log(0.5) * 5, "❌ Incorrect calculation of the likelihood"
        print("✅ OK 👌")
    except AssertionError as e:
        print(e)

def test_rl_model_log_likelihood(rl_model_class):
    try:
        model = rl_model_class(1, 1)
        actions = np.array([0, 0, 0, 1, 1])
        neutral_rewards = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        ll = model.log_likelihood(actions, neutral_rewards)

        assert isinstance(ll, float), "❌ log_likelihood() should return a float"
        assert ll < 0, "❌ log_likelihood() should return a negative value"
        assert ll == np.log(0.5) * 5, "❌ log_likelihood() should return the sum of the log likelihood of each action"

        yes_model = rl_model_class(1, 0.00001)
        actions = np.array([0, 1, 1, 1, 1])
        rewards = np.array([0., 1., 1., 1., 1.])

        ll = yes_model.log_likelihood(actions, rewards)
        assert np.isclose(ll, np.log(0.5)), "❌ log_likelihood() should update the q_values of the model at each trial"

        print("✅ OK 👌")
    except AssertionError as e:
        print(e)
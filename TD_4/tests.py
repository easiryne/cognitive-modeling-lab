import numpy as np


def test_bandit_task_generate_trajectories(bandit_task_class):
    try:
        test_means, test_rewards = bandit_task_class(0.4, 0.7, std=0.1, n_trials=40).generate_trajectories()
        # Test means
        assert isinstance(test_means, np.ndarray) , "❌ 'means' should be a NumPy array"
        assert test_means.shape == (40, 2), '❌ The means array should be of shape (n_trials, 2)'
        assert np.all(test_means[:, 0] == 0.4), '❌ The means of arm 0 are incorrect'
        assert np.all(test_means[:, 1] == 0.7), '❌ The means of arm 1 are incorrect'

        # Test rewards
        assert isinstance(test_rewards, np.ndarray), "❌ 'rewards' should be a NumPy array"
        assert test_rewards.shape == (40, 2), '❌ The rewards array should be of shape (n_trials, 2)'
        assert np.isclose(test_rewards[:, 0].mean(), 0.4, atol=0.1), '❌ The rewards of arm 0 are not centered on its mean'
        assert np.isclose(test_rewards[:, 1].mean(), 0.7, atol=0.1), '❌ The rewards of arm 1 are not centered on its mean'
        assert np.isclose(test_rewards[:, 0].std(), 0.1,
                          atol=0.1), '❌ The rewards of arm 0 do not have the correct standard deviation'
        assert np.isclose(test_rewards[:, 1].std(), 0.1,
                          atol=0.1), '❌ The rewards of arm 1 do not have the correct standard deviation'
        _, clipped_rewards = bandit_task_class(0.4, 0.9, std=0.2, n_trials=40).generate_trajectories()
        assert np.all(clipped_rewards >= 0) and np.all(clipped_rewards <= 1), '❌ The rewards should be clipped between 0 and 1'
        print("✅OK 👌")
    except AssertionError as e:
        print(e)


def test_reversal_bandit_task_generate_trajectories(reversal_bandit_task_class):
    try:
        test_means, test_rewards = reversal_bandit_task_class(0.4, 0.7, std=0.1, n_trials=40).generate_trajectories()
        assert isinstance(test_means, np.ndarray) and isinstance(test_rewards, np.ndarray), '❌Both means and rewards should be NumPy arrays'
        assert test_means.shape == (40, 2), '❌ The means array should be of shape (n_trials, 2)'
        assert test_rewards.shape == (40, 2), '❌ The rewards array should be of shape (n_trials, 2)'
        assert np.all(test_means[:20, 0] == 0.4), '❌ The means of arm 0 are incorrect before the reversal'
        assert np.all(test_means[:20, 1] == 0.7), '❌ The means of arm 1 are incorrect before the reversal'
        assert np.all(test_means[20:, 0] == 0.7), '❌ The means of arm 0 are incorrect after the reversal'
        assert np.all(test_means[20:, 1] == 0.4), '❌ The means of arm 1 are incorrect after the reversal'
        assert np.isclose(test_rewards[:20, 0].mean(), 0.4, atol=0.1), '❌ The rewards of arm 0 are not centered on its mean in the first half'
        assert np.isclose(test_rewards[:20, 1].mean(), 0.7, atol=0.1), '❌ The rewards of arm 1 are not centered on its mean in the first half'
        assert np.isclose(test_rewards[20:, 0].mean(), 0.7, atol=0.1), '❌ The rewards of arm 0 are not centered on its mean in the second half'
        assert np.isclose(test_rewards[20:, 1].mean(), 0.4, atol=0.1), '❌ The rewards of arm 1 are not centered on its mean in the second half'
        assert np.isclose(test_rewards[:20, 0].std(), 0.1, atol=0.1), '❌ The rewards of arm 0 do not have the correct standard deviation in the first half'
        assert np.isclose(test_rewards[:20, 1].std(), 0.1, atol=0.1), '❌ The rewards of arm 1 do not have the correct standard deviation in the first half'
        assert np.isclose(test_rewards[20:, 1].std(), 0.1, atol=0.1), '❌ The rewards of arm 1 do not have the correct standard deviation in the second half'
        assert np.isclose(test_rewards[20:, 0].std(), 0.1, atol=0.1), '❌ The rewards of arm 0 do not have the correct standard deviation in the second half'
        print("✅OK 👌")
    except AssertionError as e:
        print(e)


def test_rl_model_policy(rl_model_class):
    try:
        # Test basic computation without temperature
        model = rl_model_class(0.1, 1)
        prob = model.policy()
        assert isinstance(prob, float), "❌ policy() should return a float"
        assert prob == 0.5, "❌ policy() should return 0.5 on first trial"

        model.q_values = np.array([0.4, 0.6])
        prob = model.policy()
        expected_prob = 0.5498339973124778
        assert np.isclose(prob, expected_prob), "❌ incorrect softmax rule: it does not use the q_values correctly"

        # Test computation with temperature
        temp = 0.2
        model = rl_model_class(0.1, temp)
        model.q_values = np.array([0.4, 0.6])
        prob = model.policy()
        expected_prob = 0.7310585786300049
        assert np.isclose(prob, expected_prob), "❌ incorrect softmax rule: it does not use the temperature correctly"
        print("✅OK 👌")
    except AssertionError as e:
        print(e)

def test_rl_model_update(rl_model_class):
    try:
        # Test with learning rate of 1
        lr = 1
        model = rl_model_class(lr, 1)
        action = 0
        reward = 1
        model.update(action, reward)
        assert model.q_values[0] != 0.5, "❌ update() should change the q_value of the chosen action"
        assert model.q_values[1] == 0.5, "❌ update() should not change the q_value of the unchosen action"
        assert model.q_values[0] == reward, "❌ incorrect update rule: the q-value of the chosen action should move towards the reward"

        # Test with learning rate of 0
        lr = 0
        model = rl_model_class(lr, 1)
        model.update(1, 1)
        assert model.q_values[0] == 0.5, "❌ update() should not change the q_value of the chosen action with learning rate 0"
        assert model.q_values[1] == 0.5, "❌ update() should not change the q_value of the unchosen action"

        # Test with learning rate between 0 and 1
        lr = 0.5
        model = rl_model_class(lr, 1)
        action = 1
        reward = 1
        model.update(action, reward)
        assert model.q_values[1] == 0.75, "❌ incorrect update rule: the TD error should be applied according to the learning rate"
        assert model.q_values[0] == 0.5, "❌ update() should not change the q_value of the unchosen action"
        print("✅OK 👌")
    except AssertionError as e:
        print(e)


def test_rl_model_simulate(rl_model_class):
    try:
        bandit_rewards = np.array([
            [0.3, 0.5], # 0.45, 0.5
            [0.5, 0.7], # 0.45, 0.6
            [0.5, 0.8], # 0.45, 0.7
            [0.5, 0], # 0.45, 0.35
            [1, 0.5], # 0.75, 0.35
        ])
        model = rl_model_class(0.5, 0.000001)
        model.q_values = np.array([0.6, 0.5]) # Bias the model for choosing action 0 first

        actions, probs, rewards = model.simulate(bandit_rewards)

        # Check output validity
        assert isinstance(actions, np.ndarray) and isinstance(probs, np.ndarray) and isinstance(rewards, np.ndarray), "❌ simulate() should return three NumPy arrays"
        assert actions.shape == probs.shape == rewards.shape == (5,), "❌ actions, probs and rewards should have shape (n_trials,)"
        assert np.all((actions == 0) | (actions == 1)), "❌ actions should be 0 or 1"
        assert np.all((probs >= 0) & (probs <= 1)), "❌ probabilities should be between 0 and 1"
        assert np.all((rewards >= 0) & (rewards <= 1)), "❌ rewards should be between 0 and 1"

        expected_actions = np.array([0, 1, 1, 1, 0])
        expected_probs = np.array([0., 1., 1., 1., 0.])
        expected_rewards = np.array([0.3, 0.7, 0.8, 0, 1])
        assert np.all(actions == expected_actions), "❌ the returned actions are incorrect"
        assert np.allclose(probs , expected_probs), "❌ the returned probabilities are incorrect"
        assert np.allclose(rewards, expected_rewards), "❌ the returned rewards are incorrect"

        print("✅OK 👌")
    except AssertionError as e:
        print(e)
import numpy as np

from TD_10.sequential_perception_task import SequentialPerceptionTask, dist_orange, dist_blue


def test_integration_model_update(model_class):
    n_blocks = 8
    leak = 0.5
    model = model_class(0.5, 1e-6)
    model.reset(n_blocks)
    stimulus = np.ones((n_blocks, 1))

    model.update(stimulus)
    evidence_1 = model.evidence

    assert isinstance(evidence_1, np.ndarray), "Your updated evidence should be a numpy array"
    assert evidence_1.shape == (n_blocks, 1), "Your updated evidence should have shape (n_blocks, 1)"
    assert not np.array_equal(evidence_1, np.zeros((n_blocks, 1))), "Your evidence doesn't seem to be updated."

    model.update(stimulus)
    evidence_2 = model.evidence

    expected_evidence_2 = evidence_1 * (1 - leak) + (np.log(dist_orange.pdf(stimulus)) -np.log(dist_blue.pdf(stimulus)))
    np.testing.assert_almost_equal(evidence_2, expected_evidence_2)
    print("OK 👌")

def test_integration_model_policy(model_class):
    n_blocks = 4
    temp = 0.5
    model = model_class(0, temp)
    model.reset(n_blocks)
    model.evidence = np.array([[0.1], [0.1], [-0.1], [-0.1]])

    prob_choose_orange = model.policy()
    print(prob_choose_orange.shape)

    assert isinstance(prob_choose_orange, np.ndarray), "Your policy should return a numpy array"
    assert prob_choose_orange.shape == (n_blocks, 1), "Your policy should return an array of shape (n_blocks, 1)"
    assert (prob_choose_orange >= 0).all() and (prob_choose_orange <= 1).all(), "Your policy should return probabilities between 0 and 1"
    assert np.allclose(prob_choose_orange, 1 / (1 + np.exp(-model.evidence / temp))), "Your policy doesn't seem to be correct."
    print("OK 👌")

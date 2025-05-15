import numpy as np

from sequential_perception_task import dist_orange, dist_blue


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

    assert isinstance(prob_choose_orange, np.ndarray), "Your policy should return a numpy array"
    assert prob_choose_orange.shape == (n_blocks, 1), "Your policy should return an array of shape (n_blocks, 1)"
    assert (prob_choose_orange >= 0).all() and (prob_choose_orange <= 1).all(), "Your policy should return probabilities between 0 and 1"
    assert np.allclose(prob_choose_orange, 1 / (1 + np.exp(-model.evidence / temp))), "Your policy doesn't seem to be correct."
    print("OK 👌")

def test_extremum_model_update(model_class):
    n_blocks = 4
    model = model_class(0)
    model.reset(n_blocks)

    stimulus_1 = np.array([[0.1], [0.1], [-0.1], [-0.1]])
    model.update(stimulus_1)
    evidence_1 = model.evidence
    assert isinstance(evidence_1, np.ndarray), "Your updated evidence should be a numpy array"
    assert evidence_1.shape == (n_blocks, 1), "Your updated evidence should have shape (n_blocks, 1)"
    assert not np.array_equal(evidence_1, np.zeros((n_blocks, 1))), "Your evidence doesn't seem to be updated."
    assert not np.array_equal(evidence_1, np.full((n_blocks, 1), 0.1)), "Your evidence should be equal to the salient stimulus, not its absolute value."
    assert np.allclose(evidence_1, stimulus_1), "A more salient stimulus should update the evidence."

    stimulus_2 = np.array([[0.2], [0.2], [-0.2], [-0.2]])
    model.update(stimulus_2)
    evidence_2 = model.evidence
    assert np.allclose(evidence_2, stimulus_2), "A more salient stimulus should update the evidence."

    stimulus_3 = np.array([[0.15], [0.15], [-0.15], [-0.15]])
    model.update(stimulus_3)
    evidence_3 = model.evidence
    assert np.allclose(evidence_3, stimulus_2), "A less salient stimulus should not update the evidence."

    stimulus_4 = np.array([[-0.3], [0.3], [-0.3], [0.3]])
    model.update(stimulus_4)
    evidence_4 = model.evidence
    assert np.allclose(evidence_4, stimulus_4), "Salience should be compared in absolute value"

    print("OK 👌")



def test_extremum_model_policy(model_class):
    n_blocks = 4
    epsilon = 0.3
    model = model_class(epsilon)
    model.reset(n_blocks)

    model.evidence = np.array([[0.1], [0.1], [-0.1], [-0.1]])
    p_orange = model.policy()

    assert isinstance(p_orange, np.ndarray), "Your policy should return a numpy array"
    assert p_orange.shape == (n_blocks, 1), "Your policy should return an array of shape (n_blocks, 1)"

    expected_p = np.array([[0.7], [0.7], [0.3], [0.3]])
    assert p_orange[0][0] < 0.999, "You don't take epsilon into account."
    np.testing.assert_almost_equal(p_orange, expected_p), "Your computation doesn't seem to be correct."
    print("OK 👌")
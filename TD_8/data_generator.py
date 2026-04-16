import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from TD_8.bandit_task import ReversalBanditTask
from TD_8.rl_models import RLModel, BiasRLModel, RepeatRLModel


def simulate(model, task):
    model.reset(1)
    actions = []
    probs = []
    rewards = []
    for trial_rewards in task:
        prob: float = model.policy().item()
        action: float = np.random.binomial(1, prob)
        reward: float = trial_rewards[action]
        model.update(np.array([[action]]), np.array([[reward]]))
        actions.append(action)
        probs.append(prob)
        rewards.append(reward)
    return np.array(actions, dtype=int), np.array(probs, dtype=float), np.array(rewards, dtype=float)


def generate_experimental_data(task, models: list, n_blocks: int, seed=42):
    """
    Generate experimental data using the specified model.
    """
    np.random.seed(seed)
    data = []
    for i, model in enumerate(models):
        subject_id = f"SUBJ0{i+1}"
        for b in range(n_blocks):
            # Simulate the task
            task.reset(seed + b)
            actions, _, rewards = simulate(model, task)
            for j in range(len(actions)):
                data.append(pd.Series({
                    'subject_id': subject_id,
                    'block': b,
                    'trial': j,
                    'action': actions[j],
                    'reward': rewards[j],
                }))
    return pd.concat(data, axis=1).T


if __name__ == '__main__':
    seed = 42
    models = [
        RLModel(learning_rate=0.5, temperature=0.1),
        BiasRLModel(learning_rate=0.5, temperature=0.1, bias=0.1),
        RepeatRLModel(learning_rate=0.05, temperature=0.1, repeat=0.1)
    ]
    np.random.seed(seed)
    task = ReversalBanditTask(0.4, 0.6, 0.25, 100)
    task.plot()
    plt.show()
    data = generate_experimental_data(task, models, n_blocks=5, seed=42)
    data.to_csv('experimental_data.csv', index=False)
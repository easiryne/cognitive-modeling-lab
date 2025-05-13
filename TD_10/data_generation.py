import numpy as np
import pandas as pd

from TD_10.models import ExtremumDetectionModel, IntegrationModel
from sequential_perception_task import SequentialPerceptionTask


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
        ExtremumDetectionModel(0.1),
        ExtremumDetectionModel(0.2),
        ExtremumDetectionModel(0.3),
        ExtremumDetectionModel(0.4),
        IntegrationModel(0.1, 0.1),
        IntegrationModel(0.1, 0.5),
        IntegrationModel(0.5, 0.1),
        IntegrationModel(0.5, 0.5),
    ]
    task = SequentialPerceptionTask(n_trials=8, n_blocks=n_blocks, seed=seed)
    choices = generate_dataset(models, task, seed=seed)
    data = pd.DataFrame(choices, index=pd.Index([f"S{i+1:02d}" for i in range(len(models))], name='subject'))
    data.to_csv('data.csv', index=True, header=True)




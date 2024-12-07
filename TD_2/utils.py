import warnings

import numpy as np
import pandas as pd
from pybads import BADS

SEED = 42

def generate_data(coherence_range, trials_per_coherence_rate) -> pd.DataFrame:
    A = 20
    k = 0.01
    t_R = 100

    coherence_rates = []
    reaction_times = []
    outcomes = []

    for x in coherence_range:
        mean = A / (k * x) * np.tanh(A * k * x) + t_R
        std = mean / 4
        rt = np.random.normal(mean, std, trials_per_coherence_rate)
        p_c = 1 / (1 + np.exp(-2 * A * k * np.abs(x)))
        out = np.random.binomial(1, p_c, size=trials_per_coherence_rate)

        coherence_rates.extend([x] * trials_per_coherence_rate)
        reaction_times.extend(list(rt))
        outcomes.extend(list(out))

    return pd.DataFrame({'motion_coherence': coherence_rates,
                         'reaction_time': reaction_times,
                         'outcome': outcomes})


def minimize(cost_function):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        optimizer = BADS(  # Params are A, k, t_R
            fun=cost_function,
            x0=np.array([20, 0.01, 100]),
            lower_bounds=np.array([0, 0.0001, 0]),
            upper_bounds=np.array([100, 1, 1000]),
            plausible_lower_bounds=np.array([5, 0.01, 10]),
            plausible_upper_bounds=np.array([50, 0.1, 500]),
            options={
                'display': 'iter',
                'random_seed': SEED
            }
        )
        result = optimizer.optimize()
    optimal_params = {
        'A': result.x[0].item(),
        'k': result.x[1].item(),
        't_R': result.x[2].item()
    }
    print("Parameter values that minimize the cost function:")
    print(optimal_params)
    return optimal_params
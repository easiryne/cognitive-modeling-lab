import warnings

import numpy as np
from pybads import BADS

SEED = 42


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
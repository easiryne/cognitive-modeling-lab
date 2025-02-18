import pandas as pd
import numpy as np
from scipy.stats import norm

num_trials = 1000

signal_prob = 0.5

block_selection_criteria = [0.5, -0.5, 1.5]
n_blocks = len(block_selection_criteria)

participant_d_prime = [1.5, 3]
n_participants = len(participant_d_prime)


def generate_data(participant_i, block_j):
    # Set parameters
    d_prime = participant_d_prime[participant_i]  # Sensitivity (higher means better discrimination)
    criterion = block_selection_criteria[block_j]  # Decision criterion

    # Generate trials
    is_signal = np.random.rand(num_trials) < signal_prob  # True if signal is present

    # Generate decision variable (normal distributions)
    decision_values = np.random.normal(loc=d_prime * is_signal, scale=1)

    # Apply criterion
    responses = decision_values > criterion  # Respond "signal" if value exceeds criterion

    # Compute response categories
    hits = np.sum(responses & is_signal)
    misses = np.sum(~responses & is_signal)
    false_alarms = np.sum(responses & ~is_signal)
    correct_rejections = np.sum(~responses & ~is_signal)

    # Compute rates
    hit_rate = hits / (hits + misses)
    false_alarm_rate = false_alarms / (false_alarms + correct_rejections)

    # Compute d' and c
    d_prime_est = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
    c_est = -0.5 * (norm.ppf(hit_rate) + norm.ppf(false_alarm_rate))

    # Print results
    print(f"--- Participant {participant_i}, Block {block_j} ---")
    print(f"Hit rate: {hit_rate:.3f}, False alarm rate: {false_alarm_rate:.3f}")
    print(f"Estimated d': {d_prime_est:.3f}, Estimated criterion: {c_est:.3f}")
    print()

    data = pd.DataFrame({
        'participant': participant_i + 1,
        'block': block_j + 1,
        'stim': is_signal.astype(int),
        'resp': responses.astype(int)
    })
    return data

if __name__ == '__main__':
    dfs = [generate_data(i, j) for i in range(n_participants) for j in range(n_blocks)]
    data = pd.concat(dfs, ignore_index=True)
    data.to_csv('data.csv', index=False)
import numpy as np
import pandas as pd


def generate_data(coherence_range, trials_per_coherence_rate, seed=42) -> pd.DataFrame:
    A = 20
    k = 0.01
    t_R = 100

    coherence_rates = []
    reaction_times = []
    outcomes = []

    np.random.seed(seed)
    for x in coherence_range:
        mean = A / (k * x) * np.tanh(A * k * x) + t_R
        std = mean / 4
        rt = np.random.normal(mean, std, trials_per_coherence_rate)
        p_c = 1 / (1 + np.exp(-2 * A * k * np.abs(x)))
        out = np.random.binomial(1, p_c, size=trials_per_coherence_rate)

        coherence_rates.extend([x] * trials_per_coherence_rate)
        reaction_times.extend(list(rt))
        outcomes.extend(list(out))

    stimulus = np.random.randint(0, 2, trials_per_coherence_rate * len(coherence_range))
    responses = []
    for out, stim in zip(outcomes, stimulus):
        if out == 1:
            responses.append(stim)
        else:
            responses.append(int(not bool(stim)))
    responses = np.array(responses)
    return pd.DataFrame({'motion_coherence': coherence_rates,
                         'reaction_time': reaction_times,
                         'stimulus': stimulus,
                        'response': responses})

if __name__ == '__main__':
    coherence_range = np.logspace(-1, 2, num=8, base=10)
    data = generate_data(coherence_range, trials_per_coherence_rate=100, seed=42)
    data.to_csv('data.csv')
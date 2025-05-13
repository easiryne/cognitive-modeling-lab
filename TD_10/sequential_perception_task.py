import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rv_continuous
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



class ColorDistribution(rv_continuous):
    def __init__(self, category: str):
        super().__init__(name=f"{category}_distribution")
        slope = 1.4386  # Fixed value of b for a distribution overlap of 1/3
        self.x = np.linspace(-1, 1, 2000)
        p_unnormalized = 1 / (1 + np.exp(-slope * self.x))
        if category == 'orange':
            p_unnormalized = p_unnormalized
        elif category == 'blue':
            p_unnormalized = np.flip(p_unnormalized, axis=0)
        else:
            raise ValueError(category)
        self.pdf_vals = p_unnormalized / simpson(p_unnormalized, self.x)
        self.cdf_vals = np.cumsum(self.pdf_vals) / np.sum(self.pdf_vals)

    def _pdf(self, x):
        return np.interp(x, self.x, self.pdf_vals)

    def _cdf(self, x):
        return np.interp(x, self.x, np.cumsum(self.pdf_vals) / np.sum(self.pdf_vals))

    def _ppf(self, q):
        return np.interp(q, np.cumsum(self.pdf_vals) / np.sum(self.pdf_vals), self.x)

    def rvs(self, size=1):
        rand_vals = np.random.rand(size)
        return self._ppf(rand_vals)


# Orange is coded as 1
dist_orange = ColorDistribution('orange')
# Blue is coded as 0
dist_blue = ColorDistribution('blue')


class SequentialPerceptionTask:
    def __init__(self, n_trials: int, n_blocks: int, seed: int = None):
        self.n_trials: int = n_trials
        self.n_blocks: int = n_blocks

        self.stimuli, self.ground_truth = self.generate_trajectories(seed)

    def generate_trajectories(self, seed = None) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(seed)
        ground_truth = np.random.choice([0, 1], size=(self.n_blocks,1))
        stimuli = np.zeros((self.n_blocks, self.n_trials), dtype=float)
        for block in range(self.n_blocks):
            if ground_truth[block][0] == 1:
                stimuli[block] = dist_orange.rvs(size=self.n_trials)
            else:
                stimuli[block] = dist_blue.rvs(size=self.n_trials)
        return stimuli, ground_truth

    def plot(self, show=True):
        n_blocks_show = min(self.n_blocks, 10)
        fig, axes = plt.subplots(n_blocks_show, 1, figsize=(5, (n_blocks_show + 4)/2))
        if n_blocks_show == 1:
            axes = [axes]

        for block_idx in range(n_blocks_show):
            ax = axes[block_idx]
            ax.set_xlim(0, self.n_trials + 3)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_aspect('equal')

            for trial_idx, stimulus in enumerate(self.stimuli[block_idx]):
                color = mcolors.to_hex(LinearSegmentedColormap.from_list("BGO", ['blue', 'gray', 'orange'])((stimulus + 1) / 2))
                circle = plt.Circle((trial_idx + 1, 0.5), 0.2, color=color)
                ax.add_artist(circle)
            ax.axvline(x=self.n_trials + 1, color='black', linewidth=1)
            truth_circle = plt.Circle((self.n_trials + 2, 0.5), 0.25, linewidth=2, edgecolor='black',
                                      facecolor='orange' if self.ground_truth[block_idx][0] == 1 else 'blue')
            ax.add_artist(truth_circle)
            ax.text(0.25, 0.5, f"{block_idx + 1}", horizontalalignment='center', verticalalignment='center')
        axes[0].text(self.n_trials//2 + 1, 1, "Trials", fontsize=12, ha='center', va='bottom')
        axes[0].text(self.n_trials + 2, 1, "ground truth", fontsize=8, ha='center', va='bottom')
        fig.text(0, 0.5, "Block", va='center', rotation='vertical', fontsize=14)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            return fig, axes

    def reset(self, seed = None):
        self.stimuli, self.ground_truth = self.generate_trajectories(seed)
        return self

    def __iter__(self)-> Tuple[np.ndarray, bool]:
        for t in range(self.n_trials):
            stimulus = self.stimuli[:, t]
            decide = t == self.n_trials - 1
            yield stimulus.reshape(self.n_blocks, 1), decide

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_trials} x {self.n_blocks}, std={self.std})"

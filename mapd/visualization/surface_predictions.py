from typing import Dict, Tuple

import numpy as np

from mapd.probes.utils.idx_dataset import IDXDataset
import matplotlib.pyplot as plt


def make_surface_predictions(predictions: Dict[int, Tuple[str, float]], dataset: IDXDataset,
                             probe_suite: str = "typical"):
    """
    Helper function to generate a surface plot of the probe suite predictions.

    Args:
        predictions(Dict[int, Tuple[str, float]]): A dictionary of predictions from a probe suite.
        dataset(IDXDataset): The dataset used to generate the probe suite.
        probe_suite(str): The probe suite to generate the surface plot for.
    """

    all_sample_indices_for_probe = [k for k, (ps, _) in predictions.items() if ps == probe_suite]

    N_ROWS, N_COLUMNS = 4, 4
    N = N_ROWS * N_COLUMNS

    sampled_indices = np.random.choice(all_sample_indices_for_probe, N, replace=False)

    fig, axs = plt.subplots(N_ROWS, N_COLUMNS, figsize=(20, 20))
    fig.suptitle(f"Probe Suite: {probe_suite}", fontsize=20)
    fig.tight_layout(pad=3.0)

    for ax, idx in zip(axs.flatten(), sampled_indices):
        (sample, label), _ = dataset[idx]

        ax.set_title(f"Label: {label} ({idx})")
        ax.imshow(sample.squeeze(), cmap="gray")

    return fig

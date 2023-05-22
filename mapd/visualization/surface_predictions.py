from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from mapd.probes.utils.idx_dataset import IDXDataset


def display_surface_predictions(
    predictions: Dict[int, Tuple[str, float]],
    dataset: Dataset,
    probe_suite: str = "typical",
    labels: Dict[int, str] = None,
    ordered: bool = False,
    display_sample_fn: Optional[Callable[[plt.Axes, Any], None]] = None,
) -> plt.Figure:
    """
    Helper function to generate a surface plot of the probe suite predictions.

    Args:
        predictions(Dict[int, Tuple[str, float]]): A dictionary of predictions from a probe suite.
        dataset(IDXDataset): The dataset used to generate the probe suite.
        probe_suite(str): The probe suite to generate the surface plot for.
        labels(Dict[int, str]): A dictionary of labels for the dataset.
        ordered(bool): Whether to order the surface plot by prediction probability.
        display_sample_fn(Optional[Callable[[plt.Axes, Any], None]]): A function to display the sample. If None,
        the sample is displayed as a grayscale image.

    Returns:
        plt.Figure: The figure containing the surface plot.
    """

    all_sample_indices_for_probe = [
        k for k, (ps, _) in predictions.items() if ps == probe_suite
    ]
    all_sample_probas = [
        prob for _, (ps, prob) in predictions.items() if ps == probe_suite
    ]

    N_ROWS, N_COLUMNS = 4, 4
    N = N_ROWS * N_COLUMNS

    if ordered:
        sampled_indices = [
            all_sample_indices_for_probe[idx]
            for idx in np.argsort(all_sample_probas)[-N:]
        ]
    else:
        sampled_indices = np.random.choice(
            all_sample_indices_for_probe, N, replace=False
        )

    fig, axs = plt.subplots(N_ROWS, N_COLUMNS, figsize=(20, 20))
    fig.suptitle(f"Probe Suite: {probe_suite}", fontsize=20)
    fig.tight_layout(pad=3.0)

    for ax, idx in zip(axs.flatten(), sampled_indices):
        sample = dataset[idx]

        if display_sample_fn is not None:
            display_sample_fn(ax, sample)
            continue

        img, label = sample
        label_str = labels[label] if labels is not None else str(label)

        ax.set_title(f"Label: {label_str} ({idx})")
        ax.imshow(img.squeeze().T, cmap="gray")

    return fig

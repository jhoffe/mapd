import os
import random
import warnings

import click
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.utils.plot_utils import (
    load_loss_dataset,
    load_probe_suite,
    plot_styles,
)

warnings.filterwarnings("ignore")


def loss_curve_plot(
    name: str,
    probe_suite_path: str,
    loss_dataset_path: str,
    output_path: str,
    rio: bool = False,
) -> None:
    """Plot the accuracy for each probe suite pr. epoch"""

    probe_suite = load_probe_suite(probe_suite_path=probe_suite_path)
    df = load_loss_dataset(loss_dataset_path=loss_dataset_path)

    df.drop_duplicates(subset=["epoch", "sample_index", "stage"], inplace=True)

    df["epoch"] = df["epoch"].astype(int)

    max_epoch = df["epoch"].max() + 1
    suite_indices = probe_suite.index_to_suite
    num_suite_samples = len(suite_indices)
    num_train_samples = len(probe_suite)

    assert df["stage"][df["stage"] == "val"].count() == max_epoch * num_suite_samples
    assert df["stage"][df["stage"] == "train"].count() == max_epoch * num_train_samples

    val_df = df[df["stage"] == "val"]

    suite_names = {
        "typical": "Typical",
        "atypical": "Atypical",
        "random_outputs": "Random Outputs",
        "corrupted": "Corrupted",
    }

    if rio:
        suite_names["random_inputs_outputs"] = "Random Inputs and Outputs"

    # sort by epoch
    val_df.sort_values(by=["epoch"], inplace=True)

    _, _, marker_colors, plot_titles = plot_styles()

    plt.figure(figsize=(10, 6))
    plt.title(f"Loss curves for {plot_titles[name]}") if not rio else plt.title(
        f"Loss curves for {plot_titles[name]} w. RIO"
    )
    i = 0

    for suite_attr, suite_name in suite_names.items():
        indices = [idx for idx, suite in suite_indices.items() if suite == suite_attr]
        random.shuffle(indices)

        rand_indices = indices[250:]

        val_df.loc[df["sample_index"].isin(rand_indices), "suite"] = suite_name
        for idx in rand_indices:
            plt.plot(
                val_df["epoch"][val_df["sample_index"] == idx],
                val_df["loss"][val_df["sample_index"] == idx],
                alpha=0.25,
                linewidth=0.1,
                color=marker_colors[i],
                zorder=1,
            )

        # plot aggregated loss for each suite over all samples
        plt.plot(
            np.arange(max_epoch),
            val_df.loc[val_df["suite"] == suite_name].groupby(["epoch"])["loss"].mean(),
            label=suite_name,
            linewidth=1,
            color=marker_colors[i],
            zorder=2,
        )

        i += 1

    plt.ylim(0, 10) if name == "cifar10" else plt.ylim(0, 14)
    plt.xlim(0, 120) if name == "cifar10" else plt.xlim(0, max_epoch)

    plt.legend(loc="upper right", fontsize="small")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if rio:
        plt.savefig(os.path.join(output_path, name, f"{name}_loss_curve_rio.png"))
    else:
        plt.savefig(os.path.join(output_path, name, f"{name}_loss_curve.png"))


def main(name, probe_suite_path, loss_dataset_path, output_path, rio=False):
    loss_curve_plot(
        name=name,
        probe_suite_path=probe_suite_path,
        loss_dataset_path=loss_dataset_path,
        output_path=output_path,
        rio=rio,
    )


@click.command()
@click.option("--name", type=str, required=True)
@click.argument(
    "probe_suite_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.argument(
    "loss_dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "output_filepath", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.option("--rio", is_flag=True, default=False)
def main_click(name, probe_suite_path, loss_dataset_path, output_filepath, rio):
    main(
        name=name,
        probe_suite_path=probe_suite_path,
        loss_dataset_path=loss_dataset_path,
        output_path=output_filepath,
        rio=rio,
    )


if __name__ == "__main__":
    main_click()

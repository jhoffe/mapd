import os
import random

import click
import matplotlib.pyplot as plt

from src.visualization.utils.plot_utils import (
    load_loss_dataset,
    load_probe_suite,
    plot_styles,
)


def probe_accuracy_plot(
    name: str, probe_suite_path: str, loss_dataset_path: str, output_path: str
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

    suite_names = {
        "typical": "Typical",
        "atypical": "Atypical",
        "random_outputs": "Random outputs",
        "random_inputs_outputs": "Random inputs and outputs",
        "corrupted": "Corrupted",
    }

    for suite_attr, suite_name in suite_names.items():
        indices = [idx for idx, suite in suite_indices.items() if suite == suite_attr]
        random.shuffle(indices)
        train_indices = indices[:250]
        val_indices = indices[250:]
        df.loc[df["sample_index"].isin(train_indices), "suite"] = suite_name
        df.loc[df["sample_index"].isin(val_indices), "suite"] = suite_name + " [Val]"

    df["suite"] = df["suite"].fillna("Train")
    suites = sorted(df["suite"].unique())

    df["prediction"] = df["y"] == df["y_hat"]

    val_df = df[df["stage"] == "val"]
    val_df = val_df.groupby(["epoch", "suite"]).agg({"prediction": "mean"})
    val_df.reset_index(inplace=True)
    val_df["prediction"] = val_df["prediction"] * 100

    train_df = df[df["stage"] == "train"]
    train_df = train_df.groupby(["epoch", "suite"]).agg({"prediction": "mean"})
    train_df.reset_index(inplace=True)
    train_df["prediction"] = train_df["prediction"] * 100

    suites = sorted(train_df["suite"].unique())

    # Plot
    line_styles, marker_list, marker_colors, plot_titles = plot_styles()

    plt.figure(figsize=(10, 6))
    plt.title(f"Probe Suite Accuracy for {plot_titles[name]}")
    for i, suite in enumerate(suites):
        if "Train" in suite:
            plt.plot(
                train_df[train_df["suite"] == suite]["epoch"],
                train_df[train_df["suite"] == suite]["prediction"],
                label=suite,
                alpha=0.75,
                linewidth=0.5,
                linestyle=line_styles[i % len(line_styles)],
                marker=marker_list[i % len(marker_list)],
                markersize=3,
                color=marker_colors[i % len(marker_colors)],
            )
        else:
            plt.plot(
                val_df[val_df["suite"] == suite]["epoch"],
                val_df[val_df["suite"] == suite]["prediction"],
                label=suite,
                alpha=0.75,
                linewidth=0.5,
                linestyle=line_styles[i % len(line_styles)],
                marker=marker_list[i % len(marker_list)],
                markersize=3,
                color=marker_colors[i % len(marker_colors)],
            )
    plt.legend(loc="lower right", fontsize="small")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    figure_path = os.path.join(output_path, name)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    plt.savefig(os.path.join(figure_path, f"{name}_probe_suite_accuracy.png"))


def main(name, probe_suite_path, loss_dataset_path, output_filepath):
    probe_accuracy_plot(name, probe_suite_path, loss_dataset_path, output_filepath)


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
def main_click(name, probe_suite_path, loss_dataset_path, output_filepath):
    main(name, probe_suite_path, loss_dataset_path, output_filepath)


if __name__ == "__main__":
    main_click()

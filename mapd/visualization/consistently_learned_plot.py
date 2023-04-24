import os
import random

import click
import matplotlib.pyplot as plt

from src.visualization.utils.plot_utils import (
    load_loss_dataset,
    load_probe_suite,
    plot_dicts,
    plot_styles,
)


def consistently_learned_plot(
    name: str, probe_suite_path: str, loss_dataset_path: str, output_path: str
) -> None:
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

    df["prediction"] = df["y"] == df["y_hat"]

    val_df = df[df["stage"] == "val"]
    train_df = df[df["stage"] == "train"]

    consistently_learned, learned, suite_names = plot_dicts()

    temp_train = train_df[train_df["suite"] == "Train"]

    learned["Train"] = set(
        temp_train.groupby(["epoch"])
        .get_group(max_epoch - 1)["sample_index"]
        .values[temp_train.groupby(["epoch"]).get_group(max_epoch - 1)["prediction"]]
    )
    for suite in suite_names:
        suite_group = val_df.groupby(["suite"]).get_group(suite)
        learned[suite] = set(
            suite_group.groupby(["epoch"])
            .get_group(max_epoch - 1)["sample_index"]
            .values[
                suite_group.groupby(["epoch"]).get_group(max_epoch - 1)["prediction"]
            ]
        )

    suite_size = len(probe_suite.index_to_suite) / len(suite_names)
    train_size = num_train_samples

    for epoch in reversed(range(max_epoch)):
        epoch_train_group = temp_train.groupby(["epoch"]).get_group(epoch)
        epoch_train_group = epoch_train_group[
            epoch_train_group["sample_index"].isin(learned["Train"])
        ]
        consistently_learned["Train"].insert(0, len(learned["Train"]) / train_size)
        learned["Train"] = set(
            epoch_train_group["sample_index"][epoch_train_group["prediction"]].values
        )
        epoch_val_group = val_df.groupby(["epoch"]).get_group(epoch)
        for suite in suite_names:
            suite_group = epoch_val_group.groupby(["suite"]).get_group(suite)
            suite_group = suite_group[suite_group["sample_index"].isin(learned[suite])]
            learned[suite] = set(
                suite_group["sample_index"][suite_group["prediction"]].values
            )
            consistently_learned[suite].insert(0, len(learned[suite]) / suite_size)

    suites = consistently_learned.keys()

    line_styles, marker_list, marker_colors, plot_titles = plot_styles()

    plt.figure(figsize=(10, 6))
    plt.title(f"Percent Consistently Learned for {plot_titles[name]}")
    for i, suite in enumerate(suites):
        plt.plot(
            consistently_learned[suite],
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
    plt.ylabel("Percent Samples Learned (%)")

    figure_path = os.path.join(output_path, name)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    plt.savefig(os.path.join(figure_path, f"{name}_consistently_learned_accuracy.png"))


def main(name, probe_suite_path, loss_dataset_path, output_filepath):
    consistently_learned_plot(
        name, probe_suite_path, loss_dataset_path, output_filepath
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
def main_click(name, probe_suite_path, loss_dataset_path, output_filepath):
    main(name, probe_suite_path, loss_dataset_path, output_filepath)


if __name__ == "__main__":
    main_click()

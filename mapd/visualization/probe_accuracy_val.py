import os
import random

import click
import matplotlib.pyplot as plt

from src.visualization.utils.plot_utils import load_loss_dataset, load_probe_suite


def plot_probe_accuracies(
    name: str, probe_suite_path: str, loss_dataset_path: str, output_path: str
) -> None:
    """Plot the probe accuracies for the given probe suite and loss dataset"""
    ps = load_probe_suite(probe_suite_path)
    ld_df = load_loss_dataset(loss_dataset_path)

    suite_indices = ps.index_to_suite
    suite_names = set(ps.index_to_suite.values())
    probe_count = len(suite_indices)

    ld_df["epoch"] = ld_df["epoch"].astype(int)

    # Split the data
    train_df = ld_df[ld_df["stage"] == "train"]
    probe_df = ld_df[ld_df["stage"] == "val"]

    del ld_df

    print(f"Train ({train_df.shape}):")
    print(f"Probes ({probe_df.shape}):")

    # Remove the probe samples from the sanity check
    sanity_check_count = len(probe_df[probe_df["epoch"] == 0]) - probe_count
    if sanity_check_count > 0:
        indices_to_drop = probe_df[probe_df["epoch"] == 0].index[:sanity_check_count]
        probe_df = probe_df.drop(indices_to_drop)

    print(f"Probes ({probe_df.shape}):")

    # Add suite column to validation dataframe
    probe_df["suite"] = probe_df["sample_index"].map(suite_indices)

    val_ratio = 0.5
    # Take the suite indices and split them into train and validation
    val_or_train_indices = {}
    for suite_name in suite_names:
        idx = [i for i, s in suite_indices.items() if s == suite_name]
        random.shuffle(idx)

        val_count = int(len(idx) * val_ratio)
        val_indices = idx[:val_count]
        train_indices = idx[val_count:]

        for i in val_indices:
            val_or_train_indices[i] = True
        for i in train_indices:
            val_or_train_indices[i] = False

    # Remove the suite indices from the train df
    train_df = train_df[~train_df["sample_index"].isin(suite_indices.keys())]
    # Add correct to train
    train_df["correct"] = (train_df["y"] == train_df["y_hat"]).astype(int)

    probe_df["val"] = probe_df["sample_index"].map(val_or_train_indices)

    # Add correct prediction column to validation dataframe
    probe_df["correct"] = (probe_df["y"] == probe_df["y_hat"]).astype(int)

    # Get the mean accuracy for each suite for each epoch for probes
    epoch_suite_val = probe_df.groupby(["epoch", "suite", "val"]).agg(
        {"correct": "mean"}
    )
    # Get the mean accuracy for each suite for each epoch for train
    epoch_train = train_df.groupby(["epoch"]).agg({"correct": "mean"})

    # Plot the mean accuracy for each suite for each epoch and suite
    # and use different markers for each suite and train
    possible_markers = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "s",
        "p",
        "P",
        "*",
        "x",
        "+",
        "X",
        "D",
        "d",
    ]

    fig, ax = plt.subplots(figsize=(16, 10))
    i = 0
    for suite_name in suite_names:
        suite_df = epoch_suite_val.loc[(slice(None), suite_name, slice(None)), :]
        suite_df = suite_df.reset_index()

        probe_train_df = suite_df[suite_df["val"]]
        probe_val_df = suite_df[suite_df["val"]]

        suite_name = suite_name.replace("_", " ").capitalize()
        ax.plot(
            probe_train_df["epoch"],
            probe_train_df["correct"],
            label=f"{suite_name}",
            alpha=0.75,
            linewidth=0.5,
            markersize=3,
            marker=possible_markers[i],
        )
        ax.plot(
            probe_val_df["epoch"],
            probe_val_df["correct"],
            label=f"{suite_name} [Val]",
            alpha=0.75,
            linewidth=0.5,
            markersize=3,
            marker=possible_markers[i],
            linestyle=":",
        )
        i += 1

    # Add train accuracy
    ax.plot(
        epoch_train.index,
        epoch_train["correct"],
        label="Train",
        linewidth=0.5,
        alpha=0.75,
        markersize=3,
        marker=possible_markers[i],
        linestyle="--",
    )

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=False,
        ncol=5,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title(f"Mean Accuracy for each suite for each epoch ({name})")

    filename = name.replace(" ", "_").lower()

    figure_path = os.path.join(output_path, name)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    fig.savefig(os.path.join(figure_path, f"suite_accuracy_{filename}.png"))


def main(name, probe_suite_path, loss_dataset_path, output_filepath):
    plot_probe_accuracies(name, probe_suite_path, loss_dataset_path, output_filepath)


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
    main()

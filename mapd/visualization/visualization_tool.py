import os
import random
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from tqdm import tqdm

from mapd.probes.probe_suite_generator import ProbeSuiteDataset


class MAPDVisualizationTool:
    def __init__(
        self,
        mapd_loss_dataset_path: str,
        probe_suite_dataset: ProbeSuiteDataset,
    ) -> None:
        self.data = ds.dataset(
            mapd_loss_dataset_path,
            partitioning=ds.partitioning(
                pa.schema([("epoch", pa.int64()), ("stage", pa.string())]),
                flavor="filename",
            ),
            format="parquet",
        )

        self.probe_suite_dataset = probe_suite_dataset
        self._suite_indices = self.probe_suite_dataset.index_to_suite

        self._prepare_val_data()
        self._prepare_train_data()
        self._max_epoch = self.train_data["epoch"].max() + 1

        # all suites
        self.all_suites = sorted(self._val_suites) + sorted(self._train_suites)

        # prepare plot styles
        self._prepare_plot_styles()

    def probe_accuracy_plot(
        self,
        show: bool = True,
        save: Optional[bool] = None,
        save_path: Optional[str] = None,
        plot_config: Optional[Dict] = None,
    ) -> None:
        """
        Plot probe accuracy for each suite over epochs.

        Args:
            show (bool, optional): Whether to show the plot. Defaults to True.
            save (Optional[bool], optional): Whether to save the plot. Defaults to None.
            save_path (Optional[str], optional): Path to save the plot.
                Defaults to None.
            plot_config (Optional[Dict], optional): Plot configuration.
                Defaults to None.

        Raises:
            AssertionError: If save is not None and save_path is None.
        """
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert (
                save is True
            ), "Remember to set save to True if you want to save the plot!"

        self.val_df = self.val_data.groupby(["epoch", "suite"]).agg(
            {"prediction": "mean"}
        )
        self.val_df.reset_index(inplace=True)
        self.val_df["prediction"] = self.val_df["prediction"] * 100

        self.train_df = self.train_data.groupby(["epoch", "suite"]).agg(
            {"prediction": "mean"}
        )
        self.train_df.reset_index(inplace=True)
        self.train_df["prediction"] = self.train_df["prediction"] * 100

        plt.figure()
        plt.title(
            f"Probe Suite Accuracy for {len(self.all_suites)} Suites"
        ) if plot_config is None else plt.title(plot_config["title"])
        for i, suite in enumerate(self.all_suites):
            if "Other" in suite:
                plt.plot(
                    self.train_df[self.train_df["suite"] == suite]["epoch"],
                    self.train_df[self.train_df["suite"] == suite]["prediction"],
                    label=suite,
                    alpha=0.75,
                    linewidth=0.5,
                    linestyle=self._line_styles[0],
                    marker=self._marker_list[0],
                    markersize=3,
                    color=self._marker_colors[-1],
                )
            else:
                plt.plot(
                    self.val_df[self.val_df["suite"] == suite]["epoch"],
                    self.val_df[self.val_df["suite"] == suite]["prediction"],
                    label=suite,
                    alpha=0.75,
                    linewidth=0.5,
                    linestyle=self._line_styles[i % 2],
                    marker=self._marker_list[i % 2],
                    markersize=3,
                    color=self._marker_colors[i // 2],
                )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        # place legend below plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        if show:
            plt.show()
        if save:
            plt.savefig(save_path, bbox_inches="tight")

        plt.close()

    def consistently_learned_plot(
        self,
        show: bool = True,
        save: Optional[bool] = None,
        save_path: Optional[str] = None,
        plot_config: Optional[Dict] = None,
    ) -> None:
        """
        Plot consistently learned ratios for each suite over epochs.

        Args:
            show (bool, optional): Whether to show the plot. Defaults to True.
            save (Optional[bool], optional): Whether to save the plot. Defaults to None.
            save_path (Optional[str], optional): Path to save the plot.
                Defaults to None.
            plot_config (Optional[Dict], optional): Plot configuration.
                Defaults to None.

        Raises:
            AssertionError: If save is not None and save_path is None.
        """
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert (
                save is True
            ), "Remember to set save to True if you want to save the plot!"

        self._plot_dicts()

        self._learned["Other"] = set(
            self.train_data.groupby(["epoch"])
            .get_group(self._max_epoch - 1)["sample_index"]
            .values[
                self.train_data.groupby(["epoch"]).get_group(self._max_epoch - 1)[
                    "prediction"
                ]
            ]
        )

        for val_suite in self._val_suites:
            suite_group = self.val_data.groupby(["suite"]).get_group(val_suite)
            self._learned[val_suite] = set(
                suite_group.groupby(["epoch"])
                .get_group(self._max_epoch - 1)["sample_index"]
                .values[
                    suite_group.groupby(["epoch"]).get_group(self._max_epoch - 1)[
                        "prediction"
                    ]
                ]
            )

        suite_size = len(self._suite_indices) / len(self._val_suites)
        train_size = len(self.train_data["sample_index"].unique())

        print("Computing consistently learned ratios...")
        for epoch in tqdm(
            reversed(range(self._max_epoch)), desc="Epochs", total=self._max_epoch
        ):
            epoch_train_group = self.train_data.groupby(["epoch"]).get_group(epoch)
            epoch_train_group = epoch_train_group[
                epoch_train_group["sample_index"].isin(self._learned["Other"])
            ]
            self._ratios["Other"].insert(
                0, 100 * len(self._learned["Other"]) / train_size
            )
            self._learned["Other"] = set(
                epoch_train_group["sample_index"][
                    epoch_train_group["prediction"]
                ].values
            )
            epoch_val_group = self.val_data.groupby(["epoch"]).get_group(epoch)

            for val_suite in self._val_suites:
                val_suite_group = epoch_val_group.groupby(["suite"]).get_group(
                    val_suite
                )
                val_suite_group = val_suite_group[
                    val_suite_group["sample_index"].isin(self._learned[val_suite])
                ]
                self._learned[val_suite] = set(
                    val_suite_group["sample_index"][
                        val_suite_group["prediction"]
                    ].values
                )
                self._ratios[val_suite].insert(
                    0, 100 * len(self._learned[val_suite]) / suite_size
                )

        plt.figure(figsize=(10, 5))
        plt.title(
            f"Consistently Learned Samples for {len(self.all_suites)} Suites"
        ) if plot_config is None else plt.title(plot_config["title"])
        for i, suite in enumerate(self.all_suites):
            plt.plot(
                self._ratios[suite],
                label=suite,
                alpha=0.75,
                linewidth=0.5,
                linestyle=self._line_styles[i % 2],
                marker=self._marker_list[i % 2],
                markersize=3,
                color=self._marker_colors[i // 2],
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.xlabel("Epoch")
        plt.ylabel("Ratio of Consistently Learned Samples (%)")

        if show:
            plt.show()

        if save:
            plt.savefig(save_path, bbox_inches="tight")

        plt.close()

    def first_learned_plot(
        self,
        show: bool = True,
        save: Optional[bool] = None,
        save_path: Optional[str] = None,
        plot_config: Optional[Dict] = None,
    ) -> None:
        """
        Plot first learned ratios for each suite over epochs.

        Args:
            show (bool, optional): Whether to show the plot. Defaults to True.
            save (Optional[bool], optional): Whether to save the plot. Defaults to None.
            save_path (Optional[str], optional): Path to save the plot.
                Defaults to None.
            plot_config (Optional[Dict], optional): Plot configuration.
                Defaults to None.

        Raises:
            AssertionError: If save is not None and save_path is None.
        """
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert (
                save is True
            ), "Remember to set save to True if you want to save the plot!"

        self._plot_dicts()

        train_size = len(self.train_data["sample_index"].unique())

        for epoch in tqdm(range(self._max_epoch), desc="Epochs", total=self._max_epoch):
            epoch_train_group = self.train_data.groupby(["epoch"]).get_group(epoch)
            self._learned["Other"].update(
                epoch_train_group["sample_index"][
                    epoch_train_group["prediction"]
                ].values
            )
            self._ratios["Other"].append(100 * len(self._learned["Other"]) / train_size)
            epoch_val_group = self.val_data.groupby(["epoch"]).get_group(epoch)
            for val_suite in self._val_suites:
                val_suite_group = epoch_val_group.groupby(["suite"]).get_group(
                    val_suite
                )
                self._learned[val_suite].update(
                    val_suite_group["sample_index"][
                        val_suite_group["prediction"]
                    ].values
                )
                self._ratios[val_suite].append(
                    100 * len(self._learned[val_suite]) / len(val_suite_group)
                )

        plt.figure(figsize=(10, 5))
        plt.title(
            f"First Learned Samples for {len(self.all_suites)} Suites"
        ) if plot_config is None else plt.title(plot_config["title"])
        for i, suite in enumerate(self.all_suites):
            plt.plot(
                self._ratios[suite],
                label=suite,
                alpha=0.75,
                linewidth=0.5,
                linestyle=self._line_styles[i % 2],
                marker=self._marker_list[i % 2],
                markersize=3,
                color=self._marker_colors[i // 2],
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.xlabel("Epoch")
        plt.ylabel("Ratio of First Learned Samples (%)")

        if show:
            plt.show()

        if save:
            plt.savefig(save_path, bbox_inches="tight")

        plt.close()

    def loss_curve_plot(
        self,
        show: bool = True,
        save: Optional[bool] = None,
        save_path: Optional[str] = None,
        plot_config: Optional[Dict] = None,
    ) -> None:
        """
        Plot loss curve for each suite over epochs.

        Args:
            show (bool, optional): Whether to show the plot. Defaults to True.
            save (Optional[bool], optional): Whether to save the plot. Defaults to None.
            save_path (Optional[str], optional): Path to save the plot.
                Defaults to None.
            plot_config (Optional[Dict], optional): Plot configuration.
                Defaults to None.

        Raises:
            AssertionError: If save is not None and save_path is None.
        """
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert (
                save is True
            ), "Remember to set save to True if you want to save the plot!"
        lcp_df = self.val_data.sort_values(by=["epoch"])
        plt.figure(figsize=(10, 5))
        plt.title(
            f"Loss Curve for {len(self.all_suites)} Suites"
        ) if plot_config is None else plt.title(plot_config["title"])

        i = 0
        for suite in self._val_suites:
            if "[Val]" not in suite:
                indices = lcp_df["sample_index"][lcp_df["suite"] == suite].unique()
                for idx in indices:
                    plt.plot(
                        lcp_df["epoch"][lcp_df["sample_index"] == idx],
                        lcp_df["loss"][lcp_df["sample_index"] == idx],
                        alpha=0.25,
                        linewidth=0.1,
                        color=self._marker_colors[i],
                    )

                # plot aggregated loss for each suite (mean over all samples)
                plt.plot(
                    np.arange(self._max_epoch),
                    lcp_df.loc[lcp_df["suite"] == suite]
                    .groupby(["epoch"])
                    .agg({"loss": "mean"}),
                    label=suite,
                    linewidth=1,
                    color=self._marker_colors[i],
                    zorder=2,
                )

                i += 1

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        # plt.ylim(0, 14)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        if show:
            plt.show()

        if save:
            plt.savefig(save_path, bbox_inches="tight")

        plt.close()

    def all_plots(self, show: bool = True, save_path: Optional[str] = None) -> None:
        """
        Plot all plots.
        """
        # self.violin_loss_plot()
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        loss_curve_save_path = (
            os.path.join(save_path, "loss_curve.png") if save_path is not None else None
        )
        self.loss_curve_plot(
            show, save=save_path is not None, save_path=loss_curve_save_path
        )
        probe_accuracy_save_path = (
            os.path.join(save_path, "probe_accuracy.png")
            if save_path is not None
            else None
        )
        self.probe_accuracy_plot(
            show, save=save_path is not None, save_path=probe_accuracy_save_path
        )
        first_learned_save_path = (
            os.path.join(save_path, "first_learned.png")
            if save_path is not None
            else None
        )
        self.first_learned_plot(
            show, save=save_path is not None, save_path=first_learned_save_path
        )
        consistently_learned_save_path = (
            os.path.join(save_path, "consistently_learned.png")
            if save_path is not None
            else None
        )
        self.consistently_learned_plot(
            show, save=save_path is not None, save_path=consistently_learned_save_path
        )

    def violin_loss_plot(
        self,
        show: bool = True,
        save: Optional[bool] = None,
        save_path: Optional[str] = None,
        plot_config: Optional[Dict] = None,
    ) -> None:
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert (
                save is True
            ), "Remember to set save to True if you want to save the plot!"
        pass

    def _suite_to_name(self) -> Dict[str, str]:
        """
        Convert suite names to more readable names.

        Returns:
            Dict[str, str]: Dictionary mapping suite names to more readable names.
        """
        self.suites = list(set(self._suite_indices.values()))
        temp_dict = {}
        for suite in self.suites:
            test = suite.split("_")

            # capitalize first word of each suite
            test = [word.capitalize() for word in test]
            # join words with space
            test = " ".join(test)
            temp_dict[suite] = test
        return temp_dict

    def _prepare_val_data(self):
        """
        Prepare validation data.
        """
        self.val_data = (
            self.data.filter(ds.field("stage") == "val").to_table().to_pandas()
        )
        self.val_data["epoch"] = self.val_data["epoch"].astype(int)
        self.val_indices = self.val_data["sample_index"].unique()

        self.suite_to_name = self._suite_to_name()

        for suite_attr, suite_name in self.suite_to_name.items():
            indices = [
                idx for idx, suite in self._suite_indices.items() if suite == suite_attr
            ]
            random.shuffle(indices)
            train_indices = indices[:250]
            val_indices = indices[250:]
            self.val_data.loc[
                self.val_data["sample_index"].isin(train_indices), "suite"
            ] = suite_name
            self.val_data.loc[
                self.val_data["sample_index"].isin(val_indices), "suite"
            ] = (suite_name + " [Val]")

        self.val_data["prediction"] = self.val_data["y"] == self.val_data["y_hat"]

        self._val_suites = self.val_data["suite"].unique()

    def _prepare_train_data(self):
        """
        Prepare training data.
        """
        self.train_data = (
            self.data.filter(ds.field("stage") == "train").to_table().to_pandas()
        )
        self.train_data = self.train_data[
            ~self.train_data["sample_index"].isin(self.val_indices)
        ]
        self.train_data["epoch"] = self.train_data["epoch"].astype(int)

        self.train_data["suite"] = "Other"
        self.train_data["prediction"] = self.train_data["y"] == self.train_data["y_hat"]

        self._train_suites = self.train_data["suite"].unique()

    def _prepare_plot_styles(self):
        """
        Prepare plot styles.
        """
        self._line_styles = ["solid", "dashed", "dashdot", "dotted"]

        self._marker_list = ["o", "X"]

        self._marker_colors = [
            "tab:red",
            "tab:blue",
            "tab:purple",
            "tab:orange",
            "tab:green",
            "tab:pink",
            "tab:olive",
            "tab:brown",
            "tab:cyan",
        ]

    def _plot_dicts(self):
        """
        Prepare plot dicts.
        """
        # make dict with one list for each suite
        self._ratios = {}
        for suite in self.all_suites:
            self._ratios[suite] = []

        self._learned = {}
        for suite in self.all_suites:
            self._learned[suite] = set([])

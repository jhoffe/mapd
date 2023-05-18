import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from typing import List, Dict, Tuple, Union, Optional
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
        self.suite_indices = self.probe_suite_dataset.index_to_suite

        self.val_data = (
            self.data.filter(ds.field("stage") == "val").to_table().to_pandas()
        )
        self.val_data["epoch"] = self.val_data["epoch"].astype(int)
        self.val_indices = self.val_data["sample_index"].unique()

        self.train_data = (
            self.data.filter(ds.field("stage") == "train").to_table().to_pandas()
        )
        self.train_data = self.train_data[
            ~self.train_data["sample_index"].isin(self.val_indices)
        ]
        self.train_data["epoch"] = self.train_data["epoch"].astype(int)

        # add probe suite to train called "Train"
        self.train_data["probe_suite"] = "Other"

        self.suite_to_name = self._suite_to_name()

        for suite_attr, suite_name in self.suite_to_name.items():
            indices = [
                idx for idx, suite in self.suite_indices.items() if suite == suite_attr
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
        self.val_data = self.val_data.groupby(["epoch", "suite"]).agg(
            {"prediction": "mean"}
        )
        self.val_data.reset_index(inplace=True)
        self.val_data["prediction"] = self.val_data["prediction"] * 100

        self.train_data["prediction"] = self.train_data["y"] == self.train_data["y_hat"]
        self.train_data = self.train_data.groupby(["epoch", "suite"]).agg(
            {"prediction": "mean"}
        )
        self.train_data.reset_index(inplace=True)
        self.train_data["prediction"] = self.train_data["prediction"] * 100

        # all suites
        self._val_suites = self.val_data["suite"].unique()
        self._train_suites = self.train_data["suite"].unique()
        self.all_suites = sorted(self._val_suites) + sorted(self._train_suites)

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
            "tab:gray",
        ]

    def _suite_to_name(self) -> Dict[str, str]:
        self.suites = list(set(self.suite_indices.values()))
        temp_dict = {}
        for suite in self.suites:
            test = suite.split("_")

            # capitalize first word of each suite
            test = [word.capitalize() for word in test]
            # join words with space
            test = " ".join(test)
            temp_dict[suite] = test
        return temp_dict

    def probe_accuracy_plot(
        self,
        show: bool = True,
        save: Optional[bool] = None,
        save_path: Optional[os.PathLike] = None,
        plot_config: Optional[Dict] = None,
    ) -> None:
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert (
                save is True
            ), "Remember to set save to True if you want to save the plot!"

        plt.title(
            f"Probe Suite Accuracy for {len(self.all_suites)} Suites"
        ) if plot_config is None else plt.title(plot_config["title"])
        for i, suite in enumerate(self.all_suites):
            if "Other" in suite:
                plt.plot(
                    self.train_data[self.train_data["suite"] == suite]["epoch"],
                    self.train_data[self.train_data["suite"] == suite]["prediction"],
                    label=suite,
                    alpha=0.75,
                    linewidth=0.5,
                    linestyle=self._line_styles[0],
                    marker=self._marker_list[0],
                    markersize=3,
                    color=self._marker_colors[0],
                )
            else:
                plt.plot(
                    self.val_data[self.val_data["suite"] == suite]["epoch"],
                    self.val_data[self.val_data["suite"] == suite]["prediction"],
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

    def consistently_learned_plot(
        self,
        show: bool = True,
        save: Optional[bool] = None,
        save_path: Optional[os.PathLike] = None,
        plot_config: Optional[Dict] = None,
    ) -> None:
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert (
                save is True
            ), "Remember to set save to True if you want to save the plot!"
        pass

    def first_learned_plot(
        self,
        show: bool = True,
        save: Optional[bool] = None,
        save_path: Optional[os.PathLike] = None,
        plot_config: Optional[Dict] = None,
    ) -> None:
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert (
                save is True
            ), "Remember to set save to True if you want to save the plot!"
        pass

    def loss_curve_plot(
        self,
        show: bool = True,
        save: Optional[bool] = None,
        save_path: Optional[os.PathLike] = None,
        plot_config: Optional[Dict] = None,
    ) -> None:
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert (
                save is True
            ), "Remember to set save to True if you want to save the plot!"
        pass

    def violin_loss_plot(
        self,
        show: bool = True,
        save: Optional[bool] = None,
        save_path: Optional[os.PathLike] = None,
        plot_config: Optional[Dict] = None,
    ) -> None:
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert (
                save is True
            ), "Remember to set save to True if you want to save the plot!"
        pass

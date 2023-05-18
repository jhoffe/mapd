import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Tuple, Union, Optional


class MAPDVisualizationTool():
    def __init__(self, mapd_loss_dataset_path: os.PathLike, probe_suite_dataset) -> None:
        self.data = ds.dataset(
            "probes_data", 
            partitioning=ds.partitioning(
                pa.schema(
                    [("epoch", pa.int64()), 
                    ("stage", pa.string())]
                    ), 
                flavor="filename"
            ), 
            format="parquet"    
        )


        self.val_data = self.data.filter(ds.field("stage") == "val").to_table().to_pandas()
        self.val_data["epoch"] = self.val_data["epoch"].astype(int)
        self.val_indices = self.val_data["sample_index"].unique()

        self.train_data = self.data.filter(ds.field("stage") == "train").to_table().to_pandas()
        self.train_data = self.train_data[~self.train_data["sample_index"].isin(self.val_indices)]
        self.train_data["epoch"] = self.train_data["epoch"].astype(int)

    
        self._line_styles = ["solid", "dashed", "dashdot", "dotted"]

        self._marker_list = ["o", "*"]

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

    def probe_accuracy_plot(
            self,
            show: bool = True,
            save: Optional[bool] = None,
            save_path: Optional[os.PathLike] = None,
            plot_config: Optional[Dict] = None
        ) -> None:
        
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert save is True, "Remember to set save to True if you want to save the plot!"
        

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

    def consistently_learned_plot(
            self,
            show: bool = True,
            save: Optional[bool] = None,
            save_path: Optional[os.PathLike] = None,
            plot_config: Optional[Dict] = None
        ) -> None:
        
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert save is True, "Remember to set save to True if you want to save the plot!"
        pass

    def first_learned_plot(
            self,
            show: bool = True,
            save: Optional[bool] = None,
            save_path: Optional[os.PathLike] = None,
            plot_config: Optional[Dict] = None
        ) -> None:
        
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert save is True, "Remember to set save to True if you want to save the plot!"
        pass

    def loss_curve_plot(
            self,
            show: bool = True,
            save: Optional[bool] = None,
            save_path: Optional[os.PathLike] = None,
            plot_config: Optional[Dict] = None
        ) -> None:
        
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert save is True, "Remember to set save to True if you want to save the plot!"
        pass

    def violin_loss_plot(
            self,
            show: bool = True,
            save: Optional[bool] = None,
            save_path: Optional[os.PathLike] = None,
            plot_config: Optional[Dict] = None
        ) -> None:
        
        # check that save is not None if save_path is not None
        if save_path is not None:
            assert save is True, "Remember to set save to True if you want to save the plot!"
        pass
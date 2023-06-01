from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from mapd.data.make_probe_suites import ProbeSuiteGenerator


class LossDataset:
    dataset_path: str
    dataset: pq.ParquetDataset
    df: Optional[pd.DataFrame]
    probe_suite: Optional[ProbeSuiteGenerator]

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = pq.ParquetDataset(dataset_path)

    def load(self) -> "LossDataset":
        self.df = self.dataset.read().to_pandas()

        return self

    def load_probe_suite(self, probe_suite_path: str) -> "LossDataset":
        self.probe_suite = torch.load(probe_suite_path)

        return self

    def to_sklearn_train_matrix(self) -> tuple[np.array, np.array]:
        assert self.df is not None
        assert self.probe_suite is not None

        probe_suite_indices = self.probe_suite.index_to_suite

        df = self.df[self.df["stage"] == "val"]
        df = df[df["sample_index"].isin(probe_suite_indices.keys())].copy()

        # Convert to int from categorical
        df["epoch"] = df["epoch"].astype(int)

        probe_count = len(probe_suite_indices)
        # Remove the probe samples from the sanity check
        sanity_check_count = len(df[df["epoch"] == 0]) - probe_count
        if sanity_check_count > 0:
            indices_to_drop = df[df["epoch"] == 0].index[:sanity_check_count]
            df = df.drop(indices_to_drop)

        df.sort_values(["sample_index", "epoch"], inplace=True)
        sample_groups = df.groupby("sample_index", sort=False)

        losses = sample_groups["loss"].agg(list)
        losses_indices = losses.index

        X = np.array(losses.to_list())

        ps_indices_to_int = {
            key: idx for idx, key in enumerate(set(probe_suite_indices.values()))
        }
        y = np.array(
            [ps_indices_to_int[probe_suite_indices[idx]] for idx in losses_indices]
        )

        return X, y

    def to_sklearn_predict_matrix(self):
        assert self.df is not None
        assert self.probe_suite is not None

        probe_suite_indices = self.probe_suite.index_to_suite

        df = self.df[self.df["stage"] == "train"]
        df = df[~df["sample_index"].isin(probe_suite_indices.keys())].copy()

        # Convert to int from categorical
        df["epoch"] = df["epoch"].astype(int)
        df.sort_values(["sample_index", "epoch"], inplace=True)
        sample_groups = df.groupby("sample_index", sort=False)

        losses = sample_groups["loss"].agg(list)
        original_classes = sample_groups["y"].first().values
        sample_indices = losses.index

        X = np.array(losses.to_list())

        names = set(self.probe_suite.index_to_suite.values())
        index_to_label_name = {idx: label_name for idx, label_name in enumerate(names)}

        return X, original_classes, sample_indices, index_to_label_name

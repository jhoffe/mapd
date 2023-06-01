import os
from collections import defaultdict
from typing import Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from mapd.probes.probe_suite_generator import ProbeSuiteDataset


def create_sklearn_train_matrix(
    dataset_path: Union[str, os.PathLike],
    probe_suite_ds: ProbeSuiteDataset,
    epoch_range: Optional[Tuple[int, int]] = None,
):
    """
    Create a matrix of losses for each sample in the probes-dataset.

    Args:
        dataset_path(str, os.PathLike): Path to the probes-dataset.
        probe_suite_ds(ProbeSuiteDataset): ProbeSuiteDataset object.
        epoch_range(Tuple[int, int]): Range of epochs to include in the matrix.

    Returns:
        Tuple[np.ndarray, list]: Tuple of X and y.
    """
    dataset = ds.dataset(
        dataset_path,
        partitioning=ds.partitioning(
            pa.schema([("epoch", pa.int64()), ("stage", pa.string())]),
            flavor="filename",
        ),
        format="parquet",
    )
    sample_index_to_loss = defaultdict(list)

    i = 0
    while True:
        if epoch_range is not None and (i < epoch_range[0] or i > epoch_range[1]):
            continue

        epoch_df = (
            dataset.filter((ds.field("epoch") == i) & (ds.field("stage") == "val"))
            .to_table()
            .to_pandas()
        )
        if epoch_df.empty:
            break

        for sample_index, loss_data in (
            epoch_df.groupby("sample_index").agg({"loss": "first"}).iterrows()
        ):
            loss = loss_data.values[0]

            sample_index_to_loss[sample_index].append(loss)

        i += 1

    sample_index_to_probe_suite = {
        idx: probe_suite_ds.index_to_suite[idx] for idx in sample_index_to_loss.keys()
    }

    X = np.array([losses for losses in sample_index_to_loss.values()])
    y = list(sample_index_to_probe_suite.values())

    return X, y


def create_sklearn_predict_matrix(
    dataset_path: Union[str, os.PathLike], epoch_range: Optional[Tuple[int, int]] = None
):
    """
    Create a matrix of losses for each sample in the training-dataset.

    Args:
        dataset_path(str, os.PathLike): Path to the probes-dataset.
        epoch_range(Tuple[int, int]): Range of epochs to include in the matrix.

    Returns:
        Tuple[np.ndarray, list]: Tuple of X and y.
    """
    dataset = ds.dataset(
        dataset_path,
        partitioning=ds.partitioning(
            pa.schema([("epoch", pa.int64()), ("stage", pa.string())]),
            flavor="filename",
        ),
        format="parquet",
    )
    sample_index_to_loss = defaultdict(list)

    i = 0
    while True:
        if epoch_range is not None and (i < epoch_range[0] or i > epoch_range[1]):
            continue
        epoch_df = (
            dataset.filter((ds.field("epoch") == i) & (ds.field("stage") == "train"))
            .to_table()
            .to_pandas()
        )
        if epoch_df.empty:
            break

        for sample_index, loss_data in (
            epoch_df.groupby("sample_index").agg({"loss": "first"}).iterrows()
        ):
            loss = loss_data.values[0]

            sample_index_to_loss[sample_index].append(loss)

        i += 1

    return np.array([losses for losses in sample_index_to_loss.values()]), list(
        sample_index_to_loss.keys()
    )

import os
from collections import defaultdict
from typing import Union, Dict, Any, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, is_classifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from mapd.probes.probe_suite_generator import ProbeSuiteDataset


def _create_sklearn_predict_matrix(dataset_path: Union[str, os.PathLike],
                                   epoch_range: Optional[Tuple[int, int]] = None):
    dataset = ds.dataset(dataset_path,
                         partitioning=ds.partitioning(pa.schema([("epoch", pa.int64()), ("stage", pa.string())]),
                                                      flavor="filename"), format="parquet")
    sample_index_to_loss = defaultdict(list)

    i = 0
    while True:
        if epoch_range is not None and (i < epoch_range[0] or i > epoch_range[1]):
            continue
        epoch_df = dataset.filter((ds.field("epoch") == i) & (ds.field("stage") == "train")).to_table().to_pandas()
        if epoch_df.empty:
            break

        for sample_index, loss_data in epoch_df.groupby("sample_index").agg({"loss": "first"}).iterrows():
            loss = loss_data.values[0]

            sample_index_to_loss[sample_index].append(loss)

        i += 1

    return np.array([losses for losses in sample_index_to_loss.values()]), list(sample_index_to_loss.keys())


def make_predictions(dataset_path: Union[str, os.PathLike], clf: BaseEstimator, label_encoder: LabelEncoder,
                     epoch_range: Optional[Tuple[int, int]] = None, n_jobs: int = 1):
    X, sample_indices = _create_sklearn_predict_matrix(dataset_path, epoch_range=epoch_range)

    batches = np.array_split(X, n_jobs)

    def _predict(X_predict):
        y_pred = clf.predict(X_predict)
        y_probas = clf.predict_proba(X_predict)
        return y_pred, np.max(y_probas, axis=1), label_encoder.inverse_transform(y_pred)

    results = Parallel(n_jobs=n_jobs)(delayed(_predict)(batch) for batch in tqdm(batches))

    confidence = np.concatenate([result[1] for result in results])
    labels = np.concatenate([result[2] for result in results])

    return dict(zip(sample_indices, zip(labels, confidence)))

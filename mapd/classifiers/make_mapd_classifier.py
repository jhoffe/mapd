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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, is_classifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.pipeline import make_pipeline

from mapd.probes.probe_suite_generator import ProbeSuiteGenerator


def _create_sklearn_train_matrix(dataset_path: Union[str, os.PathLike], probe_suite_ds: ProbeSuiteGenerator,
                                 epoch_range: Optional[Tuple[int, int]] = None):
    dataset = ds.dataset(dataset_path,
                         partitioning=ds.partitioning(pa.schema([("epoch", pa.int64()), ("stage", pa.string())]),
                                                      flavor="filename"), format="parquet")
    sample_index_to_loss = defaultdict(list)

    i = 0
    while True:
        epoch_df = dataset.filter((ds.field("epoch") == i) & (ds.field("stage") == "val")).to_table().to_pandas()
        if epoch_df.empty:
            break
        if epoch_range is not None and (i < epoch_range[0] or i > epoch_range[1]):
            break

        for sample_index, loss_data in epoch_df.groupby("sample_index").agg({"loss": "first"}).iterrows():
            loss = loss_data.values[0]

            sample_index_to_loss[sample_index].append(loss)

        i += 1

    sample_index_to_probe_suite = {idx: probe_suite_ds.index_to_suite[idx] for idx in sample_index_to_loss.keys()}

    X = np.array([losses for losses in sample_index_to_loss.values()])
    y = list(sample_index_to_probe_suite.values())

    return X, y


CLASSIFIERS = {
    "knn": lambda: KNeighborsClassifier(20),
    "svc": lambda: SVC(),
    "naive_bayes": lambda: GaussianNB(),
    "decision_tree": lambda: DecisionTreeClassifier(),
    "logistic_regression": lambda: LogisticRegression(max_iter=3000),
    "random_forest": lambda: RandomForestClassifier(),
    "ada_boost": lambda: AdaBoostClassifier(),
    "mlp": lambda: MLPClassifier((50, 25, 10), max_iter=1000),
    "xgboost": lambda: XGBClassifier(),
    "xgboost_rf": lambda: XGBRFClassifier(n_estimators=100),
}


def make_mapd_classifier(dataset_path: Union[str, os.PathLike], probe_suite_ds: ProbeSuiteGenerator,
                         clf: Union[str, BaseEstimator] = "xgboost", clf_kwargs: Optional[Dict[str, Any]] = None,
                         epoch_range: Optional[Tuple[int, int]] = None):
    if isinstance(clf, str):
        if clf not in CLASSIFIERS:
            raise ValueError(f"Invalid classifier name: {clf}")

        clf = CLASSIFIERS[clf]()

    if clf_kwargs is not None:
        clf.set_params(**clf_kwargs)

    X, y = _create_sklearn_train_matrix(dataset_path, probe_suite_ds, epoch_range=epoch_range)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    clf.fit(X, y)
    return clf, label_encoder

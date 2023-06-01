import os
from typing import Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from mapd.classifiers.utils.create_sklearn_matrix import \
    create_sklearn_predict_matrix


def make_predictions(
    dataset_path: Union[str, os.PathLike],
    clf: BaseEstimator,
    label_encoder: LabelEncoder,
    epoch_range: Optional[Tuple[int, int]] = None,
    n_jobs: int = 1,
):
    X, sample_indices = create_sklearn_predict_matrix(
        dataset_path, epoch_range=epoch_range
    )

    batches = np.array_split(X, n_jobs)

    def _predict(X_predict):
        y_pred = clf.predict(X_predict)
        y_probas = clf.predict_proba(X_predict)
        return y_pred, np.max(y_probas, axis=1), label_encoder.inverse_transform(y_pred)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_predict)(batch) for batch in tqdm(batches)
    )

    confidence = np.concatenate([result[1] for result in results])
    labels = np.concatenate([result[2] for result in results])

    return dict(zip(sample_indices, zip(labels, confidence)))

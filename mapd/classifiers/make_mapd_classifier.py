import os
from typing import Any, Dict, Optional, Tuple, Union

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from mapd.classifiers.utils.create_sklearn_matrix import \
    create_sklearn_train_matrix
from mapd.probes.probe_suite_generator import ProbeSuiteDataset

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


def make_mapd_classifier(
    dataset_path: Union[str, os.PathLike],
    probe_suite_ds: ProbeSuiteDataset,
    clf: Union[str, BaseEstimator] = "xgboost",
    clf_kwargs: Optional[Dict[str, Any]] = None,
    epoch_range: Optional[Tuple[int, int]] = None,
    plot_calibration_curves: bool = False,
):
    """
    Creates and trains a MAPD classifier that can predict
    the probe suite.

    Args:
        dataset_path(str, os.PathLike): Path to the dataset
        probe_suite_ds(ProbeSuiteDataset): Probe suite dataset
        clf(str, BaseEstimator): Classifier to use. If str, must be one of
            the following: knn, svc, naive_bayes, decision_tree,
            logistic_regression, random_forest, ada_boost, mlp, xgboost,
            xgboost_rf.
        clf_kwargs:

        epoch_range:

    Returns:

    """
    if isinstance(clf, str):
        if clf not in CLASSIFIERS:
            raise ValueError(f"Invalid classifier name: {clf}")

        clf = CLASSIFIERS[clf]()

    if clf_kwargs is not None:
        clf.set_params(**clf_kwargs)

    X, y = create_sklearn_train_matrix(
        dataset_path, probe_suite_ds, epoch_range=epoch_range
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    if plot_calibration_curves:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, stratify=y
        )
        clf.fit(X_train, y_train)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title("Calibration curve")

        from sklearn.calibration import CalibrationDisplay

        y_probas = clf.predict_proba(X_test)
        for i in range(y_probas.shape[1]):
            CalibrationDisplay.from_predictions(
                (y_test == i).astype(int),
                y_probas[:, i],
                ax=ax,
                name=label_encoder.classes_[i],
            )

        fig.tight_layout()
        plt.legend()

        return clf, label_encoder, fig
    else:
        clf.fit(X, y)

    return clf, label_encoder

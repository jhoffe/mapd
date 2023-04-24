import os

import click
import numpy as np
import pandas as pd
from lightning import seed_everything
from sklearn.base import BaseEstimator, is_classifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from src.data.loss_dataset import LossDataset


def cv_metadata_model(classifier: BaseEstimator, X: np.array, y: np.array) -> np.array:
    assert is_classifier(classifier)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y, shuffle=True
    )

    folds = cross_validate(classifier, X_train, y_train, cv=10, return_estimator=True)

    best_fold = np.argmax(folds["test_score"])

    best_est = folds["estimator"][best_fold]

    gen_score = np.mean(folds["test_score"])
    test_score = best_est.score(X_test, y_test)

    return gen_score, test_score


@click.command()
@click.argument(
    "loss_dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "probe_suite_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.option(
    "-r",
    "--results-dir",
    type=click.Path(dir_okay=False, file_okay=True),
    default="results/tables/",
)
@click.option("--seed", type=int, default=42)
def main(loss_dataset_path, probe_suite_path, results_dir, seed):
    seed_everything(seed)

    loss_dataset = LossDataset(loss_dataset_path)
    loss_dataset.load()
    loss_dataset.load_probe_suite(probe_suite_path)
    X, y = loss_dataset.to_sklearn_train_matrix()

    # Run all classifiers
    classifiers = {
        "KNN": KNeighborsClassifier(20),
        "SVM": SVC(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "MLP": MLPClassifier((50, 25, 10), max_iter=1000),
        "XGBoost": XGBClassifier(),
        "XGBoost RF": XGBRFClassifier(n_estimators=100),
    }

    results_table = []

    for name, classifier in classifiers.items():
        print(f"Running {name}")
        test_score, gen_score = cv_metadata_model(classifier, X, y)
        print(f"Gen score={np.mean(gen_score)}")
        print(f"Test score={np.mean(test_score)}")

        results_table.append([name, gen_score, test_score])

    results_table = pd.DataFrame(
        results_table, columns=["Classifier", "Gen Score", "Test Score"]
    )
    print(results_table)

    os.makedirs(results_dir, exist_ok=True)
    results_table.to_csv(os.path.join(results_dir, "results_table.csv"), index=False)
    results_table.to_latex(
        os.path.join(results_dir, "results_table.tex"), index=False, float_format="%.3f"
    )


if __name__ == "__main__":
    main()

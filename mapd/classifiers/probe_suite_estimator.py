import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator, is_classifier
from xgboost import XGBClassifier

from src.data.loss_dataset import LossDataset


def train_metadata_model(
    classifier: BaseEstimator, X: np.array, y: np.array
) -> BaseEstimator:
    assert is_classifier(classifier)

    classifier.fit(X, y)

    return classifier


@click.command()
@click.argument(
    "loss_dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "probe_suite_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.argument("output_path", type=click.Path(dir_okay=False, file_okay=True))
def main(loss_dataset_path, probe_suite_path, output_path):
    loss_dataset = LossDataset(loss_dataset_path)

    print("Loading loss dataset...")
    loss_dataset.load()
    print("Loading probe suite...")
    loss_dataset.load_probe_suite(probe_suite_path)

    print("Creating train matrix...")
    X_train, y_train = loss_dataset.to_sklearn_train_matrix()
    print("Creating predict matrix...")
    (
        X_predict,
        original_class,
        sample_indices,
        index_to_label_name,
    ) = loss_dataset.to_sklearn_predict_matrix()

    classifier = XGBClassifier(n_estimators=100)
    print("Training model...")
    print(X_predict.shape)
    classifier = train_metadata_model(classifier, X_train, y_train)

    print("Predicting...")
    y_pred = classifier.predict(X_predict)
    print("Predicting probabilities...")
    y_prob = classifier.predict_proba(X_predict)

    table = pa.Table.from_arrays(
        arrays=[
            pa.array(sample_indices),
            pa.array(y_pred),
            pa.array(original_class),
            pa.array(np.max(y_prob, axis=1)),
            pa.array([index_to_label_name[idx] for idx in y_pred]),
        ],
        names=["sample_index", "label", "original_class", "probs", "label_name"],
    )

    pq.write_table(table, where=output_path)


if __name__ == "__main__":
    main()

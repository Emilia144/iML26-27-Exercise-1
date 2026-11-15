
"""Evaluate interpretability metrics for every dataset and model."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability.data import Dataset
from iml_interpretability.metric import InterpretabilityMetric


def evaluate(
    datasets: Iterable[Dataset],
    models: dict[str, ClassifierMixin],
    metrics: Iterable[InterpretabilityMetric],
    seed: int = 0,
    test_size: float = 0.25,
) -> pd.DataFrame:
    """Train models and evaluate accuracy and interpretability."""

    metrics = list(metrics)
    rows = []

    for dataset in datasets:

        # Split dataset into training and test data.
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X,
            dataset.y,
            test_size=test_size,
            random_state=seed,
            stratify=dataset.y,
        )

        train = Dataset(
            name=dataset.name,
            X=X_train,
            y=y_train,
        )

        for model_name, model in models.items():

            # Train a fresh copy of the model.
            fitted = clone(model).fit(X_train, y_train)

            # Calculate prediction accuracy on test data.
            row = {
                "dataset": dataset.name,
                "model": model_name,
                "accuracy": accuracy_score(
                    y_test,
                    fitted.predict(X_test),
                ),
            }

            # Record actual tree properties.
            if isinstance(fitted, DecisionTreeClassifier):

                row["actual_depth"] = fitted.get_depth()

                row["split_nodes"] = (
                    fitted.tree_.node_count
                    - fitted.get_n_leaves()
                )

                row["used_features"] = len(
                    set(
                        fitted.tree_.feature[
                            fitted.tree_.feature >= 0
                        ]
                    )
                )

            # Calculate interpretability scores.
            for metric in metrics:

                row[metric.name] = metric.score(
                    fitted,
                    train,
                )

                # Store individual metric components.
                if hasattr(metric, "components"):

                    components = metric.components(
                        fitted,
                        train,
                    )

                    for component_name, value in components.items():
                        row[component_name] = value

            rows.append(row)

    return pd.DataFrame(rows)

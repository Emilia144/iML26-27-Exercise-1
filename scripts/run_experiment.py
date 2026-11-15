
"""Evaluate decision tree interpretability on two datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from iml_interpretability import (
    Dataset,
    evaluate,
    get_models,
    load_titanic,
)
from iml_interpretability.metric import GlobalTreeInterpretability


OUTPUT = Path("results/scores_two_datasets.csv")
SEED = 0


def load_cancer_dataset() -> Dataset:
    """Load the Breast Cancer Wisconsin dataset."""

    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names,
    )

    y = pd.Series(data.target, name="target")

    return Dataset(
        name="breast_cancer",
        X=X,
        y=y,
    )


def main() -> None:
    datasets = [
        load_titanic(),
        load_cancer_dataset(),
    ]

    results = evaluate(
        datasets=datasets,
        models=get_models(seed=SEED),
        metrics=[GlobalTreeInterpretability()],
        seed=SEED,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(OUTPUT, index=False)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

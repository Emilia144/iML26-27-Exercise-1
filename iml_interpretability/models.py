
"""Decision Trees with different complexity levels."""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


def get_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return unfitted decision trees with different maximum depths."""

    depths = [1, 2, 3, 4, 5, 6, 8, 10, 15]

    models = {
        f"tree_depth_{depth}": DecisionTreeClassifier(
            max_depth=depth,
            random_state=seed,
        )
        for depth in depths
    }

    # An additional tree without a maximum depth restriction.
    models["tree_unlimited"] = DecisionTreeClassifier(
        random_state=seed,
    )

    return models

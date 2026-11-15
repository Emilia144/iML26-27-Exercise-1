
"""Decision Trees with different complexity levels.

Compare tree structures by varying:
- Maximum tree depth
- Maximum number of leaf nodes
- Minimum samples per leaf
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


def get_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return unfitted Decision Trees with different complexity settings."""

    # 1. Decision Trees with different maximum depths
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 15]

    models = {
        f"tree_depth_{depth}": DecisionTreeClassifier(
            max_depth=depth,
            random_state=seed,
        )
        for depth in depths
    }

    # 2. Decision Tree without depth restriction
    models["tree_unlimited"] = DecisionTreeClassifier(
        random_state=seed,
    )

    # 3. Decision Trees with different maximum leaf counts
    for leaves in [5, 10, 20, 40]:
        models[f"tree_leaves_{leaves}"] = DecisionTreeClassifier(
            max_leaf_nodes=leaves,
            random_state=seed,
        )

    # 4. Decision Trees with different minimum leaf sizes
    for min_samples in [2, 5, 10, 20]:
        models[f"tree_min_samples_{min_samples}"] = DecisionTreeClassifier(
            min_samples_leaf=min_samples,
            random_state=seed,
        )

    return models

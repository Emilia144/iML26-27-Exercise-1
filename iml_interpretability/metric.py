
"""Global Interpretability Metric for Decision Trees.

Measures four aspects of structural complexity:
1. Tree Depth
2. Number of Split Nodes
3. Number of Used Features
4. Feature Combination Complexity

Scores range from 0 to 100.
Higher scores indicate greater estimated interpretability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability.data import Dataset


class InterpretabilityMetric(ABC):
    """Base class for interpretability metrics."""

    name: str

    @abstractmethod
    def score(
        self,
        model: ClassifierMixin,
        dataset: Dataset,
    ) -> float:
        """Calculate the interpretability score."""


class GlobalTreeInterpretability(InterpretabilityMetric):
    """Structural interpretability metric for decision trees."""

    name = "global_tree_interpretability"

    # Experimental normalization parameters
    DEPTH_SCALE = 5.0
    NODE_SCALE = 10.0
    FEATURE_SCALE = 5.0
    COMBINATION_SCALE = 2.0

    def components(
        self,
        model: ClassifierMixin,
        dataset: Dataset,
    ) -> dict[str, float]:
        """Calculate normalized complexity components."""

        if not isinstance(model, DecisionTreeClassifier):
            raise TypeError(
                "Only DecisionTreeClassifier is supported."
            )

        if not hasattr(model, "tree_"):
            raise ValueError("The model must be fitted first.")

        tree = model.tree_

        # 1. Tree Depth
        depth = model.get_depth()

        depth_complexity = (
            depth / (depth + self.DEPTH_SCALE)
        )

        # 2. Number of Split Nodes
        split_nodes = (
            tree.node_count - model.get_n_leaves()
        )

        node_complexity = (
            split_nodes / (split_nodes + self.NODE_SCALE)
        )

        # 3. Number of Used Features
        used_features = len(
            set(tree.feature[tree.feature >= 0])
        )

        feature_complexity = (
            used_features / (used_features + self.FEATURE_SCALE)
        )

        # 4. Feature Combination Complexity
        # Traverse every root-to-leaf path.
        path_combinations = []

        stack = [(0, frozenset())]

        while stack:
            node_id, path_features = stack.pop()

            feature = int(tree.feature[node_id])

            if feature < 0:
                # Leaf reached: count additional unique features.
                path_combinations.append(
                    max(0, len(path_features) - 1)
                )
                continue

            new_features = path_features | {feature}

            stack.append(
                (int(tree.children_left[node_id]), new_features)
            )
            stack.append(
                (int(tree.children_right[node_id]), new_features)
            )

        # Average over all complete decision paths.
        combination_raw = (
            sum(path_combinations) / len(path_combinations)
        )

        combination_complexity = (
            combination_raw
            / (combination_raw + self.COMBINATION_SCALE)
        )

        return {
            "depth_complexity": float(depth_complexity),
            "node_complexity": float(node_complexity),
            "feature_complexity": float(feature_complexity),
            "combination_complexity": float(combination_complexity),
        }

    def score(
        self,
        model: ClassifierMixin,
        dataset: Dataset,
    ) -> float:
        """Return interpretability score between 0 and 100."""

        components = self.components(model, dataset)

        complexity = sum(components.values()) / 4

        interpretability = 100 * (1 - complexity)

        return float(interpretability)

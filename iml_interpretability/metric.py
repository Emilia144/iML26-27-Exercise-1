
"""Global Interpretability Metric for Decision Trees.

Measures three aspects of structural complexity:
1. Tree Depth
2. Number of Split Nodes
3. Number of Used Features

Scores range from 0 to 100.
Higher scores indicate greater estimated global interpretability.
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
    """Estimate global interpretability from tree structure."""

    name = "global_tree_interpretability"

    # Experimental normalization parameters
    DEPTH_SCALE = 5.0
    NODE_SCALE = 10.0
    FEATURE_SCALE = 5.0

    def components(
        self,
        model: ClassifierMixin,
        dataset: Dataset,
    ) -> dict[str, float]:
        """Calculate the three normalized complexity components."""

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

        return {
            "depth_complexity": float(depth_complexity),
            "node_complexity": float(node_complexity),
            "feature_complexity": float(feature_complexity),
        }

    def score(
        self,
        model: ClassifierMixin,
        dataset: Dataset,
    ) -> float:
        """Return interpretability score between 0 and 100."""

        components = self.components(model, dataset)

        # Equal weighting of the three components
        complexity = sum(components.values()) / 3

        # Convert complexity to interpretability
        interpretability = 100 * (1 - complexity)

        return float(interpretability)

class GlobalTreeInterpretability(InterpretabilityMetric):
    """Combine four normalized structural complexity components.

    depth = d / (d + 5)
    nodes = n / (n + 10)
    features = f / (f + 5)
    combinations = c / (c + 2)

    Here c = mean_over_root_to_leaf_paths(unique_features_on_path - 1),
    with a floor of zero. Each leaf path is weighted equally.
    """

    name = "global_tree_interpretability"

    DEPTH_SCALE = 5.0
    NODE_SCALE = 10.0
    FEATURE_SCALE = 5.0
    COMBINATION_SCALE = 2.0

    def components(self, model: ClassifierMixin, dataset: Dataset) -> dict[str, float]:
        """Return normalized components and diagnostic raw values.

        ``dataset`` is accepted to preserve the repository metric interface;
        this metric deliberately measures the full tree structure independent
        of sample frequency or dataset size.
        """
        if not isinstance(model, DecisionTreeClassifier):
            raise TypeError(
                "GlobalTreeInterpretability requires DecisionTreeClassifier, "
                f"got {type(model).__name__}"
            )
        if not hasattr(model, "tree_"):
            raise ValueError("The decision tree must be fitted first.")

        tree = model.tree_
        depth = int(model.get_depth())
        leaves = int(model.get_n_leaves())
        split_nodes = int(tree.node_count - leaves)
        split_features = {int(f) for f in tree.feature if f >= 0}
        used_features = len(split_features)

        # Collect each complete root-to-leaf path. A repeated feature counts
        # just once within the path, because we measure feature diversity.
        extra_feature_counts: list[int] = []
        stack: list[tuple[int, frozenset[int]]] = [(0, frozenset())]
        while stack:
            node_id, features_on_path = stack.pop()
            feature = int(tree.feature[node_id])
            if feature < 0:  # leaf
                extra_feature_counts.append(max(0, len(features_on_path) - 1))
                continue

            next_features = features_on_path | {feature}
            stack.append((int(tree.children_left[node_id]), next_features))
            stack.append((int(tree.children_right[node_id]), next_features))

        combination_raw = sum(extra_feature_counts) / len(extra_feature_counts)

        return {
            "depth": depth / (depth + self.DEPTH_SCALE),
            "split_nodes": split_nodes / (split_nodes + self.NODE_SCALE),
            "used_features": used_features / (used_features + self.FEATURE_SCALE),
            "feature_combinations": combination_raw / (
                combination_raw + self.COMBINATION_SCALE
            ),
            "raw_depth": float(depth),
            "raw_split_nodes": float(split_nodes),
            "raw_used_features": float(used_features),
            "raw_feature_combinations": float(combination_raw),
        }

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        parts = self.components(model, dataset)
        complexity = (
            parts["depth"]
            + parts["split_nodes"]
            + parts["used_features"]
            + parts["feature_combinations"]
        ) / 4.0
        return float(100.0 * (1.0 - complexity))

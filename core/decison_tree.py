import numpy as np
from core.node import Node
from core.metrics import calculate_gain_raio


class C45Decisiontree:
    def __init__(self, min_sample_split=2, max_depth=100):
        self.root = None
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

    def _split_data(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()

        return left_idxs, right_idxs

    def _get_best_split(self, X, y, feature_indices):
        best_split = {}
        max_gain_ratio = -1

        for feature_index in feature_indices:
            X_column = X[:, feature_index]
            possible_thresholds = np.unique(X_column)

            for threshold in possible_thresholds:
                left_idxs, right_idxs = self._split_data(X_column, threshold)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                y_left = y[left_idxs]
                y_right = y[right_idxs]

                gain_ratio = calculate_gain_raio(y, [y_left, y_right])

                if gain_ratio > max_gain_ratio:
                    max_gain_ratio = gain_ratio
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "left_idxs": left_idxs,
                        "right_idxs": right_idxs,
                        "gain_ratio": gain_ratio,
                    }
        return best_split

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
            (n_labels == 1)
            or (depth >= self.max_depth)
            or (n_samples < self.min_sample_split)
        ):
            most_common_label = self._most_common_label(y)
            return Node(value=most_common_label, is_leaf=True)
        feature_indices = np.random.choice(n_features, n_features, replace=False)
        best_split = self._get_best_split(X, y, feature_indices)

        # if couldn't find gain > 0
        if best_split.get("gain_ratio", 0) == 0:
            most_common_label = self._most_common_label(y)
            return Node(value=most_common_label, is_leaf=True)
        # leaf n right tree
        left_subtree = self._build_tree(
            X[best_split["left_idxs"], :], y[best_split["left_idxs"]], depth + 1
        )
        right_subtree = self._build_tree(
            X[best_split["right_idxs"], :], y[best_split["right_idxs"]], depth + 1
        )

        return Node(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            childern={"left": left_subtree, "right": right_subtree},
            is_leaf=False,
        )

    def _most_common_label(self, y):
        if len(y) == 0:
            return None
        return np.bincount(y).argmax()

    def fit(self, X, y):
        # recursive building tree
        self.root = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.childern["left"])
        else:
            return self._traverse_tree(x, node.childern["right"])

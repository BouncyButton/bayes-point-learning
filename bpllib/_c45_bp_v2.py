import math
import numpy as np


class C45DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _calculate_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = sum(probabilities * -np.log2(probabilities))
        return entropy

    def _calculate_information_gain(self, X, y, feature_index):
        entropy_parent = self._calculate_entropy(y)

        feature_values = np.unique(X[:, feature_index])
        entropy_children = 0
        for feature_value in feature_values:
            _, y_sub = y[X[:, feature_index] == feature_value], X[X[:, feature_index] == feature_value]
            weight = len(y_sub) / len(y)
            entropy_children += weight * self._calculate_entropy(y_sub)

        information_gain = entropy_parent - entropy_children
        return information_gain

    def _calculate_split_criterion(self, X, y):
        n_features = X.shape[1]
        information_gains = [self._calculate_information_gain(X, y, i) for i in range(n_features)]
        feature_index = np.argmax(information_gains)
        information_gain = information_gains[feature_index]
        return feature_index, information_gain

    def _split_data(self, X, y, feature_index):
        feature_values = np.unique(X[:, feature_index])
        X_splits = [X[X[:, feature_index] == value] for value in feature_values]
        y_splits = [y[X[:, feature_index] == value] for value in feature_values]
        return X_splits, y_splits

    def _create_leaf_node(self, y):
        _, counts = np.unique(y, return_counts=True)
        class_label = np.argmax(counts)
        return class_label

    def _build_tree(self, X, y, depth):
        # Stop if node has fewer samples than min_samples_split or reached max_depth
        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return self._create_leaf_node(y)

        # Split data and calculate split criterion
        feature_index, information_gain = self._calculate_split_criterion(X, y)

        # Stop if no information gain can be achieved
        if information_gain == 0:
            return self._create_leaf_node(y)

        # Create internal node and recursively build children
        internal_node = {"feature_index": feature_index, "information_gain": information_gain, "children": {}}
        feature_values = np.unique(X[:, feature_index])
        for feature_value in feature_values:
            X_split, y_split = X[X[:, feature_index] == feature_value], y[X[:, feature_index] == feature_value]
            if len(X_split) == 0:
                internal_node["children"][feature_value] = self._create_leaf_node(y)
            else:
                internal_node["children"][feature_value] = self._build_tree(X_split, y_split, depth + 1)

        return internal_node

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        y_pred = []
        for instance in X:
            node = self.tree
            while isinstance(node, dict):
                feature_value = instance[node["feature_index"]]
                if feature_value in node["children"]:
                    node = node["children"][feature_value]
                else:
                    # Handle missing values by predicting the most common class in the parent node
                    _, counts = np.unique(X[:, node["feature_index"]], return_counts=True)
                    most_common_value = np.argmax(counts)
                    if most_common_value in node["children"]:
                        node = node["children"][most_common_value]
                    else:
                        # If the most common value is not in the children, return the parent node's class
                        node = node

            y_pred.append(node)

            return np.array(y_pred)


def get_tree_paths(node, path=None):
    """Recursively traverse the decision tree and return all paths from the root to the leaves."""
    if path is None:
        path = []
    paths = []

    # If the node is a leaf, append the path to the list of paths
    if "class" in node:
        paths.append(path)

    # If the node is not a leaf, recursively traverse its children
    else:
        for feature_value, child_node in node["children"].items():
            # Create a copy of the path and append the feature and value of the current node
            child_path = path.copy()
            child_path.append((node["feature_index"], feature_value))
            paths.extend(get_tree_paths(child_node, child_path))

    return paths

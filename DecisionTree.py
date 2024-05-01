import numpy as np
import math
from typing import Any, Optional, List, Tuple
from collections import Counter

Vector = List[Any]
Matrix = List[Vector]


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        *,
        value: Any = None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left  # the left node that we are pointing to
        self.right = right  # the left node that we are pointing to
        self.value = value  # value if this is a leaf node

    def is_leaf_node(self) -> bool:
        """Returns true if this is a leaf node."""
        return self.value is not None


class DecisionTree:
    def __init__(
        self,
        min_samples_split: int = 2,
        max_depth: int = 100,
        n_features: int | None = None,
    ):
        self.min_samples_split: int = min_samples_split
        self.max_depth: int = max_depth
        self.n_features_to_use: int = n_features
        self.root = None

    def fit(self, xs: Matrix, ys: Vector) -> None:
        """Fit training data."""
        # check if n_features is less than actual feature
        # if more than actual feature, use actual feature
        n_features = xs.shape[1]
        if not self.n_features_to_use:
            self.n_features_to_use = n_features
        else:
            self.n_features_to_use = min(self.n_features_to_use, n_features)
        # create tree
        self.root = self._grow_tree(xs, ys)

    def _grow_tree(self, xs: Matrix, ys: Vector, depth: int = 0) -> Node:
        """Recursive function to split nodes until specified condition is met."""
        n_samples, n_features = xs.shape
        n_labels = len(np.unique(ys))

        # base case
        # return leaf node
        if (
            depth >= self.max_depth  # if max depth is reached
            or n_labels == 1  # if the remaining class is only 1
            or n_samples < self.min_samples_split  # if reach min number of samples
        ):
            return Node(value=self._most_common_label(ys))

        # recursive case
        # find the best split
        feature_idxs = np.random.choice(n_features, self.n_features_to_use, replace=False)
        best_feature, best_threshold = self._best_split(xs, ys, feature_idxs)
        # create child nodes
        left_idxs, right_idxs = self._split(xs[:, best_feature], best_threshold)
        left = self._grow_tree(xs[left_idxs, :], ys[left_idxs], depth + 1)
        right = self._grow_tree(xs[right_idxs, :], ys[right_idxs], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, xs: Matrix, ys: Vector, feature_idxs: Vector) -> Tuple[Vector, float]:
        """Find the feature and threshold that maximize information gain."""
        best_gain = -1  # init
        split_idx, split_threshold = None, None  # init

        # iterate through each feature
        for i in feature_idxs:
            xs_col = xs[:, i]  # get the entire feature (i.e. col in a DF)
            thresholds = np.unique(xs_col)  # get all possible value of this feature
            # iterate through each value in this feature
            for threshold in thresholds:
                # calculate the information gain
                gain = self._information_gain(xs_col, ys, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = i
                    split_threshold = threshold
        return split_idx, split_threshold

    def _information_gain(self, xs_col: Vector, ys: Vector, threshold: float) -> float:
        """Calculates information gain of a split.
        entropy(parent) - weighted_avg(entropy(child))
        """
        left_idx, right_idx = self._split(xs_col, threshold)  # create children
        
        # if the the class is still on one of the split, then the IG = 0
        # i.e. the split has no effect
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        
        # else, calculate weighted entropy of child
        parent_ent = self._entropy(ys)  # get parent entropy
        n = len(ys)
        n_left= len(left_idx)
        n_right = len(right_idx)
        ent_left = self._entropy(ys[left_idx])
        ent_right = self._entropy(ys[right_idx])
        children_ent = (ent_left * n_left / n) + (ent_right * n_right / n) # weighted child ent
        return parent_ent - children_ent # information gain

    def _split(self, xs_col: Vector, threshold: float) -> Tuple[Vector, Vector]:
        """Returns an index of split data points based on threshold."""
        left_idx = np.argwhere(xs_col <= threshold).flatten()
        right_idx = np.argwhere(xs_col > threshold).flatten()
        return left_idx, right_idx

    def _entropy(self, ys: Vector) -> float:
        """
        - sum(prob(x) * log2(prob(x)))
        where prob(x) = count(x) / num_data_point
        """
        n = len(ys)
        counter = Counter(ys)
        return -1 * sum(
            (count / n) * (math.log2(count / n)) for count in counter.values()
        )

    def _most_common_label(self, labels: Vector) -> Any:
        """Returns the most common label."""
        return Counter(labels).most_common(1)[0][0]

    def predict(self, xs: Matrix) -> Vector:
        """Predict a set of samples."""
        return np.array([self._traverse_tree(x, self.root) for x in xs])

    def _traverse_tree(self, x, node: Node) -> Any:
        """Traverse the tree recursively until meeting leaf node."""
        # base case
        if node.is_leaf_node():
            return node.value
        # recursive case: go to left node
        elif x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        # recursive case: go to right node
        else:
            return self._traverse_tree(x, node.right)
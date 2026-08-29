from collections.abc import Callable
from typing import Any

from mllib.math.graph.split_function import GiniImpurity, SplitFunction
from mllib.math.graph.tree_structures import TreeNode
from mllib.ml.evaluators.generic_evaluator import DecisionTreeModelEvaluator


class NoSplitError(ValueError):
    """Raised when a tree cannot be fitted because the data offers nothing to split on."""


class NodeSplitCriteria:
    def __init__(self, column, value, criteria_func: Callable[[Any, Any], bool]):
        self.metadata = {
            "name": "Decision Tree Split Criteria",
            "description": "Encapsulates a column/value rule used to split nodes in a decision tree.",
        }
        # TODO(BL-16): derive metadata by introspection
        self.column = column
        self.value = value
        self.criteria_function = criteria_func

    def get_split(self, data_value):
        return self.criteria_function(data_value, self.value)


class DecisionTree:  # TODO(BL-13): rebuild on mllib.math.graph tree utilities
    def __init__(self, split_function: SplitFunction = None, root=None):
        self.metadata = {
            "name": "Decision Tree Base Class",
            "description": "Core decision tree implementation with training, prediction, and evaluation helpers.",
        }
        # TODO(BL-16): derive metadata by introspection
        self.root = root
        if self.root == None:
            self.root = TreeNode()

        # hyperparameters
        self.max_depth = 15

        self.split_function = GiniImpurity()
        self.evaluator = DecisionTreeModelEvaluator()
        self.evaluation_meta_data = {
            "modelName": "DecisionTreeModel",
            "maxDepth": self.max_depth,
        }

    def fit(self, data_values, data_targets):
        self.build_tree(data_values, data_targets, self.root)
        return self

    def build_tree(self, data_values, data_targets, current_node: TreeNode, depth=0):
        if len(data_targets) == 0:
            raise NoSplitError("cannot fit a decision tree on zero rows")
        if data_values.shape[1] == 0:
            raise NoSplitError("cannot fit a decision tree with no feature columns")

        current_node.prediction = data_targets.mode()[0]  # set the majority as the prediction
        split_column = self.split_function.calculate_split(data_values, data_targets)

        if data_values[split_column].nunique(dropna=True) <= 1:
            # The best available column has a single category here: splitting on it would produce
            # one child identical to this node. Stop and predict the majority instead.
            current_node.is_leaf_node = True
            return
        child_nodes, split_subsets = self.build_split(
            split_column, current_node, data_values, data_targets
        )

        for child_node, (child_data_values_subset, child_data_targets_subset) in zip(
            child_nodes, split_subsets
        ):
            if depth + 1 == self.max_depth:
                child_node.is_leaf_node = True
                child_node.prediction = child_data_targets_subset.mode()[
                    0
                ]  # set the majority as the prediction
            else:
                self.build_tree(
                    child_data_values_subset, child_data_targets_subset, child_node, depth + 1
                )

        if current_node.child_node_count == 0 and not current_node.is_leaf_node:
            current_node.is_leaf_node = True

    def build_split(self, split_column, current_node, data_values, data_targets):
        child_nodes = []
        split_subsets = []

        # we have a column name for the split
        # we need to get all the types for the split column
        unique_column_values = data_values[
            split_column
        ].unique()  # treating all values as categorical bc age has been binned and is the only numerical column atm

        # for each subset we need to create a child code and pair it with the subsets it has
        for category in unique_column_values:
            child_subset = data_values[split_column] == category
            child_data_values_subset = data_values[child_subset]
            child_data_target_subset = data_targets[child_subset]

            if child_data_values_subset.empty:
                continue  # had a nan come up here so just skipping things that done need to be nodes

            # very specific to a data set of all categorical values
            criteria = NodeSplitCriteria(
                split_column, category, criteria_func=lambda val, cat=category: val == cat
            )

            child_node = TreeNode()
            child_node.data = criteria
            current_node.add_child(child_node)
            child_nodes.append(child_node)
            split_subsets.append((child_data_values_subset, child_data_target_subset))

        return child_nodes, split_subsets

    def predict(self, data):
        current_node = self.root
        leaf_node_result = self.traverse_tree(current_node, data)
        return leaf_node_result

    def traverse_tree(self, current_node, data):
        if current_node.is_leaf_node:
            return current_node.prediction

        for child in current_node.child_nodes:
            if isinstance(child.data, int):
                return current_node.prediction

            node_split_criteria = child.data
            if node_split_criteria.criteria_function(data[node_split_criteria.column]):
                return self.traverse_tree(child, data)

        # No child criterion matched (a category unseen at fit time): fall back to this node's
        # majority prediction rather than failing. Intentional — see docs/BACKLOG.md BL-13.
        return current_node.prediction

    def predict_values(self, data_values):
        predicted_values = []
        for idx, data in data_values.iterrows():
            predicted_values.append(self.predict(data))
        return predicted_values

    def evaluate(self, data_values, data_targets, evaluation_meta_data=None):
        if evaluation_meta_data == None:
            evaluation_meta_data = self.evaluation_meta_data
        # Standardize Inputs for compatibility with pandas dataframes as parameters
        predicted_values = self.predict_values(data_values)
        self.evaluator.update_testing_prediction_data(
            data_values, data_targets, predicted_values, evaluation_meta_data
        )


class MyDecisionTree(DecisionTree):
    # implementation for AdClickPredictionProject - scratched this and am using decisionTree which will be abstracted later
    def __init__(self, split_function: SplitFunction = None, root=None):
        super().__init__(split_function=split_function, root=root)
        self.metadata = {
            "name": "Decision Tree Project Wrapper",
            "description": "Project-specific decision tree wrapper for experimentation and extension.",
        }
        # TODO(BL-16): derive metadata by introspection

    def add_node(self, node: TreeNode):
        if self.root == None:
            self.root = node
        self.insert_node()

    def insert_node(self):
        pass

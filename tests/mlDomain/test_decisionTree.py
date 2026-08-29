import pytest

pd = pytest.importorskip("pandas")

from mllib.mlDomain.decisionTree import DecisionTree, NoSplitError


def test_fit_creates_children_and_leaf_predictions():
    data_values = pd.DataFrame({"color": ["red", "red", "blue", "blue"]})
    data_targets = pd.Series([0, 0, 1, 1])

    tree = DecisionTree()
    tree.maxDepth = 1

    tree.fit(data_values, data_targets)

    assert tree.root.childNodeCount == 2
    assert all(child.isLeafNode for child in tree.root.childNodes)

    child_predictions = {child.data.value: child.prediction for child in tree.root.childNodes}
    assert child_predictions["red"] == 0
    assert child_predictions["blue"] == 1


def test_predict_uses_split_criteria():
    data_values = pd.DataFrame({"color": ["red", "red", "blue", "blue"]})
    data_targets = pd.Series([0, 0, 1, 1])

    tree = DecisionTree()
    tree.maxDepth = 1
    tree.fit(data_values, data_targets)

    assert tree.predict(pd.Series({"color": "red"})) == 0
    assert tree.predict(pd.Series({"color": "blue"})) == 1


def test_fit_raises_typed_error_without_feature_columns():
    data_values = pd.DataFrame(index=range(4))
    data_targets = pd.Series([0, 1, 0, 1])

    with pytest.raises(NoSplitError, match="no feature columns"):
        DecisionTree().fit(data_values, data_targets)


def test_fit_raises_typed_error_on_zero_rows():
    data_values = pd.DataFrame({"color": pd.Series([], dtype=str)})
    data_targets = pd.Series([], dtype=int)

    with pytest.raises(NoSplitError, match="zero rows"):
        DecisionTree().fit(data_values, data_targets)


def test_single_category_column_makes_a_leaf_not_a_chain():
    data_values = pd.DataFrame({"color": ["red", "red", "red", "red"]})
    data_targets = pd.Series([0, 0, 1, 0])

    tree = DecisionTree()
    tree.fit(data_values, data_targets)

    assert tree.root.isLeafNode
    assert tree.root.childNodeCount == 0
    assert tree.predict(pd.Series({"color": "red"})) == 0


def test_unseen_category_falls_back_to_parent_majority():
    data_values = pd.DataFrame({"color": ["red", "red", "blue", "blue", "blue"]})
    data_targets = pd.Series([0, 0, 1, 1, 1])

    tree = DecisionTree()
    tree.maxDepth = 1
    tree.fit(data_values, data_targets)

    assert tree.predict(pd.Series({"color": "green"})) == 1

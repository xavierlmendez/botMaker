import pytest

pd = pytest.importorskip("pandas")

from MlLib.mlDomain.decisionTree import DecisionTree


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

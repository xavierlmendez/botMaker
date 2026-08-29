from mllib.math.graph.tree_structures import TreeNode


def test_tree_node_defaults():
    node = TreeNode()

    assert node.parent_node is None
    assert node.data is None
    assert node.child_nodes == []
    assert node.child_node_count == 0
    assert node.is_leaf_node is False


def test_add_child_sets_parent_and_counts():
    parent = TreeNode(data="root")
    child = TreeNode(data="leaf")

    parent.add_child(child)

    assert parent.child_node_count == 1
    assert parent.child_nodes == [child]
    assert child.parent_node is parent


def test_remove_child_clears_parent_and_counts():
    parent = TreeNode(data="root")
    child = TreeNode(data="leaf")
    parent.add_child(child)

    parent.remove_child(child)

    assert parent.child_node_count == 0
    assert parent.child_nodes == []
    assert child.parent_node is None


def test_multiple_children_order_preserved():
    parent = TreeNode(data="root")
    child_a = TreeNode(data="a")
    child_b = TreeNode(data="b")

    parent.add_child(child_a)
    parent.add_child(child_b)

    assert parent.child_node_count == 2
    assert parent.child_nodes == [child_a, child_b]
    assert child_a.parent_node is parent
    assert child_b.parent_node is parent

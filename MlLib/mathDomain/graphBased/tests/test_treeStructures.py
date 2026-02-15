from MlLib.mathDomain.graphBased.treeStructures import TreeNode


def test_tree_node_defaults():
    node = TreeNode()

    assert node.parentNode is None
    assert node.data is None
    assert node.childNodes == []
    assert node.childNodeCount == 0
    assert node.isLeafNode is False


def test_add_child_sets_parent_and_counts():
    parent = TreeNode(data="root")
    child = TreeNode(data="leaf")

    parent.addChild(child)

    assert parent.childNodeCount == 1
    assert parent.childNodes == [child]
    assert child.parentNode is parent


def test_remove_child_clears_parent_and_counts():
    parent = TreeNode(data="root")
    child = TreeNode(data="leaf")
    parent.addChild(child)

    parent.removeChild(child)

    assert parent.childNodeCount == 0
    assert parent.childNodes == []
    assert child.parentNode is None


def test_multiple_children_order_preserved():
    parent = TreeNode(data="root")
    child_a = TreeNode(data="a")
    child_b = TreeNode(data="b")

    parent.addChild(child_a)
    parent.addChild(child_b)

    assert parent.childNodeCount == 2
    assert parent.childNodes == [child_a, child_b]
    assert child_a.parentNode is parent
    assert child_b.parentNode is parent

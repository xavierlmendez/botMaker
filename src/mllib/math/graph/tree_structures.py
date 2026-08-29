import numpy as np


class TreeNode:
    """Generic tree node with parent/children tracking."""

    def __init__(self, parent_node=None, data=None, children: np.ndarray = None):
        self.parent_node = parent_node
        self.data = data  # leaving abstract here to allow more options in the decision tree and other tree implementations
        self.child_nodes = []
        self.child_node_count = 0
        self.is_leaf_node = False

    def add_child(self, child: object):
        self.child_node_count += 1
        self.child_nodes.append(child)
        child.parent_node = self  # if not set on init then this will correct

    def remove_child(self, child: object):
        self.child_node_count -= 1
        self.child_nodes.remove(child)
        child.parent_node = None  # if not set on init then this will correct

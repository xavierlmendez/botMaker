import os
import sys

# add parent folder (/mlLib) to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import from the sibling package mathDomain
from mathDomain.splitFunction import SplitFunction
from mathDomain.treeNode import TreeNode
class AbstractDecisionTree:
    def __init__(self, splitFunction:SplitFunction, root = None):
        self.splitFunction = splitFunction
        self.root = root
        self.maxDepth = 5
        self.maxLeaves = 5
        
    def addNode(self, node:TreeNode):
        if self.root == None:
            self.root = node
        self.insertNode()
        
    def insertNode(self):
        pass
    
class MyDecisionTree(AbstractDecisionTree):
    # implementation for AdClickPredictionProject
    def addNode(self, node:TreeNode):
        if self.root == None:
            self.root = node
        self.insertNode()

    def insertNode(self):
        pass
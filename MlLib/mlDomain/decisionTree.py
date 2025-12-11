import os
import sys

# add parent folder (/mlLib) to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import from the sibling package mathDomain
from mathDomain.splitFunction import SplitFunction
from mathDomain.treeNode import TreeNode
from typing import Callable, Any

class nodeSplitCriteria:
    def __init__(self, column, value, criteriaFunc:Callable[[Any, Any], bool]):
        self.column = column
        self.value = value
        self.criteriaFunction = criteriaFunc
        
    def getSplit(self, dataValue):
        return self.criteriaFunction(dataValue, self.value)

class DecisionTree:
    def __init__(self, splitFunction:SplitFunction, root = None):
        self.splitFunction = splitFunction
        self.root = root
        if self.root == None:
            root = TreeNode()
            self.addNode(root)
            
        self.maxDepth = 5
        self.maxLeaves = 5
        self.numSplits = 3

    def fit(self, dataValues, dataTargets):
        splitClass = self.splitFunction.determineSplit(dataValues, dataTargets)
        childNodes = self.buildSplit(splitClass)
        
        for childNode in childNodes:
            addNode(self.root, childNode)
            splitNode = fit(dataValuesSubset, dataTargetsSubset)
        return self

    def buildSplit(self, splitClass):
        for i in range(self.numSplits):
            
        pass


    def addRootNode(self, node:TreeNode, splitCriteria):
            if self.root == None:
                self.root = node
    
class MyDecisionTree(DecisionTree):
    # implementation for AdClickPredictionProject
    def addNode(self, node:TreeNode):
        if self.root == None:
            self.root = node
        self.insertNode()

    def insertNode(self):
        pass
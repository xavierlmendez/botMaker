import os
import sys

import numpy as np
import pandas as pd

# add parent folder (/mlLib) to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import from the sibling package mathDomain
from mathDomain.graphBased.splitFunction import SplitFunction, GiniImpurity
from mathDomain.graphBased.treeNode import TreeNode
from mlDomain.modelEvaluators.genericEvaluator import DecisionTreeModelEvaluator
from typing import Callable, Any

class nodeSplitCriteria:
    def __init__(self, column, value, criteriaFunc:Callable[[Any, Any], bool]):
        self.column = column
        self.value = value
        self.criteriaFunction = criteriaFunc
        
    def getSplit(self, dataValue):
        return self.criteriaFunction(dataValue, self.value)

class DecisionTree:
    def __init__(self, splitFunction:SplitFunction = None, root = None):
        self.root = root
        if self.root == None:
            self.root = TreeNode()
            
        # hyperparameters
        self.maxDepth = 20
        
        self.splitFunction = GiniImpurity()
        self.evaluator = DecisionTreeModelEvaluator()
        self.evaluationMetaData = np.array([
            {
                'modelName': ['DecisionTreeModel'],
            }
        ])

    def fit(self, dataValues, dataTargets):
        self.buildTree(dataValues, dataTargets, self.root)
        return self
    
    def buildTree(self, dataValues, dataTargets, currentNode:TreeNode, depth=0):
        
        splitColumn = self.splitFunction.calculateSplit(dataValues, dataTargets)
        childNodes, splitSubsets = self.buildSplit(splitColumn, currentNode, dataValues, dataTargets)

        for childNode, (childDataValuesSubset, childDataTargetsSubset) in zip(childNodes, splitSubsets):
            if depth + 1 == self.maxDepth:
                childNode.isLeafNode = True
                childNode.prediction = childDataTargetsSubset.mode()[0] # set the majority as the prediction
            else:
                self.buildTree(
                    childDataValuesSubset,
                    childDataTargetsSubset,
                    childNode,
                    depth + 1
                )
        
        if currentNode.childNodeCount == 0:
            currentNode.isLeafNode = True

    def buildSplit(self, splitColumn, currentNode, dataValues, dataTargets):
        childNodes = []
        splitSubsets = []
        
        # we have a column name for the split
        # we need to get all the types for the split column
        uniqueColumnValues = dataValues[splitColumn].unique() # treating all values as categorical bc age has been binned and is the only numerical column atm
        
        # for each subset we need to create a child code and pair it with the subsets it has
        for category in uniqueColumnValues:
            childSubset = (dataValues[splitColumn] == category)
            childDataValuesSubset = dataValues[childSubset]
            childDataTargetSubset = dataTargets[childSubset]

            if childDataValuesSubset.empty:
                continue # had a nan come up here so just skipping things that done need to be nodes
                
            # very specific to a data set of all categorical values
            criteria = nodeSplitCriteria(splitColumn, category, criteriaFunc=lambda val, cat=category: val == cat)
            
            childNode = TreeNode()
            childNode.data = criteria
            currentNode.addChild(childNode)
            childNodes.append(childNode)
            splitSubsets.append((childDataValuesSubset, childDataTargetSubset))

        return childNodes, splitSubsets

    def predict(self, data):
        currentNode = self.root
        leafNodeResult = self.traverseTree(currentNode, data)
        return leafNodeResult
    
    def traverseTree(self, currentNode, data):
        if currentNode.isLeafNode:
            return currentNode.prediction
        
        for child in currentNode.childNodes:
            if isinstance(child.data, int):
                return child.data
            
            nodeSplitCriteria = child.data
            if nodeSplitCriteria.criteriaFunction(data[nodeSplitCriteria.column]):
                return self.traverseTree(child, data)
        
        test = 1
        # todo add error handling if no split found
            

    def predictValues(self, dataValues):
        predictedValues = []
        for idx, data in dataValues.iterrows():
            predictedValues.append(self.predict(data))
        return predictedValues

    def evaluate(self, dataValues, dataTargets, evaluationMetaData= None):
        if evaluationMetaData == None:
            evaluationMetaData = self.evaluationMetaData
        # Standardize Inputs for compatibility with pandas dataframes as parameters
        predictedValues = self.predictValues(dataValues)
        self.evaluator.updateTestingPredictionData(dataValues, dataTargets, predictedValues, evaluationMetaData)

    
class MyDecisionTree(DecisionTree):
    # implementation for AdClickPredictionProject - scratched this and am using decisionTree which will be abstracted later
    def addNode(self, node:TreeNode):
        if self.root == None:
            self.root = node
        self.insertNode()

    def insertNode(self):
        pass
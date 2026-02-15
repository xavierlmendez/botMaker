import numpy as np
import pandas as pd

from MlLib.mathDomain.graphBased.splitFunction import SplitFunction, GiniImpurity
from MlLib.mathDomain.graphBased.treeStructures import TreeNode
from MlLib.mlDomain.modelEvaluators.genericEvaluator import DecisionTreeModelEvaluator
from typing import Callable, Any

class nodeSplitCriteria:
    def __init__(self, column, value, criteriaFunc:Callable[[Any, Any], bool]):
        self.metadata = {
            "name": "Decision Tree Split Criteria",
            "description": "Encapsulates a column/value rule used to split nodes in a decision tree."
        }
        # TODO: review metadata (auto-generated)
        self.column = column
        self.value = value
        self.criteriaFunction = criteriaFunc
        
    def getSplit(self, dataValue):
        return self.criteriaFunction(dataValue, self.value)

class DecisionTree: # todo refactor to use graphBased utilities of MathDomain
    def __init__(self, splitFunction:SplitFunction = None, root = None):
        self.metadata = {
            "name": "Decision Tree Base Class",
            "description": "Core decision tree implementation with training, prediction, and evaluation helpers."
        }
        # TODO: review metadata (auto-generated)
        self.root = root
        if self.root == None:
            self.root = TreeNode()
            
        # hyperparameters
        self.maxDepth = 15
        
        self.splitFunction = GiniImpurity()
        self.evaluator = DecisionTreeModelEvaluator()
        self.evaluationMetaData = {
                'modelName': 'DecisionTreeModel',
                'maxDepth': self.maxDepth,
            }


    def fit(self, dataValues, dataTargets):
        self.buildTree(dataValues, dataTargets, self.root)
        return self
    
    def buildTree(self, dataValues, dataTargets, currentNode:TreeNode, depth=0):
        currentNode.prediction = dataTargets.mode()[0] # set the majority as the prediction
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
        
        if currentNode.childNodeCount == 0 and not currentNode.isLeafNode :
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
                return currentNode.prediction
            
            nodeSplitCriteria = child.data
            if nodeSplitCriteria.criteriaFunction(data[nodeSplitCriteria.column]):
                return self.traverseTree(child, data)
        
        return currentNode.prediction # case when no children have the class needed to continue likely due to too little model complexity
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
    def __init__(self, splitFunction:SplitFunction = None, root = None):
        super().__init__(splitFunction=splitFunction, root=root)
        self.metadata = {
            "name": "Decision Tree Project Wrapper",
            "description": "Project-specific decision tree wrapper for experimentation and extension."
        }
        # TODO: review metadata (auto-generated)

    def addNode(self, node:TreeNode):
        if self.root == None:
            self.root = node
        self.insertNode()

    def insertNode(self):
        pass

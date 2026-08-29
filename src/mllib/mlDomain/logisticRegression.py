import time

import numpy as np
import pandas as pd
from numpy import array, mean, sum
from sklearn.model_selection import ParameterGrid

# Linear and logistic are very similar however have a major difference in that logistic is for classification
# TODO(BL-20): share a gradient-descent base with linear regression
#  as the main difference is the compute prediction function
from mllib.mathDomain.hypothesis import HypothesisFunction
from mllib.mathDomain.hypothesisExpander import PolynomialRegressionExpander
from mllib.mathDomain.lossFunction import MSE
from mllib.mlDomain.modelEvaluators.genericEvaluator import LogisticRegressionModelEvaluator


class MyLogisticRegression:  # prefixing with my for the comparison script, rename later when cleaning up files
    # choosing 0.001 for default learning rate bc thats what adam uses
    def __init__(self, learningRate=0.001, epochs=10, numWeights=1):
        self.metadata = {
            "name": "Logistic Regression Base Class",
            "description": "Core logistic regression implementation with training, prediction, and evaluation helpers.",
        }
        # TODO(BL-16): derive metadata by introspection
        seededRand = np.random.default_rng(10)  # seeting a seed for random initial weights,
        self.numWeights = numWeights  # number of feature weights; project subclasses pass theirs
        initialWeights = seededRand.random(self.numWeights)
        initialBias = 0
        # Same construction gridFit uses per permutation; a plain linear hypothesis by default.
        self.learningModel = HypothesisFunction(
            initialWeights,
            initialBias,
            degree=1,
            hypothesisExpander=PolynomialRegressionExpander(1),
        )
        self.lossFunction = MSE()
        self.learningRate = learningRate
        self.epochs = epochs
        self.evaluator = LogisticRegressionModelEvaluator()
        self.hyperparameterGridOptions = None

    def gridFit(
        self, trainValues, testValues, trainTargets, testTargets
    ):  # TODO(BL-20): move the train/test split into gridFit as a grid parameter
        hyperparameterCombinations = list(ParameterGrid(self.hyperparameterGridOptions))
        modelImplementationName = self.hyperparameterGridOptions[0]["modelName"][0]
        print(
            f" Starting model training for {type(self).__name__} implementation: {modelImplementationName}\n"
        )

        countModels = hyperparameterCombinations.__len__()
        print(f" Total model permutations: {countModels}")

        modelNumber = 0
        startTime = time.perf_counter()
        for parameterSetting in hyperparameterCombinations:
            modelNumber += 1

            self.lossFunction = parameterSetting["lossFunction"]
            self.epochs = parameterSetting["epoch"]
            self.learningRate = parameterSetting["learningRate"]
            seededRand = np.random.default_rng(parameterSetting["weightRandSeed"])
            initialWeights = seededRand.random(self.numWeights)
            initialBias = parameterSetting["initialBias"]
            self.learningModel = HypothesisFunction(
                initialWeights,
                initialBias,
                parameterSetting["polynomialDegree"],
                parameterSetting["HypothesisExpander"],
            )
            hypothesisSpaceAdjustedWeights = self.learningModel.hypothesisExpander.expandHypothesis(
                initialWeights
            )
            self.updateWeights(hypothesisSpaceAdjustedWeights, initialBias)

            for epoch in range(self.epochs):
                newWeights, newBias = self.calculateGradientDescent(trainValues, trainTargets)
                self.updateWeights(newWeights, newBias)
                cost = self.calculateCostFunction(trainValues, trainTargets)

            self.evaluate(testValues, testTargets, parameterSetting)

            if (
                modelNumber % 1 == 0
            ):  # modify depending on number of permutations i.e. 300+ then probably modulo 40 or 80
                print(f"\tModel Number {modelNumber}/{countModels} complete")

        endTime = time.perf_counter()
        timeElapsed = endTime - startTime
        timePerModel = timeElapsed / countModels
        print(f" Training Time Elapsed: {timeElapsed}, time per model: {timePerModel}")

    def fit(self, dataValues, dataTargets):
        # for n epochs
        # calculate the cost function - in the gradientDescent function
        # compute the gradient
        # update the weights
        # repeat
        for epoch in range(self.epochs):
            newWeights, newBias = self.calculateGradientDescent(dataValues, dataTargets)
            self.updateWeights(newWeights, newBias)
            cost = self.calculateCostFunction(dataValues, dataTargets)
            # use evaluator class here to aggregate data on performance during training
        return self

    def predict(self, data):
        return self.learningModel.computeClassification(data)

    def predictValues(self, dataValues, isDataframe=False):
        predictedValues = []
        for data in dataValues:
            predictedValues.append(self.predict(data))
        return array(predictedValues)

    def evaluate(self, dataValues, dataTargets, evaluationMetaData):
        # Standardize Inputs for compatibility with pandas dataframes as parameters
        dataValues, dataTargets = self.dataFrameCrossCapatibility(dataValues, dataTargets)
        predictedValues = self.predictValues(dataValues, dataTargets)
        self.evaluator.updateTestingPredictionData(
            dataValues, dataTargets, predictedValues, evaluationMetaData
        )

    def calculateGradientDescent(self, dataValues, dataTargets):
        # standardize Inputs for compatibility with pandas dataframes as parameters
        dataValues, dataTargets = self.dataFrameCrossCapatibility(dataValues, dataTargets)

        # calculate the gradient
        predicted = self.predictValues(dataValues)
        gradientDescentAdjustedDataTargets = self.lossFunction.computeGradient(
            dataTargets, predicted
        )
        dataValues = self.learningModel.hypothesisExpander.fitDataToHypothesis(dataValues)
        gradientDescentAdjustedWeights = dataValues.T @ gradientDescentAdjustedDataTargets
        adjustedBias = sum(gradientDescentAdjustedDataTargets)
        return gradientDescentAdjustedWeights, adjustedBias

    def updateWeights(self, gradientDescentAdjustedWeights, gradientDescentAdjustedBias):
        # update the weights and bias
        newWeights = (
            self.learningModel.getWeights() - self.learningRate * gradientDescentAdjustedWeights
        )
        newBias = self.learningModel.getBias() - self.learningRate * gradientDescentAdjustedBias
        self.learningModel.updateWeights(newWeights)
        self.learningModel.updateBias(newBias)

    def calculateCostFunction(self, dataValues, dataTargets):
        # Standardize Inputs for compatibility with pandas dataframes as parameters
        dataValues, dataTargets = self.dataFrameCrossCapatibility(dataValues, dataTargets)

        # here were putting together the cost function as a set of linear equations
        # doing it this way to leverage linear algebra packages
        predicted = self.predictValues(dataValues)
        lossAcrossData = self.lossFunction.computeLoss(dataTargets, predicted)
        return mean(lossAcrossData)

    def dataFrameCrossCapatibility(self, dataValues, dataTargets):
        if isinstance(dataValues, pd.DataFrame):
            dataValues = dataValues.to_numpy()
            dataTargets = np.asarray(dataTargets).ravel()
        return dataValues, dataTargets

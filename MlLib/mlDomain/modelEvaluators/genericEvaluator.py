import numpy as np
from numpy import ndarray


class ModelEvaluator:
    def __init__(self):
        self.runIteration = 0
        self.evaluationRecord: dict[int, dict] = {}

    def updateTestingPredictionData(self, test_values:ndarray, test_targets:ndarray, predictions:ndarray):
        self.runIteration += 1
        self.test_values = test_values
        self.test_targets = test_targets
        self.predictions = predictions
        self.correctPredictions = 0
        self.truePositives = 0
        self.falsePositives = 0
        self.trueNegatives = 0
        self.falseNegatives = 0
        self.evaluateModel()
        self.persistEvaluationRecord()
        
    def getAccuracy(self):
        correctPredictions = 0
        countTotalPredictions = self.predictions.size
        
        for i in range(countTotalPredictions):
            if self.test_targets[i] == self.predictions[i]:
                correctPredictions += 1
                
        # set to self to reuse for future calculations that would run after this
        self.correctPredictions = correctPredictions
        return correctPredictions / countTotalPredictions
    
    def persistEvaluationRecord(self):
        self.evaluationRecord[self.runIteration] = {
            self.correctPredictions,
            self.truePositives,
            self.falsePositives,
            self.trueNegatives,
            self.falseNegatives,
        }
        
    def printEvaluation(self):
        print(self.evaluationRecord)

    def evaluateModel(self):
        raise NotImplementedError("Subclasses must implement evaluateModel()")


class LogisticRegressionModelEvaluator(ModelEvaluator):
    
    def evaluateModel(self):
        self.setConfusionMatrixValues()
        self.accuracy = self.getAccuracy()
        self.precision = self.getPrecision()
        self.recall = self.getRecall()
            
    def setConfusionMatrixValues(self):
        truePositives = 0
        falsePositives = 0
        trueNegatives = 0
        falseNegatives = 0
        countTotalPredictions = self.predictions.size
        
        for i in range(countTotalPredictions):
            if self.test_targets[i] == self.predictions[i] and self.test_targets[i] == 1:
                truePositives += 1
            if self.test_targets[i] != self.predictions[i] and self.test_targets[i] == 1:
                falsePositives += 1
            if self.test_targets[i] == self.predictions[i] and self.test_targets[i] == 0:
                trueNegatives += 1
            if self.test_targets[i] != self.predictions[i] and self.test_targets[i] == 0:
                falseNegatives += 1
                
        self.truePositives = truePositives
        self.falsePositives = falsePositives
        self.trueNegatives = trueNegatives
        self.falseNegatives = falseNegatives
                
        
    def getPrecision(self):
        return self.truePositives / (self.truePositives + self.falsePositives)
    
    def getRecall(self):
        return self.truePositives / (self.truePositives + self.falseNegatives)
        
    def getMSE(self):
        pass

    def persistEvaluationRecord(self):
        self.evaluationRecord[self.runIteration] = {
            "correctPredictions": self.correctPredictions,
            "truePositives": self.truePositives,
            "falsePositives": self.falsePositives,
            "trueNegatives": self.trueNegatives,
            "falseNegatives": self.falseNegatives,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall
        }
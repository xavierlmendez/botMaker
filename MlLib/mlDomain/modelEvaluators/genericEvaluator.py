from json import dumps

import numpy as np
from numpy import ndarray


class ModelEvaluator:
    def __init__(self):
        self.runIteration = 0
        self.evaluationRecord: dict[int, dict] = {}

    def updateTestingPredictionData(self, test_values:ndarray, test_targets:ndarray, predictions:ndarray, evaluationMetaData):
        self.runIteration += 1
        self.test_values = test_values
        self.test_targets = test_targets
        self.predictions = predictions
        self.evaluationMetaData = evaluationMetaData
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
            self.evaluationMetaData,
            self.correctPredictions,
            self.truePositives,
            self.falsePositives,
            self.trueNegatives,
            self.falseNegatives,
        }
        
    def printEvaluation(self, printBestModelStatsOnly=False):
        if not printBestModelStatsOnly:
            formattedEvalJson = dumps(self.evaluationRecord, indent=4)
            print(formattedEvalJson)
        self.printEvaluationStats()

    def printEvaluationStats(self):
        eval_data = self.evaluationRecord

        bestAccuracy = {'iteration': None, 'value': float('-inf')}
        bestPrecision = {'iteration': None, 'value': float('-inf')}
        bestRecall = {'iteration': None, 'value': float('-inf')}
    
        for iteration, metrics in self.evaluationRecord.items():
            accuracy = metrics.get('accuracy')
            precision = metrics.get('precision')
            recall = metrics.get('recall')
    
            if accuracy > bestAccuracy['value']:
                bestAccuracy = {'iteration': iteration, 'value': accuracy}
    
            if precision > bestPrecision['value']:
                bestPrecision = {'iteration': iteration, 'value': precision}
    
            if recall > bestRecall['value']:
                bestRecall = {'iteration': iteration, 'value': recall}
    
        print(f"\nEvaluation Summary : {self.evaluationMetaData['modelName']} ")
    
        print(
            f"Best Accuracy : {bestAccuracy['value']:.4f} "
            f"(Iteration {bestAccuracy['iteration']})"
        )
    
        print(
            f"Best Precision: {bestPrecision['value']:.4f} "
            f"(Iteration {bestPrecision['iteration']})"
        )
    
        print(
            f"Best Recall   : {bestRecall['value']:.4f} "
            f"(Iteration {bestRecall['iteration']})"
        )
        bestModelIterations = [bestAccuracy['iteration'], bestPrecision['iteration'], bestRecall['iteration']]
        bestModelIterations = list(set(bestModelIterations)) # remove duplicate if the same model is best for multiple metrics

        for iteration in bestModelIterations:
            print(f"\n(Iteration {iteration})")
            modelIteration = self.evaluationRecord.get(iteration)
            formattedEvalJson = dumps(modelIteration, indent=4)
            print(formattedEvalJson)
            
        
        

    def evaluateModel(self):
        raise NotImplementedError("Subclasses must implement evaluateModel()")


class LogisticRegressionModelEvaluator(ModelEvaluator):
    
    def __init__(self):
        super().__init__()
        self.evaluationMetaData = None

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
        # since the json package cant serialize python class objects we'll replace the class object with a class name str
        parsedMetaData = dict(self.evaluationMetaData)
        if 'lossFunction' in parsedMetaData:
            parsedMetaData['lossFunction'] = (
                parsedMetaData['lossFunction'].__class__.__name__
            )

        self.evaluationRecord[self.runIteration] = {
            "modelData": parsedMetaData,
            "correctPredictions": self.correctPredictions,
            "truePositives": self.truePositives,
            "falsePositives": self.falsePositives,
            "trueNegatives": self.trueNegatives,
            "falseNegatives": self.falseNegatives,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall
        }
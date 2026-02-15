import pandas as pd


class SplitFunction:
    def __init__(self):
        self.metadata = {
            "name": "Split Function Base Class",
            "description": "Base class for decision tree split criteria implementations."
        }
        # TODO: review metadata (auto-generated)
    
    def calculateSplit(self, dataValues, dataTargets):
        return "class"
    
    def classProbabilities(self, columnValues:pd.DataFrame, dataTargets):
        classesInColumn = columnValues.unique()
        classProbabilities: dict[str, float] = {}
        
        for uniqueClass in classesInColumn:
            recordCriteria = (columnValues == uniqueClass)
            total = columnValues.count()
            numTargetsWithClass = dataTargets[recordCriteria].sum()
            
            if numTargetsWithClass != 0:
                probabilityOfClass = (numTargetsWithClass / total)
                classProbabilities[uniqueClass] = probabilityOfClass
            
        return classProbabilities

class GiniImpurity(SplitFunction):
    metadata = {
        "name": "Gini Impurity",
        "description": "Split function using Gini impurity to choose the best feature."
    }
    # TODO: review metadata (auto-generated)
    def calculateGiniImpurities(self, dataValues, dataTargets):
        columns = dataValues.columns
        giniImpurities: dict[str, int] = {}
    
        for columnName in columns:
            column = dataValues[columnName]
            classProbabilities = self.classProbabilities(column, dataTargets)
            giniImpurities[columnName] = 1 - sum(p ** 2 for p in classProbabilities.values()) # 1 - summation of class probabilits squard is gini impurity formula 

        return giniImpurities
    
    def calculateSplit(self, dataValues, dataTargets):
        giniImpurities = self.calculateGiniImpurities(dataValues, dataTargets)
        return  max(giniImpurities, key=lambda col: abs(giniImpurities[col] - 0.5)) # the value furthest from .5 provides the most information

class InformationGain(SplitFunction):
    def __init__(self):
        self.metadata = {
            "name": "Information Gain",
            "description": "Split function placeholder for information gain."
        }
        # TODO: review metadata (auto-generated)

class ChiSquare(SplitFunction):
    def __init__(self):
        self.metadata = {
            "name": "Chi Square",
            "description": "Split function placeholder for chi-square based splitting."
        }
        # TODO: review metadata (auto-generated)

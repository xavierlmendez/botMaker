import pandas as pd


class SplitFunction:
    def __init__(self):
        pass
    
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
        pass

class ChiSquare(SplitFunction):
    def __init__(self):
        pass
import pandas as pd
import numpy  as np
import json

import importlib

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# class TransformerPipeline:
#     def __init__(self):
#         self.transformers = []
#         
#     def addTransformer(self, column, transformer):
#         
# 
#     def load_class(module_path: str, class_name: str):
#         module = importlib.import_module(module_path)
#         cls = getattr(module, class_name)
#         return cls

class DataTransformer:
    def __init__(self, df:pd.DataFrame): #, transformationFile:str):
        self.metadata = {
            "name": "Data Transformer",
            "description": "This class provides data transformation operations and to columns in a dataset"
        }
        
        # self.transformations = transformationFile # default to none
        # self.transformationPipeline = TransformerPipeline()
        # convert transformation settings stored under project specific files in data domain to np array 
        # json_file_path = transformationFile
        # with open(json_file_path, 'r') as f:
        #     transformations = np.array(json.load(f)) # format "model" {"columnToApplyTo" : "transformation"}
            # need a per model basis here to account for differences in how data is used 
            # i.e. decision trees split on features where neural networks use features for calculations and would be affected by one hot encoding
        # self.transformations = transformations
        # self.pipeline = TransformerPipeline()
        
        # TODO remove and create pipeline for future projects
        self.buildTransformedDataframes(df)
        
        
    def transformData(self, model:str):
        self.transforedData = self.transformerPipeline.excute()
        pass
    
    def addTransformation(self, column: str, transformation: str):
        if transformation not in self.transformations[column]:
            self.transformations = np.append(self.transformations[column], transformation)
            
    # TODO finish above pipeline arch when time allows, for current project just get it done
    def buildTransformedDataframes(self, df:DataFrame):
        self.logisticModelDataFrame = self.tempLogisticRegModelTransformer(df)
        self.logisticModelWithAgeBinningDataFrame = self.tempLogisticRegModelWithAgeBinningTransformer(df)
        self.decisionTreeDataFrame = self.tempDecisionTreeTransformer(df)
        self.neuralNetworkDataFrame = self.tempNeuralNetworkModelTransformer(df)
    
    def tempLogisticRegModelTransformer(self, df:DataFrame):
        transformedDataFrame = df

        allColumnsForEasyReference = np.array([
            "id",
            "full_name",
            "age",
            "gender",
            "device_type",
            "ad_position",
            "browsing_history",
            "time_of_day",
            "click"
        ])

        columnsToRemove = np.array([
            "id",
            "full_name",
        ])
        transformedDataFrame = self.removeColumns(transformedDataFrame, columnsToRemove)
        
        columnsToOneHotEncode = np.array([
            "gender",
            "device_type",
            "ad_position",
            "browsing_history",
            "time_of_day",
        ])
        transformedDataFrame = self.oneHotEncodeCategoricalColumns(transformedDataFrame, columnsToOneHotEncode)

        columnsToStandardize = np.array([
            "age",
        ])
        transformedDataFrame = self.standardizeNumericColumns(transformedDataFrame, columnsToStandardize)
        
        valuesToReplaceWithAverage = np.array([
            np.nan,
        ])
        transformedDataFrame = self.replaceValuesWithNumericAvg(transformedDataFrame, columnsToStandardize, valuesToReplaceWithAverage)
        
        return transformedDataFrame

    def tempLogisticRegModelWithAgeBinningTransformer(self, df:DataFrame):
        transformedDataFrame = df

        allColumnsForEasyReference = np.array([
            "id",
            "full_name",
            "age",
            "gender",
            "device_type",
            "ad_position",
            "browsing_history",
            "time_of_day",
            "click"
        ])

        columnsToRemove = np.array([
            "id",
            "full_name",
        ])
        transformedDataFrame = self.removeColumns(transformedDataFrame, columnsToRemove)

        columnsToStandardize = np.array([
            "age",
        ])
        transformedDataFrame = self.standardizeNumericColumns(transformedDataFrame, columnsToStandardize)

        valuesToReplaceWithAverage = np.array([
            np.nan,
        ])
        transformedDataFrame = self.replaceValuesWithNumericAvg(transformedDataFrame, columnsToStandardize, valuesToReplaceWithAverage)
        
        columnsToBin = np.array([
            "age"
        ])
        transformedDataFrame  = self.binNumericColumnsByStdRanges(transformedDataFrame, columnsToBin) # using ten bins for now 

        columnsToOneHotEncode = np.array([
            "age",
            "gender",
            "device_type",
            "ad_position",
            "browsing_history",
            "time_of_day",
        ])
        transformedDataFrame = self.oneHotEncodeCategoricalColumns(transformedDataFrame, columnsToOneHotEncode)

        return transformedDataFrame

    def tempDecisionTreeTransformer(self, df:DataFrame):
        transformedDataFrame = df

        allColumnsForEasyReference = np.array([
            "id",
            "full_name",
            "age",
            "gender",
            "device_type",
            "ad_position",
            "browsing_history",
            "time_of_day",
            "click"
        ])
        
        transformedDataFrame = self.replaceNanWithString(df, allColumnsForEasyReference)

        columnsToRemove = np.array([
            "id",
            "full_name",
        ])
        transformedDataFrame = self.removeColumns(transformedDataFrame, columnsToRemove)

        columnsToStandardize = np.array([
            "age",
        ])
        transformedDataFrame = self.standardizeNumericColumns(transformedDataFrame, columnsToStandardize)

        valuesToReplaceWithAverage = np.array([
            np.nan,
        ])
        transformedDataFrame = self.replaceValuesWithNumericAvg(transformedDataFrame, columnsToStandardize, valuesToReplaceWithAverage)

        columnsToBin = np.array([
            "age"
        ])
        transformedDataFrame  = self.binNumericColumnsByStdRanges(transformedDataFrame, columnsToBin) # using ten bins for now 

        return transformedDataFrame

    def tempNeuralNetworkModelTransformer(self, df:DataFrame):
        transformedDataFrame = df
        # todo transformations
        return transformedDataFrame

    # TODO migrate below function to pipeline arch when time allows, for current project just get it done
    def oneHotEncodeCategoricalColumns(self, df, columns: np.ndarray, asBoolean = False):
        dtypeOption = bool if asBoolean else float
        # Data set for click prediction project has blanks so setting dummy_na to true here
        df = pd.get_dummies(df, columns=columns, drop_first=True, dummy_na=True, dtype=dtypeOption)
        return df

    def standardizeNumericColumns(self, df, columns: np.ndarray):
        sc = StandardScaler() # todo implement custom version of this and move this function there directly 
        # to allow for signature of res = func(df, columnsToStandardize) and in place scaling + other scaling options in the signature
        df[columns] = sc.fit_transform(df[columns])
        return df
    
    def replaceValuesWithNumericAvg(self, df, columns: np.ndarray, values: np.ndarray):
        for col in columns:
            colMean = df[col].mean()
            for val in values:
                df[col] = df[col].replace(val, colMean)

        return df

    def replaceNanWithString(self, df, columns: np.ndarray):
        for col in columns:
            strNan = 'nan'
            df[col] = df[col].replace(np.nan, strNan)
        return df

    def removeColumns(self, transformedDataFrame, columnsToRemove):
        return transformedDataFrame.drop(columns=columnsToRemove) # Kinda silly to put in another function but ill leave it for consistency

    def binNumericColumnsByStdRanges(self, df, columns: np.ndarray):
        # only need this for one column atm # todo abstract to multiple for pipeline
        std = df["age"].std()
        mean = df["age"].mean()
        min = df["age"].min()
        max = df["age"].max()
        binRanges = [
            min,
            mean - 2 * std,
            mean - 1 * std,
            mean - 0.5 * std,
            mean,
            mean + 0.5 * std,
            mean + 1 * std,
            mean + 2 * std,
            max
        ]
        df["age"] = pd.cut(df["age"], bins=binRanges) # might add labels some how here in the future but for now this is getting one hot encoded anyway
        
        return df
        


class DataOrchestrator:
    def __init__(self, dataSource, dataSourceType: str, transformationFile:str):
        self.metadata = {
            "name": "Data Orchestrator",
            "description": "This will be moved to an abstract class that uses dependency injection in the future to allow for reuse across many use cases"
        }
        self.dataFrame = pd.DataFrame() # to be overridden in the load_data function
        self.source = dataSource
        self.dataSourceType = dataSourceType
        self.transformationFile = transformationFile
        self.load_data()
        self.dataTransformer = DataTransformer(self.dataFrame)
        
    def load_data(self):
        if self.dataSourceType == "csvFilePath" or self.dataSourceType == "csv":
            # cvs implementation for now but will make this abstract and dependent on a dataLoader implementation
            self.dataFrame = pd.read_csv(self.source, header=0) 
            
        if self.dataSourceType == "pandasDataFrame" or self.dataSourceType == "pd":
            self.dataFrame = self.source
        
    def clean_data(self):
        # implement later, luckily the datasets used so far have been clean or cleaning as acceptable to be in the transformer
        pass

    def get_transformed_data(self, model:str):
        # todo add validation to handle case were dataframe may not be set if model input doesnt match case
        # todo switch to match case statement after upgrade from python 3.9
        if model == 'logisticReg':
            dataFrame = self.dataTransformer.logisticModelDataFrame
        elif model == 'logisticRegWithAgeBinning':
            dataFrame = self.dataTransformer.logisticModelWithAgeBinningDataFrame
        elif model == 'decisionTree':
            dataFrame = self.dataTransformer.decisionTreeDataFrame
        elif model == 'neuralNetwork':
            dataFrame = self.dataTransformer.neuralNetworkDataFrame
            
        # pull out the target column (RN only using this for purchase project refactor in future to have this column defined upstream)
        X = dataFrame.drop(columns=["click"])
        y = dataFrame["click"]
        return X, y
        
    def build_test_train_split(self, model:str):
        X, y = self.get_transformed_data(model)
        return train_test_split(X, y, test_size=0.15)
    
    # helper/ functions that can be deleted in the future
    
    def print_Data_Short_Summary_View(self):
        print("Record Preview:")
        print(self.dataFrame.head(5))
        print(f"\n Shape: {self.dataFrame.shape}")
        print("\n Column Types:")
        print(self.dataFrame.dtypes)
        print("\n Numerical Summary:")
        print(self.dataFrame.describe())
        print("\n Memory Usage:")
        print(self.dataFrame.memory_usage(deep=True).sum() / 1024**2, "MB")
        print("\n Duplicate Rows:", self.dataFrame.duplicated().sum())

    def print_Data_Post_Transformation_View(self):
        print("\n Logistic Model Transformed Data Record Preview:")
        print(self.dataTransformer.logisticModelDataFrame.head(5))
        print("\n Column Types:")
        print(self.dataTransformer.logisticModelDataFrame.dtypes)
        print("\n Numerical Summary:")
        print(self.dataTransformer.logisticModelDataFrame.describe())
        print("\n Memory Usage:")
        print(self.dataTransformer.logisticModelDataFrame.memory_usage(deep=True).sum() / 1024**2, "MB")
        print("\n Duplicate Rows:", self.dataTransformer.logisticModelDataFrame.duplicated().sum())
        
    # second view to play with such that im not messing with the summary view intended for the full pipeline run or other views
    def print_Data_Verboise_Summary(self):
        print("Record Preview:")
        pd.set_option('display.max_colwidth', None)
        print(self.dataFrame.head(5))
        print(f"\nShape: {self.dataFrame.shape}")
        print("\nColumn Types:")
        print(self.dataFrame.dtypes)
        print("\nNumerical Summary:")
        print(self.dataFrame.describe())
        print("\nMemory Usage:")
        print(self.dataFrame.memory_usage(deep=True).sum() / 1024**2, "MB")
    
import pandas as pd
import numpy  as np
import json
import pandas as np
import importlib

class TransformerPipeline:
    def __init__(self):
        self.transformers = []
        
    def addTransformer(self, column, transformer):
        

    def load_class(module_path: str, class_name: str):
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls

class DataTransformer:
    def __init__(self, transformationFile:str):
        self.metadata = {
            "name": "Data Transformer",
            "description": "This class provides data transformation operations and to columns in a dataset"
        }
        
        self.transformations = transformationFile # default to none
        self.transformationPipeline = TransformerPipeline()
        # convert transformation settings stored under project specific files in data domain to np array 
        json_file_path = transformationFile
        with open(json_file_path, 'r') as f:
            transformations = np.array(json.load(f)) # format "model" {"columnToApplyTo" : "transformation"}
            # need a per model basis here to account for differences in how data is used 
            # i.e. decision trees split on features where neural networks use features for calculations and would be affected by one hot encoding
        self.transformations = transformations
        self.pipeline = TransformerPipeline()
        
    def transformData(self, model:str):
        self.transforedData = self.transformerPipeline.excute()
        pass
    
    def addTransformation(self, column: str, transformation: str):
        if transformation not in self.transformations[column]:
            self.transformations = np.append(self.transformations[column], transformation)
        

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
        
    def load_data(self):
        if self.dataSourceType == "csvFilePath" or self.dataSourceType == "csv":
            # cvs implementation for now but will make this abstract and dependent on a dataLoader implementation
            df = pd.read_csv(self.source)
            self.dataFrame = pd.read_csv(self.source)
            
        if self.dataSourceType == "pandasDataFrame" or self.dataSourceType == "pd":
            self.dataFrame = self.source
        
    def clean_data(self):
        
        pass

    def transform_data(self, model:str):
        
        pass
    
    # helper/ functions that can be deleted in the future
    
    def print_Data_Short_Summary_View(self):
        print("Record Preview:")
        print(self.dataFrame.head(5))
        print(f"\nShape: {self.dataFrame.shape}")
        print("\nColumn Types:")
        print(self.dataFrame.dtypes)
        print("\nNumerical Summary:")
        print(self.dataFrame.describe())
        print("\nMemory Usage:")
        print(self.dataFrame.memory_usage(deep=True).sum() / 1024**2, "MB")
        print("\nDuplicate Rows:", self.dataFrame.duplicated().sum())

    # second view to play with such that im not messing with the summary view intended for the full pipeline run
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
        print("\nDuplicate Rows:", self.dataFrame.duplicated().sum())
    
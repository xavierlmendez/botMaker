import os
import sys
from os.path import isfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './mlDomain')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './mathDomain')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './dataDomain')))

from dataDomain.DataOrchestrator import DataOrchestrator

# import kagglehub
# from kagglehub import KaggleDatasetAdapter
class PurchasePredictionModelBuilder:
    def __init__(self):
        self.projectName = "PurchasePrediction" # Used to get other project specific files in downstream procecsses
        self.dataFilePath = "./shopping_behavior_updated.csv"  
        self.modelMetaData = {} # This function will contain the end results of each model to be used on the frontend
        self.models = {}
        self.metadata = {
            "name": "Purchase Prediction Model Builder and Evaluator",
            "description": "ML 2025 Course Project looking at the use of different models in application to a problem"
        }

        # Ran into issues with the kaggle api so ended up manually downloading csv 
        #if not isfile(self.dataFilePath):
        #    kagglehub.dataset_download("ranaghulamnabi/shopping-behavior-and-preferences-study")
        
        self.dataOrchestrator = DataOrchestrator(self.dataFilePath, 'csv', 'Purchase')
        
        
    def buildModels(self):
        # Logistic Regression
        
        # Decision Tree
        
        # Neural Network
        pass
    
    def compileModelComparison(self):
        pass

    def printModelComparison(self):
       pass
    
    def compileArtifactForWebApp(self):
        pass

modelBuilder = PurchasePredictionModelBuilder()
# modelBuilder.dataOrchestrator.print_Data_Short_Summary_View()
modelBuilder.dataOrchestrator.print_Data_Verboise_Summary()
modelBuilder.dataOrchestrator.transform_data() # using preset transformation future updates will add more automation
# modelBuilder.dataOrchestrator.print_Data_Short_Summary_View()
modelBuilder.buildModels()
modelBuilder.compileModelComparison()
modelBuilder.printModelComparison()
modelBuilder.compileArtifactForWebApp()
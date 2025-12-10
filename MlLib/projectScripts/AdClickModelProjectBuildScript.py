import os
import sys
from os.path import isfile

from sklearn.linear_model import LogisticRegression

# adding the packages for data math and ml domains
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

from dataDomain.DataOrchestrator import DataOrchestrator
from mlDomain.logisticRegression import MyLogisticRegression

# import kagglehub
# from kagglehub import KaggleDatasetAdapter
class AdClickPredictionModelBuilder:
    def __init__(self):
        self.projectName = "AdClickPrediction" # Used to get other project specific files in downstream procecsses
        self.dataFilePath = "../dataDomain/dataSets/ad_click_dataset.csv"  
        self.modelMetaData = {} # This function will contain the end results of each model to be used on the frontend
        self.models = {}
        self.metadata = {
            "name": "Ad Click Model Builder and Evaluator",
            "description": "ML 2025 Course Project looking at the use of different models in application to a problem"
        }

        # Ran into issues with the kaggle api so ended up manually downloading csv 
        #if not isfile(self.dataFilePath):
        #    kagglehub.dataset_download("ranaghulamnabi/shopping-behavior-and-preferences-study")
        
        self.dataOrchestrator = DataOrchestrator(self.dataFilePath, 'csv', 'Purchase')
        
        
    def buildModels(self):
        
        # logisticReg decisionTree neuralNetwork
        print("\n Building Models...")
        
        # Logistic Regression
        logisticModel = MyLogisticRegression()
        logisticModel.gridFit(*self.dataOrchestrator.build_test_train_split('logisticReg'))
        logisticModel.evaluator.printEvaluation()

        # Decision Tree
        # tree_X_train, tree_X_test, tree_y_train, tree_y_test = self.dataOrchestrator.build_test_train_split('decisionTree')
        # treeModel = MyDecisionTree()
        # treeModel.fit(tree_X_train, tree_y_train)
        # treeModelEval = treeModel.evaluate(tree_X_test, tree_y_test)
        
        # Neural Network
        # Decision Tree
        # neural_X_train, neural_X_test, neural_y_train, neural_y_test = self.dataOrchestrator.build_test_train_split('neuralNetwork')
        # neuralModel = MyNeuralNetwork()
        # neuralModel.fit(neural_X_train, neural_y_train)
        # neuralModelEval = treeModel.evaluate(neural_X_test, neural_y_test)
    
    def compileModelComparison(self):
        pass

    def printModelComparison(self):
       pass
    
    def compileArtifactForWebApp(self):
        pass

modelBuilder = AdClickPredictionModelBuilder()
# modelBuilder.dataOrchestrator.print_Data_Short_Summary_View()
modelBuilder.dataOrchestrator.print_Data_Verboise_Summary()
# modelBuilder.dataOrchestrator.transform_data() # currently triggered on data orchestrator init using preset transformations future updates will add more automation and control
# rereview data post transformation
modelBuilder.dataOrchestrator.print_Data_Post_Transformation_View()
modelBuilder.buildModels()
modelBuilder.compileModelComparison()
modelBuilder.printModelComparison()
modelBuilder.compileArtifactForWebApp()
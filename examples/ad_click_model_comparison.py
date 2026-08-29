from pathlib import Path

from mllib.data.data_orchestrator import DataOrchestrator
from mllib.ml.projects.ad_click_logistic_regression import (
    LogisticRegression,
    LogisticRegressionWithAgeBinning,
)


# import kagglehub
# from kagglehub import KaggleDatasetAdapter
class AdClickPredictionModelBuilder:
    """ML 2025 Course Project looking at the use of different models in application to a problem"""

    def __init__(self):
        self.project_name = (
            "AdClickPrediction"  # Used to get other project specific files in downstream procecsses
        )
        self.data_file_path = str(
            Path(__file__).resolve().parents[1] / "data" / "ad_click_dataset.csv"
        )
        self.model_meta_data = {}  # This function will contain the end results of each model to be used on the frontend
        self.models = {}

        # Ran into issues with the kaggle api so ended up manually downloading csv
        # if not isfile(self.dataFilePath):
        #    kagglehub.dataset_download("ranaghulamnabi/shopping-behavior-and-preferences-study")

        config_path = str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "configs"
            / "ad_click_transformations.json"
        )
        self.data_orchestrator = DataOrchestrator(self.data_file_path, "csv", config_path)

    def build_models(self):

        # logisticReg decisionTree neuralNetwork
        print("\n Building Models...")

        # Logistic Regression
        logistic_model = LogisticRegression()
        logistic_model.grid_fit(*self.data_orchestrator.get_transformed_data("logisticReg"))
        logistic_model.evaluator.print_evaluation(print_best_model_stats_only=True)

        # Logistic Regression with binning age
        logistic2_model = LogisticRegressionWithAgeBinning()
        logistic2_model.grid_fit(
            *self.data_orchestrator.get_transformed_data("logisticRegWithAgeBinning")
        )
        logistic2_model.evaluator.print_evaluation(print_best_model_stats_only=True)

        # Decision Tree
        # tree_X_train, tree_X_test, tree_y_train, tree_y_test = self.dataOrchestrator.build_test_train_split('decisionTree')
        # treeModel = DecisionTree()
        # treeModel.fit(tree_X_train, tree_y_train)
        # treeModel.evaluate(tree_X_test, tree_y_test)
        # treeModel.evaluator.printEvaluation(printBestModelStatsOnly=True)

    # Neural Network
    # Decision Tree
    # neural_X_train, neural_X_test, neural_y_train, neural_y_test = self.dataOrchestrator.build_test_train_split('neuralNetwork')
    # neuralModel = MyNeuralNetwork()
    # neuralModel.fit(neural_X_train, neural_y_train)
    # neuralModelEval = treeModel.evaluate(neural_X_test, neural_y_test)

    def compile_model_comparison(self):
        pass

    def print_model_comparison(self):
        pass

    def compile_artifact_for_web_app(self):
        pass


model_builder = AdClickPredictionModelBuilder()
# modelBuilder.dataOrchestrator.print_Data_Short_Summary_View()
# modelBuilder.dataOrchestrator.print_Data_Verboise_Summary()
# modelBuilder.dataOrchestrator.transform_data() # currently triggered on data orchestrator init using preset transformations future updates will add more automation and control
# rereview data post transformation
# modelBuilder.dataOrchestrator.print_Data_Post_Transformation_View()
model_builder.build_models()
model_builder.compile_model_comparison()
model_builder.print_model_comparison()
model_builder.compile_artifact_for_web_app()

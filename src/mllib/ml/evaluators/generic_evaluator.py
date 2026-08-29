from json import dumps

import numpy as np
from numpy import ndarray


class ModelEvaluator:
    def __init__(self):
        self.metadata = {
            "name": "Model Evaluator Base Class",
            "description": "Base evaluator for computing and persisting model evaluation metrics.",
        }
        # TODO(BL-16): derive metadata by introspection
        self.run_iteration = 0
        self.evaluation_record: dict[int, dict] = {}

    def update_testing_prediction_data(
        self,
        test_values: ndarray,
        test_targets: ndarray,
        predictions: ndarray,
        evaluation_meta_data,
    ):
        self.run_iteration += 1
        self.test_values = test_values
        # Own copies: callers may pass read-only views (pandas >= 3 copy-on-write `.to_numpy()`),
        # and setConfusionMatrixValues normalises labels in place.
        self.test_targets = np.array(test_targets, copy=True)
        self.predictions = np.array(predictions, copy=True)
        self.evaluation_meta_data = evaluation_meta_data
        self.correct_predictions = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.evaluate_model()
        self.persist_evaluation_record()

    def set_confusion_matrix_values(self):
        # Labels are normalised so {-1, 0} both mean "negative" (np.sign classifiers emit -1).
        self.predictions[self.predictions == -1] = 0
        self.test_targets[self.test_targets == -1] = 0
        targets, predictions = self.test_targets, self.predictions
        self.true_positives = int(((targets == 1) & (predictions == 1)).sum())
        self.false_positives = int(((targets == 0) & (predictions == 1)).sum())
        self.true_negatives = int(((targets == 0) & (predictions == 0)).sum())
        self.false_negatives = int(((targets == 1) & (predictions == 0)).sum())

    def get_accuracy(self):
        correct_predictions = 0
        count_total_predictions = self.predictions.size

        for i in range(count_total_predictions):
            if self.test_targets[i] == self.predictions[i]:
                correct_predictions += 1

        # set to self to reuse for future calculations that would run after this
        self.correct_predictions = correct_predictions
        return correct_predictions / count_total_predictions

    def persist_evaluation_record(self):
        # Base record: the confusion-matrix counts every evaluator shares. Subclasses extend
        # this dict with their own metrics (see LogisticRegressionModelEvaluator).
        self.evaluation_record[self.run_iteration] = {
            "modelData": self.evaluation_meta_data,
            "correctPredictions": self.correct_predictions,
            "truePositives": self.true_positives,
            "falsePositives": self.false_positives,
            "trueNegatives": self.true_negatives,
            "falseNegatives": self.false_negatives,
        }

    def print_evaluation(self, print_best_model_stats_only=False):
        if not print_best_model_stats_only:
            formatted_eval_json = dumps(self.evaluation_record, indent=4)
            print(formatted_eval_json)
        self.print_evaluation_stats()

    def print_evaluation_stats(self):
        eval_data = self.evaluation_record

        best_accuracy = {"iteration": None, "value": float("-inf")}
        best_precision = {"iteration": None, "value": float("-inf")}
        best_recall = {"iteration": None, "value": float("-inf")}

        for iteration, metrics in self.evaluation_record.items():
            accuracy = metrics.get("accuracy")
            precision = metrics.get("precision")
            recall = metrics.get("recall")

            if accuracy > best_accuracy["value"]:
                best_accuracy = {"iteration": iteration, "value": accuracy}

            if precision > best_precision["value"]:
                best_precision = {"iteration": iteration, "value": precision}

            if recall > best_recall["value"]:
                best_recall = {"iteration": iteration, "value": recall}
        model_name = self.evaluation_meta_data["modelName"]
        print(f"\n Evaluation Summary : {model_name} ")

        print(
            f"\tBest Accuracy : {best_accuracy['value']:.4f} (Iteration {best_accuracy['iteration']})"
        )

        print(
            f"\tBest Precision: {best_precision['value']:.4f} "
            f"\t(Iteration {best_precision['iteration']})"
        )

        print(
            f"\tBest Recall   : {best_recall['value']:.4f} \t(Iteration {best_recall['iteration']})"
        )
        best_model_iterations = [
            best_accuracy["iteration"]
        ]  # [bestAccuracy['iteration'], bestPrecision['iteration'], bestRecall['iteration']]
        best_model_iterations = list(
            set(best_model_iterations)
        )  # remove duplicate if the same model is best for multiple metrics

        for iteration in best_model_iterations:
            print(f"\n (Iteration {iteration})")
            model_iteration = self.evaluation_record.get(iteration)
            formatted_eval_json = dumps(model_iteration, indent=4)
            print(formatted_eval_json)

    def evaluate_model(self):
        raise NotImplementedError("Subclasses must implement evaluateModel()")


class LogisticRegressionModelEvaluator(ModelEvaluator):
    def __init__(self):
        super().__init__()
        self.metadata = {
            "name": "Logistic Regression Evaluator",
            "description": "Evaluator for logistic regression classification metrics.",
        }
        # TODO(BL-16): derive metadata by introspection
        self.evaluation_meta_data = None

    def evaluate_model(self):
        self.set_confusion_matrix_values()
        self.accuracy = self.get_accuracy()
        self.precision = self.get_precision()
        self.recall = self.get_recall()

    def get_precision(self):
        return self.true_positives / (
            (self.true_positives + self.false_positives) or 1
        )  # Account for divide by zero

    def get_recall(self):
        return self.true_positives / ((self.true_positives + self.false_negatives) or 1)

    def get_mse(self):
        pass

    def persist_evaluation_record(self):
        # since the json package cant serialize python class objects we'll replace the class object with a class name str
        parsed_meta_data = dict(self.evaluation_meta_data)

        for potential_class_object in parsed_meta_data:
            if hasattr(parsed_meta_data[potential_class_object], "__class__") and not isinstance(
                parsed_meta_data[potential_class_object], (str, int, float, bool, list, dict)
            ):
                parsed_meta_data[potential_class_object] = parsed_meta_data[
                    potential_class_object
                ].__class__.__name__

        self.evaluation_record[self.run_iteration] = {
            "modelData": parsed_meta_data,
            "correctPredictions": self.correct_predictions,
            "truePositives": self.true_positives,
            "falsePositives": self.false_positives,
            "trueNegatives": self.true_negatives,
            "falseNegatives": self.false_negatives,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
        }

    def class_object_deserializer(self):

        return self.parsed_meta_data


class DecisionTreeModelEvaluator(ModelEvaluator):
    def __init__(self):
        super().__init__()
        self.metadata = {
            "name": "Decision Tree Evaluator",
            "description": "Evaluator for decision tree classification metrics.",
        }
        # TODO(BL-16): derive metadata by introspection
        self.evaluation_meta_data = None

    def evaluate_model(self):
        self.set_confusion_matrix_values()
        self.accuracy = self.get_accuracy()
        self.precision = self.get_precision()
        self.recall = self.get_recall()

    def get_accuracy(self):
        correct_predictions = 0
        count_total_predictions = self.predictions.__len__()

        for i in range(count_total_predictions):
            if self.test_targets[i] == self.predictions[i]:
                correct_predictions += 1

        # set to self to reuse for future calculations that would run after this
        self.correct_predictions = correct_predictions
        return correct_predictions / count_total_predictions

    def get_precision(self):
        return self.true_positives / (
            (self.true_positives + self.false_positives) or 0.000000001
        )  # Account for divide by zero

    def get_recall(self):
        return self.true_positives / ((self.true_positives + self.false_negatives) or 0.000000001)

    def get_mse(self):
        pass

    def persist_evaluation_record(self):
        # since the json package cant serialize python class objects we'll replace the class object with a class name str
        parsed_meta_data = dict(self.evaluation_meta_data)
        if "lossFunction" in parsed_meta_data:
            parsed_meta_data["lossFunction"] = parsed_meta_data["lossFunction"].__class__.__name__

        self.evaluation_record[self.run_iteration] = {
            "modelData": parsed_meta_data,
            "correctPredictions": self.correct_predictions,
            "truePositives": self.true_positives,
            "falsePositives": self.false_positives,
            "trueNegatives": self.true_negatives,
            "falseNegatives": self.false_negatives,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
        }

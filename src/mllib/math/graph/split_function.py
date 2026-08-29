import pandas as pd


class SplitFunction:
    def __init__(self):
        self.metadata = {
            "name": "Split Function Base Class",
            "description": "Base class for decision tree split criteria implementations.",
        }
        # TODO(BL-16): derive metadata by introspection

    def calculate_split(self, data_values, data_targets):
        return "class"

    def class_probabilities(self, column_values: pd.DataFrame, data_targets):
        classes_in_column = column_values.unique()
        class_probabilities: dict[str, float] = {}

        for unique_class in classes_in_column:
            record_criteria = column_values == unique_class
            total = column_values.count()
            num_targets_with_class = data_targets[record_criteria].sum()

            if num_targets_with_class != 0:
                probability_of_class = num_targets_with_class / total
                class_probabilities[unique_class] = probability_of_class

        return class_probabilities


class GiniImpurity(SplitFunction):
    metadata = {
        "name": "Gini Impurity",
        "description": "Split function using Gini impurity to choose the best feature.",
    }

    # TODO(BL-16): derive metadata by introspection
    def calculate_gini_impurities(self, data_values, data_targets):
        columns = data_values.columns
        gini_impurities: dict[str, int] = {}

        for column_name in columns:
            column = data_values[column_name]
            class_probabilities = self.class_probabilities(column, data_targets)
            gini_impurities[column_name] = 1 - sum(
                p**2 for p in class_probabilities.values()
            )  # 1 - summation of class probabilits squard is gini impurity formula

        return gini_impurities

    def calculate_split(self, data_values, data_targets):
        gini_impurities = self.calculate_gini_impurities(data_values, data_targets)
        return max(
            gini_impurities, key=lambda col: abs(gini_impurities[col] - 0.5)
        )  # the value furthest from .5 provides the most information


class InformationGain(SplitFunction):
    def __init__(self):
        self.metadata = {
            "name": "Information Gain",
            "description": "Split function placeholder for information gain.",
        }
        # TODO(BL-16): derive metadata by introspection


class ChiSquare(SplitFunction):
    def __init__(self):
        self.metadata = {
            "name": "Chi Square",
            "description": "Split function placeholder for chi-square based splitting.",
        }
        # TODO(BL-16): derive metadata by introspection

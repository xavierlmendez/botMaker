import numpy as np
import pandas as pd
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
    """This class provides data transformation operations and to columns in a dataset"""

    def __init__(self, df: pd.DataFrame):  # , transformationFile:str):

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

        # TODO(BL-09): replaced by the declarative TransformerPipeline
        self.build_transformed_dataframes(df)

    def transform_data(self, model: str):
        self.transfored_data = self.transformer_pipeline.excute()
        pass

    def add_transformation(self, column: str, transformation: str):
        if transformation not in self.transformations[column]:
            self.transformations = np.append(self.transformations[column], transformation)

    # TODO(BL-09): finish the declarative TransformerPipeline
    def build_transformed_dataframes(self, df: DataFrame):
        self.logistic_model_data_frame = self.temp_logistic_reg_model_transformer(df)
        self.logistic_model_with_age_binning_data_frame = (
            self.temp_logistic_reg_model_with_age_binning_transformer(df)
        )
        self.decision_tree_data_frame = self.temp_decision_tree_transformer(df)
        self.neural_network_data_frame = self.temp_neural_network_model_transformer(df)

    def temp_logistic_reg_model_transformer(self, df: DataFrame):
        transformed_data_frame = df

        all_columns_for_easy_reference = np.array(
            [
                "id",
                "full_name",
                "age",
                "gender",
                "device_type",
                "ad_position",
                "browsing_history",
                "time_of_day",
                "click",
            ]
        )

        columns_to_remove = np.array(
            [
                "id",
                "full_name",
            ]
        )
        transformed_data_frame = self.remove_columns(transformed_data_frame, columns_to_remove)

        columns_to_one_hot_encode = np.array(
            [
                "gender",
                "device_type",
                "ad_position",
                "browsing_history",
                "time_of_day",
            ]
        )
        transformed_data_frame = self.one_hot_encode_categorical_columns(
            transformed_data_frame, columns_to_one_hot_encode
        )

        columns_to_standardize = np.array(
            [
                "age",
            ]
        )
        transformed_data_frame = self.standardize_numeric_columns(
            transformed_data_frame, columns_to_standardize
        )

        values_to_replace_with_average = np.array(
            [
                np.nan,
            ]
        )
        transformed_data_frame = self.replace_values_with_numeric_avg(
            transformed_data_frame, columns_to_standardize, values_to_replace_with_average
        )

        return transformed_data_frame

    def temp_logistic_reg_model_with_age_binning_transformer(self, df: DataFrame):
        transformed_data_frame = df

        all_columns_for_easy_reference = np.array(
            [
                "id",
                "full_name",
                "age",
                "gender",
                "device_type",
                "ad_position",
                "browsing_history",
                "time_of_day",
                "click",
            ]
        )

        columns_to_remove = np.array(
            [
                "id",
                "full_name",
            ]
        )
        transformed_data_frame = self.remove_columns(transformed_data_frame, columns_to_remove)

        columns_to_standardize = np.array(
            [
                "age",
            ]
        )
        transformed_data_frame = self.standardize_numeric_columns(
            transformed_data_frame, columns_to_standardize
        )

        values_to_replace_with_average = np.array(
            [
                np.nan,
            ]
        )
        transformed_data_frame = self.replace_values_with_numeric_avg(
            transformed_data_frame, columns_to_standardize, values_to_replace_with_average
        )

        columns_to_bin = np.array(["age"])
        transformed_data_frame = self.bin_numeric_columns_by_std_ranges(
            transformed_data_frame, columns_to_bin
        )  # using ten bins for now

        columns_to_one_hot_encode = np.array(
            [
                "age",
                "gender",
                "device_type",
                "ad_position",
                "browsing_history",
                "time_of_day",
            ]
        )
        transformed_data_frame = self.one_hot_encode_categorical_columns(
            transformed_data_frame, columns_to_one_hot_encode
        )

        return transformed_data_frame

    def temp_decision_tree_transformer(self, df: DataFrame):
        transformed_data_frame = df

        all_columns_for_easy_reference = np.array(
            [
                "id",
                "full_name",
                "age",
                "gender",
                "device_type",
                "ad_position",
                "browsing_history",
                "time_of_day",
                "click",
            ]
        )

        transformed_data_frame = self.replace_nan_with_string(df, all_columns_for_easy_reference)

        columns_to_remove = np.array(
            [
                "id",
                "full_name",
            ]
        )
        transformed_data_frame = self.remove_columns(transformed_data_frame, columns_to_remove)

        columns_to_standardize = np.array(
            [
                "age",
            ]
        )
        transformed_data_frame = self.standardize_numeric_columns(
            transformed_data_frame, columns_to_standardize
        )

        values_to_replace_with_average = np.array(
            [
                np.nan,
            ]
        )
        transformed_data_frame = self.replace_values_with_numeric_avg(
            transformed_data_frame, columns_to_standardize, values_to_replace_with_average
        )

        columns_to_bin = np.array(["age"])
        transformed_data_frame = self.bin_numeric_columns_by_std_ranges(
            transformed_data_frame, columns_to_bin
        )  # using ten bins for now

        return transformed_data_frame

    def temp_neural_network_model_transformer(self, df: DataFrame):
        transformed_data_frame = df
        # TODO(BL-09): transformations move to config-driven transformer classes
        return transformed_data_frame

    # TODO(BL-09): migrate to the TransformerPipeline
    def one_hot_encode_categorical_columns(self, df, columns: np.ndarray, as_boolean=False):
        dtype_option = bool if as_boolean else float
        # Data set for click prediction project has blanks so setting dummy_na to true here
        df = pd.get_dummies(df, columns=columns, drop_first=True, dummy_na=True, dtype=dtype_option)
        return df

    def standardize_numeric_columns(self, df, columns: np.ndarray):
        sc = StandardScaler()  # TODO(BL-09): hand-built encoder becomes a transformer class
        # to allow for signature of res = func(df, columnsToStandardize) and in place scaling + other scaling options in the signature
        df[columns] = sc.fit_transform(df[columns])
        return df

    def replace_values_with_numeric_avg(self, df, columns: np.ndarray, values: np.ndarray):
        for col in columns:
            col_mean = df[col].mean()
            for val in values:
                df[col] = df[col].replace(val, col_mean)

        return df

    def replace_nan_with_string(self, df, columns: np.ndarray):
        for col in columns:
            str_nan = "nan"
            df[col] = df[col].replace(np.nan, str_nan)
        return df

    def remove_columns(self, transformed_data_frame, columns_to_remove):
        return transformed_data_frame.drop(
            columns=columns_to_remove
        )  # Kinda silly to put in another function but ill leave it for consistency

    def bin_numeric_columns_by_std_ranges(self, df, columns: np.ndarray):
        # only need this for one column atm # TODO(BL-09): generalise to many columns in the pipeline
        std = df["age"].std()
        mean = df["age"].mean()
        min = df["age"].min()
        max = df["age"].max()
        bin_ranges = [
            min,
            mean - 2 * std,
            mean - 1 * std,
            mean - 0.5 * std,
            mean,
            mean + 0.5 * std,
            mean + 1 * std,
            mean + 2 * std,
            max,
        ]
        df["age"] = pd.cut(
            df["age"], bins=bin_ranges
        )  # might add labels some how here in the future but for now this is getting one hot encoded anyway

        return df


class DataOrchestrator:
    """Loads a dataset, applies the project transformations, and produces train/test splits. To
    become an injectable pipeline (BL-09).
    """

    def __init__(self, data_source, data_source_type: str, transformation_file: str):
        self.data_frame = pd.DataFrame()  # to be overridden in the load_data function
        self.source = data_source
        self.data_source_type = data_source_type
        self.transformation_file = transformation_file
        self.load_data()
        self.data_transformer = DataTransformer(self.data_frame)

    def load_data(self):
        if self.data_source_type == "csvFilePath" or self.data_source_type == "csv":
            # cvs implementation for now but will make this abstract and dependent on a dataLoader implementation
            self.data_frame = pd.read_csv(self.source, header=0)

        if self.data_source_type == "pandasDataFrame" or self.data_source_type == "pd":
            self.data_frame = self.source

    def clean_data(self):
        # implement later, luckily the datasets used so far have been clean or cleaning as acceptable to be in the transformer
        pass

    def get_transformed_data(self, model: str):
        # TODO(BL-09): model-name ladder retired with the pipeline
        # TODO(BL-09): model-name ladder retired with the pipeline
        if model == "logisticReg":
            data_frame = self.data_transformer.logistic_model_data_frame
        elif model == "logisticRegWithAgeBinning":
            data_frame = self.data_transformer.logistic_model_with_age_binning_data_frame
        elif model == "decisionTree":
            data_frame = self.data_transformer.decision_tree_data_frame
        elif model == "neuralNetwork":
            data_frame = self.data_transformer.neural_network_data_frame

        # pull out the target column (RN only using this for purchase project refactor in future to have this column defined upstream)
        X = data_frame.drop(columns=["click"])
        y = data_frame["click"]
        return X, y

    def build_test_train_split(self, model: str):
        X, y = self.get_transformed_data(model)
        return train_test_split(X, y, test_size=0.20)

    # helper/ functions that can be deleted in the future

    def print_data_short_summary_view(self):
        print("Record Preview:")
        print(self.data_frame.head(5))
        print(f"\n Shape: {self.data_frame.shape}")
        print("\n Column Types:")
        print(self.data_frame.dtypes)
        print("\n Numerical Summary:")
        print(self.data_frame.describe())
        print("\n Memory Usage:")
        print(self.data_frame.memory_usage(deep=True).sum() / 1024**2, "MB")
        print("\n Duplicate Rows:", self.data_frame.duplicated().sum())

    def print_data_post_transformation_view(self):
        print("\n Logistic Model Transformed Data Record Preview:")
        print(self.data_transformer.logistic_model_data_frame.head(5))
        print("\n Column Types:")
        print(self.data_transformer.logistic_model_data_frame.dtypes)
        print("\n Numerical Summary:")
        print(self.data_transformer.logistic_model_data_frame.describe())
        print("\n Memory Usage:")
        print(
            self.data_transformer.logistic_model_data_frame.memory_usage(deep=True).sum() / 1024**2,
            "MB",
        )
        print(
            "\n Duplicate Rows:", self.data_transformer.logistic_model_data_frame.duplicated().sum()
        )

    # second view to play with such that im not messing with the summary view intended for the full pipeline run or other views
    def print_data_verboise_summary(self):
        print("Record Preview:")
        pd.set_option("display.max_colwidth", None)
        print(self.data_frame.head(5))
        print(f"\nShape: {self.data_frame.shape}")
        print("\nColumn Types:")
        print(self.data_frame.dtypes)
        print("\nNumerical Summary:")
        print(self.data_frame.describe())
        print("\nMemory Usage:")
        print(self.data_frame.memory_usage(deep=True).sum() / 1024**2, "MB")

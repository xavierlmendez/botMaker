import pandas as pd
from sklearn.model_selection import train_test_split

from mllib.data.pipeline import ProjectTransformations


class DataOrchestrator:
    """Loads a dataset and builds one transformed frame per pipeline declared in the project's
    transformer config (``data/configs/*.json``), then hands out feature/target pairs by frame name.
    Adding a frame or a transformation is a config edit (R1, BL-09).
    """

    def __init__(self, data_source, data_source_type: str, transformation_file: str):
        self.data_frame = pd.DataFrame()  # to be overridden in the load_data function
        self.source = data_source
        self.data_source_type = data_source_type
        self.transformation_file = transformation_file
        self.load_data()
        self.transformations = ProjectTransformations.from_file(transformation_file)
        self.frames = {
            name: pipeline.fit_transform(self.data_frame)
            for name, pipeline in self.transformations.pipelines.items()
        }

    def load_data(self):
        if self.data_source_type == "csvFilePath" or self.data_source_type == "csv":
            # cvs implementation for now but will make this abstract and dependent on a dataLoader implementation
            self.data_frame = pd.read_csv(self.source, header=0)

        if self.data_source_type == "pandasDataFrame" or self.data_source_type == "pd":
            self.data_frame = self.source

    def clean_data(self):
        # implement later, luckily the datasets used so far have been clean or cleaning as acceptable to be in the transformer
        pass

    def get_transformed_data(self, frame_name: str):
        """Features and target for a named frame from the project config."""
        if frame_name not in self.frames:
            raise KeyError(
                f"unknown frame {frame_name!r}; configured: {self.transformations.frame_names()}"
            )
        frame = self.frames[frame_name]
        target = self.transformations.target
        return frame.drop(columns=[target]), frame[target]

    def build_test_train_split(self, frame_name: str):
        X, y = self.get_transformed_data(frame_name)
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

    def print_data_post_transformation_view(self, frame_name: str = "logisticReg"):
        frame = self.frames[frame_name]
        print(f"\n {frame_name} Transformed Data Record Preview:")
        print(frame.head(5))
        print("\n Column Types:")
        print(frame.dtypes)
        print("\n Numerical Summary:")
        print(frame.describe())
        print("\n Memory Usage:")
        print(frame.memory_usage(deep=True).sum() / 1024**2, "MB")
        print("\n Duplicate Rows:", frame.duplicated().sum())

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

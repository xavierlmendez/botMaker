class BaseTransformer:
    """Abstract base class for column/data transformers."""

    name = None

    def transform(self, series):
        raise NotImplementedError

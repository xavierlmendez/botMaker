class BaseTransformer:
    name = None
    metadata = {
        "name": "Base Transformer",
        "description": "Abstract base class for column/data transformers."
    }
    # TODO: review metadata (auto-generated)
    def transform(self, series):
        raise NotImplementedError

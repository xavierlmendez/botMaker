class BaseTransformer:
    name = None
    metadata = {
        "name": "Base Transformer",
        "description": "Abstract base class for column/data transformers.",
    }

    # TODO(BL-16): derive metadata by introspection
    def transform(self, series):
        raise NotImplementedError

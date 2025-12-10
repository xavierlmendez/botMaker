class BaseTransformer:
    name = None
    def transform(self, series):
        raise NotImplementedError
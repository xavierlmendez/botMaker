class GaussianPrior:  # add prior class as injection to allow different implementations of prior
    """A class to provide a Gaussian Prior and have related helper functions"""

    def __init__(
        self, mean: float = 0, variance: float = 0
    ):  # reminder that mean is mu and variance is sigma squared
        self.mean = mean
        self.variance = variance

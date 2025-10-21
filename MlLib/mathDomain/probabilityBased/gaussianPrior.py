class GaussianPrior(Prior):
    def __init__(mean:float = , variance:float = ): #reminder that mean is mu and variance is sigma squared
        self.metadata = {
            "name": "Gaussian Based Prior Class",
            "description": "A class to provide a Gaussian Prior and have related helper functions"
        }
        self.mean = mean
        self.variance = float
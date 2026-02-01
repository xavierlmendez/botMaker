import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from  import prior
class GaussianPrior(): # add prior class as injection to allow different implementations of prior
    def __init__(self, mean:float = 0 , variance:float = 0): #reminder that mean is mu and variance is sigma squared
        self.metadata = {
            "name": "Gaussian Based Prior Class",
            "description": "A class to provide a Gaussian Prior and have related helper functions"
        }
        self.mean = mean
        self.variance = float
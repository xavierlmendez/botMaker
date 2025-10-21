import os
import sys

import numpy as np
import pandas as pd
import numpy as np
import sklearn
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './mlDomain')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './mathDomain')))

from mlDomain.linearRegression import MyLinearRegression
from mathDomain.lossFunction import MSE
from mathDomain.hypothesis import HypothesisFunction

from mathDomain.lossFunction import LossFunction
class BostonHousingDataset:
    def __init__(self):
        self.url = "http://lib.stat.cmu.edu/datasets/boston"
        self.feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

    def load_dataset(self):
        # Fetch data from URL
        raw_df = pd.read_csv(self.url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]

        # Create the dictionary in sklearn format
        dataset = {
            'data': [],
            'target': [],
            'feature_names': self.feature_names,
            'DESCR': 'Boston House Prices dataset'
        }

        dataset['data'] = data
        dataset['target'] = target

        return dataset
    
boston_housing = BostonHousingDataset()
boston_dataset = boston_housing.load_dataset()
boston_dataset.keys(), boston_dataset['DESCR']
boston = pd.DataFrame(boston_dataset['data'], columns=boston_dataset['feature_names'])
boston['MEDV'] = boston_dataset['target']
boston.head()

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

X = boston.to_numpy()
X = np.delete(X, 13, 1)
y = boston['MEDV'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Let us now evaluate on the test set
y_pred_test = lin_model.predict(X_test)
rmse_test = rmse(y_pred_test, y_test)
r2_test = r2_score(y_test, y_pred_test)
print("Test RMSE = " + str(rmse_test))
print("Test R2 = " + str(r2_test))

seededRand = np.random.default_rng(10) # seeting a seed for random initial weights after a few tests
initialWeights = seededRand.random(13)
initialBias = 0
hypothesisFunction = HypothesisFunction(initialWeights, initialBias)

lossFunction = MSE()
MSEBasedModel = MyLinearRegression(hypothesisFunction, lossFunction, 0.0000032, 15000)
MSEBasedModel.fit(X_train, y_train)

preditctedMSETest = MSEBasedModel.predictValues(X_test)
rmse_test = rmse(preditctedMSETest, y_test)
r2_test = r2_score(y_test, preditctedMSETest)
print("Test RMSE = " + str(rmse_test))
print("Test R2 = " + str(r2_test))
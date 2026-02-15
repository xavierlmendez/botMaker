#%%
import numpy as np
from numpy import linalg

testInitFeatures = np.array([1,1,1,1])
featureSpaceExpansion = testInitFeatures.reshape(-1, 1) @ testInitFeatures.reshape(1,-1)

testInitWeigths = np.array([1,1,1,1])
np.set_printoptions(precision=3)
# print(testInitWeigths)
hypothesisSpaceExpansion = testInitWeigths.reshape(-1, 1) @ testInitWeigths.reshape(1,-1)
# print(hypothesisSpaceExpansion)
hypothesisSpaceExpansion3rdDegree = hypothesisSpaceExpansion.reshape(-1, 1) @ hypothesisSpaceExpansion.reshape(1,-1)
# print(hypothesisSpaceExpansion3rdDegree)

result = featureSpaceExpansion.reshape(-1) @ hypothesisSpaceExpansion3rdDegree
# print(np.sum(result))



x = np.array([1,1,1,1])
tensor = np.outer(x, x)
w = np.array([1,1,1,1])
A = np.outer(w, w)
result = tensor.T @ A @ tensor
# print("test ")
# print(np.sum(result))
initialArray = np.array([1,1,1,1])
expandedHypothesis = np.ndarray(2 * initialArray.shape[0], dtype=float)
index = 0
for feature in initialArray:
    for degree in range(2):
        expandedHypothesis[index] = feature ** (degree + 1)  # for hypothesis function should result in the initial weights being scaled the same but will need to see if this affect model performance
        index += 1
print(expandedHypothesis)

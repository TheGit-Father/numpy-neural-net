import numpy as np
from sklearn.datasets import make_moons
from model import softmax

X = np.array([
     [2,1],
     [3,5]])

result = softmax(X)

print(result)




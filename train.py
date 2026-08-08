import numpy as np
from sklearn.datasets import make_moons
from model import cross_entropy

predictions = np.array([[0.8, 0.2],
                        [0.1, 0.9],
                        [0.7, 0.3],
                        [0.4, 0.6]])

labels = np.array([0, 1, 0, 1])

loss = cross_entropy(predictions, labels)

print("Loss :", loss)




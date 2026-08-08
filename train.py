import numpy as np
from sklearn.datasets import make_moons
from model import NeuralNetwork

X,y = make_moons(
  n_samples = 500,
  noise = 0.2,
  random_state = 42
)

model = NeuralNetwork()
loss = model.loss(X, y)

print("Loss :", loss)




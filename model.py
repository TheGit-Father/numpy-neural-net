import numpy as np

def relu(x):
  return np.maximum(0, x)

def softmax(X):
  X = np.exp(X)
  return X / np.sum(X, axis=1, keepdims=True)

class NeuralNetwork:
  def __init__(self):
    self.Linear1 = Linear(2, 16)
    self.Linear2 = Linear(16, 2)


  def forward(self, x):
    z1 = self.Linear1.forward(x)
    a1 = relu(z1)
    z2 = self.Linear2.forward(a1)
    return z2

class Linear:
  def __init__(self, in_features, out_features):
    self.W = np.random.randn(in_features, out_features)
    self.b = np.zeros(out_features)

  def forward(self, X):
    return X @ self.W + self.b


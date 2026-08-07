import numpy as np

class Linear:
  def __init__(self, in_features, out_features):
    self.W = np.random.randn(in_features, out_features)
    self.b = np.zeros(out_features)

  def forward(self, X):
    return X @ self.W + self.b


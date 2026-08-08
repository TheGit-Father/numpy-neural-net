import numpy as np

def relu(x):
  return np.maximum(0, x)

def softmax(X):
  X = X - np.max(X, axis=1, keepdims=True)
  exp = np.exp(X)
  return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(predictions, labels):
  n = len(labels)
  correct_probs = predictions[np.arange(n), labels]
  correct_probs = np.clip(correct_probs, 1e-15, 1.0)
  losses = - np.log(correct_probs)
  return  np.mean(losses)


class NeuralNetwork:
  def __init__(self):
    self.Linear1 = Linear(2, 16)
    self.Linear2 = Linear(16, 2)

  def forward(self, x):
    z1 = self.Linear1.forward(x)
    a1 = relu(z1)
    z2 = self.Linear2.forward(a1)
    return z2

  def loss(self, X, y):
    logits = self.forward(X)
    probabilities = softmax(logits)
    print("Logits shape :", logits.shape)
    print("Probab shape :", probabilities.shape)
    print("Labels shape :", y.shape)
    return cross_entropy(probabilities, y)

class Linear:
  def __init__(self, in_features, out_features):
    self.W = np.random.randn(in_features, out_features)
    self.b = np.zeros(out_features)

  def forward(self, X):
    return X @ self.W + self.b


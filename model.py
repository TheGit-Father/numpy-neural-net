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
    self.z1 = self.Linear1.forward(x)
    a1 = relu(self.z1)
    z2 = self.Linear2.forward(a1)
    return z2

  def loss(self, X, y):
    logits = self.forward(X)
    probabilities = softmax(logits)
    return cross_entropy(probabilities, y)

  def backward(self, probabilities, y):
    Y = np.zeros_like(probabilities)
    Y[np.arange(len(y)), y] = 1
    dz2 = (probabilities - Y) / len(y)
    dW2, db2, da1 = self.Linear2.backward(dz2)
    dz1 = da1 * (self.z1 > 0)
    dW1, db1, dX = self.Linear1.backward(dz1)
    return dW1, db1, dW2, db2

  def update(self, dW1, db1, dW2, db2, lr):
    self.Linear1.W -= lr * dW1
    self.Linear1.b -= lr * db1
    self.Linear2.W -= lr * dW2
    self.Linear2.b -= lr * db2

  def predict(self, X):
    logits = self.forward(X)
    probabilities = softmax(logits)
    return np.argmax(probabilities, axis=1)


class Linear:
  def __init__(self, in_features, out_features):
    self.W = np.random.randn(in_features, out_features)
    self.b = np.zeros(out_features)

  def forward(self, X):
    self.input = X
    return X @ self.W + self.b

  def backward(self, dz):
    dW = self.input.T @ dz
    db = np.sum(dz, axis=0)
    dX = dz @ self.W.T
    return dW, db, dX


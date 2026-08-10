from dataset import X,y
from model import NeuralNetwork
from model import softmax
import numpy as np

model = NeuralNetwork()

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
  #Forward pass
  logits = model.forward(X)
  probabilities = softmax(logits)

  #Loss
  loss = model.loss(X,y)

  #Backward Pass
  dW1, db1, dW2, db2 = model.backward(probabilities, y)

  #Update parameters
  model.update(dW1, db1, dW2, db2, learning_rate)

  if epoch % 100 == 0:
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")


import matplotlib.pyplot as plt

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]

grid_predictions = model.predict(grid)

grid_predictions = grid_predictions.reshape(xx.shape)

plt.contourf(xx, yy, grid_predictions, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Neural Network Decision Boundary")
plt.show()




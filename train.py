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




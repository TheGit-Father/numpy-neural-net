from dataset import X_train, y_train, X_test, y_test
from model import NeuralNetwork
from model import softmax
import numpy as np

model = NeuralNetwork()

learning_rate = 0.01
epochs = 200

for epoch in range(epochs):

  #Forward pass
  logits = model.forward(X_train)
  probabilities = softmax(logits)

  #Loss using already-computed probabilities
  correct_probs = probabilities[np.arange(len(y_train)), y_train]
  correct_probs = np.clip(correct_probs, 1e-25, 1.0)
  loss = -np.mean(np.log(correct_probs))

  #Backward Pass
  dW1, db1, dW2, db2 = model.backward(probabilities, y_train)

  #Update parameters
  model.update(dW1, db1, dW2, db2, learning_rate)

  if epoch % 10 == 0:
    predictions = np.argmax(probabilities, axis=1)
    accuracy = np.mean(predictions == y_train)

    print(
      f"Epoch {epoch}, "
      f"Loss: {loss:.4f}, "
      f"Accuracy: {accuracy * 100:.2f}%"
      )


# Final evaluation on the unseen test set

test_logits = model.forward(X_test)
test_probabilities = softmax(test_logits)

test_predictions = np.argmax(test_probabilities, axis=1)

test_accuracy = np.mean(test_predictions == y_test)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")



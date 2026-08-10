from dataset import X,y
print("X shape :", X.shape)
print("y shape :", y.shape)
from model import NeuralNetwork
from model import softmax

model = NeuralNetwork()
logits = model.forward(X)
probabilities = softmax(logits)

loss = model.loss(X, y)

print("Loss :", loss)

model.backward(probabilities, y)




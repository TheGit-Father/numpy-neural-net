import numpy as np
from model import Linear

X = np.random.randn(500, 2)

layer = Linear(2,4)

output = layer.forward(X)

print("Input shape :", X.shape)
print("Weight shape :", layer.W.shape)
print("Bias shape :", layer.b.shape)
print("Output shape :", output.shape)

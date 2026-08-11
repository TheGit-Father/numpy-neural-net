from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int64)

# Standard MNIST split
X_train = X[:60000]
y_train = y[:60000]

X_test = X[60000:]
y_test = y[60000:]




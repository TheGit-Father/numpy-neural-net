### Development Log

### Progress 1
Loaded the make_moons dataset using scikit-learn
Explored dataset structure and feature-label shapes
Visualized the non-linearly separable dataset

### Progress 2
Implemented a fully vectorized Linear layer using NumPy
Initialized weight matrices and bias vectors
Verified matrix multiplication and output dimensions
Introduced batched forward propagation with X @ W + b

### Progress 3
Built a fully vectorized two-layer neural network
Added a hidden layer with ReLU activation
Implemented the complete forward pass using NumPy
Verified output dimensions on the make_moons dataset
Produced raw logits for binary classification

#### Progress 4
Implemented the Softmax activation function using NumPy
Converted output logits into normalized probability distributions
Learned row-wise operations using axis=1 and keepdims=True
Verified probability outputs sum to 1 for every sample
Completed the inference pipeline for a feedforward neural network

### Progress 5
- Improved Softmax with a numerically stable implementation by subtracting the row-wise maximum before exponentiation to prevent overflow.

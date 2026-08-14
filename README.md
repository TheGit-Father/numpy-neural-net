# NumPy Neural Net

A feedforward neural network built from scratch using only NumPy — no PyTorch, no TensorFlow, no autograd library. Manual forward pass, manual backpropagation, manual gradient descent, trained on MNIST.

Reaches **78.6% test accuracy** after 200 epochs of full-batch gradient descent (still improving — loss was dropping steadily at cutoff, not yet converged).


## Results

| Metric | Value |
|---|---|
| Dataset | MNIST (60,000 train / 10,000 test) |
| Architecture | 784 → 128 (ReLU) → 10 (Softmax) |
| Epochs | 200, full-batch |
| Learning rate | 0.01 |
| Final train accuracy | 77.10% |
| Final test accuracy | **78.60%** |
| Final train loss | 1.1528 (cross-entropy) |


## Structure

```
dataset.py  — loads and normalizes MNIST via fetch_openml, standard 60k/10k split
model.py    — Linear layer, ReLU, numerically-stable Softmax, cross-entropy loss,
              full manual forward/backward pass for a 2-layer network
train.py    — training loop: forward, loss, backward, gradient descent update,
              logs loss/accuracy every 10 epochs, evaluates on held-out test set
```


## How it works

- **`Linear`** stores its own weight matrix and bias, does `X @ W + b` on the forward pass, and computes `dW`, `db`, and `dX` on the backward pass from the incoming gradient.
- **Forward pass**: `Linear1 → ReLU → Linear2 → Softmax`, producing class probabilities for each sample.
- **Loss**: cross-entropy on the correct-class probability, clipped to avoid `log(0)`.
- **Backward pass**: gradient of softmax + cross-entropy simplifies to `(probabilities - one_hot_labels) / n`; this is backpropagated through `Linear2`, masked by the ReLU derivative, then through `Linear1`.
- **Update**: vanilla gradient descent — no momentum, no Adam, no learning rate schedule.


## Milestones

- Loaded and visualized the `make_moons` dataset; explored feature/label shapes on a non-linearly separable toy problem
- Built a fully vectorized `Linear` layer in NumPy (`X @ W + b`), verified output dimensions
- Built a two-layer network with ReLU, producing raw logits for binary classification
- Implemented Softmax, verified row-wise outputs sum to 1
- Made Softmax numerically stable (subtract row-wise max before exponentiation)
- Implemented cross-entropy loss with probability clipping to avoid `log(0)`
- Implemented full backpropagation through both layers by hand
- Added the gradient descent training loop and a decision-boundary visualization on `make_moons`
- Ported the network to MNIST: `dataset.py` for loading/normalizing, full training and evaluation pipeline in `train.py`


## Next steps

- Train longer / raise the learning rate — loss curve had not plateaued at epoch 200
- Mini-batch instead of full-batch gradient descent

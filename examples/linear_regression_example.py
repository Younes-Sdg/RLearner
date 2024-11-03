# Example usage
import numpy as np
from rlearner.linear_model import LinearModelRL , Policy

np.random.seed(42)
X = np.random.randn(1000, 2)
true_weights = np.array([0.5, -0.8])
y = X.dot(true_weights) + np.random.randn(1000) * 0.1

 
print("\nGradient Policy:")
model_grad = LinearModelRL(learning_rate=0.1, policy=Policy.GRADIENT)
model_grad.fit(X, y)
print("True weights:", true_weights)
print("Found weights:", model_grad.weights)
print("Final MSE:", np.mean((model_grad.predict(X) - y) ** 2))

print("\nDirect RL Policy:")
model_direct = LinearModelRL(learning_rate=0.1, policy=Policy.DIRECT)
model_direct.fit(X, y)
print("True weights:", true_weights)
print("Found weights:", model_direct.weights)
print("Final MSE:", np.mean((model_direct.predict(X) - y) ** 2))
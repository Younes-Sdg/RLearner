import numpy as np
from rlearner.models.linear_model import LinearModelRL
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.flatten() + np.random.randn(100) * 2


model_rl = LinearModelRL(
    learning_rate=0.01,
    max_steps=5000,
    mse_threshold=0.001,
    n_features=X.shape[1]
)

# Fit the RL model
model_rl.fit(X, y)

# Make predictions with the RL model
predictions_rl = model_rl.predict(X)

# Calculate metrics for RL model
mse_rl = mean_squared_error(y, predictions_rl)
r2_rl = r2_score(y, predictions_rl)

print("Reinforcement Learning Model Results:")
print(f"MSE: {mse_rl:.4f}")
print(f"R² Score: {r2_rl:.4f}")
print("Model Weights:", model_rl.get_weights())

# Create and train the sklearn Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X, y)
predictions_lr = model_lr.predict(X)

# Calculate metrics for sklearn model
mse_lr = mean_squared_error(y, predictions_lr)
r2_lr = r2_score(y, predictions_lr)

print("\nLinear Regression Model Results:")
print(f"MSE: {mse_lr:.4f}")
print(f"R² Score: {r2_lr:.4f}")
print("Model Weights:", model_lr.coef_)

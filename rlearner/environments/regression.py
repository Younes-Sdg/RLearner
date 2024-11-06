import numpy as np
from rlearner.base import BaseEnvironment

class RegressionEnvironmentRL(BaseEnvironment):
    def __init__(self, mse_threshold: float, gamma: float = 0.99):
        """
        Initialize the regression environment.
        
        Args:
            mse_threshold (float): Threshold for mean squared error to consider problem solved
            gamma (float): Discount factor for rewards
        """
        self.mse_threshold = mse_threshold
        self.gamma = gamma
        self.weights = None
        self.X = None
        self.y = None
        self.best_mse = float('inf')
        self.best_weights = None
        self.episode_history = []
        
    def set_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Set the training data for the environment."""
        self.X = X
        self.y = y
        self.weights = np.random.randn(X.shape[1]) * 0.01
        self.best_weights = self.weights.copy()
        self.best_mse = self.get_mse()

    def get_mse(self) -> float:
        """Calculate the Mean Squared Error."""
        predictions = self.X.dot(self.weights)
        return np.mean((predictions - self.y) ** 2)

    def get_gradient(self) -> np.ndarray:
        """Calculate the gradient of MSE with respect to weights."""
        predictions = self.X.dot(self.weights)
        error = predictions - self.y
        return 2 * self.X.T.dot(error) / len(self.y)

    def step(self, action: np.ndarray) -> tuple[dict, float, bool]:
        """
        Perform a step in the environment given an action.
        
        Args:
            action (np.ndarray): Weight updates to apply
            
        Returns:
            tuple: (state, reward, done)
        """
        old_mse = self.get_mse()
        self.weights += action
        new_mse = self.get_mse()
        
        self.episode_history.append(new_mse)
        
        # simplified : avoid overflow
        improvement = old_mse - new_mse
        
        # Scale improvement to reasonable range
        reward = np.clip(improvement, -1.0, 1.0)
        
        # Add small bonus for finding new best MSE
        if new_mse < self.best_mse:
            reward += 0.1
            self.best_mse = new_mse
            self.best_weights = self.weights.copy()
        
        # Terminate if solved or diverging
        done = new_mse < self.mse_threshold or new_mse > 10 * old_mse
        
        state = {
            'weights': self.weights.copy(),
            'mse': new_mse,
            'gradient': self.get_gradient(),
            'best_mse': self.best_mse
        }
        
        return state, reward, done

    def reset(self) -> dict:
        """Reset the environment to initial state."""
        self.weights = np.random.randn(self.X.shape[1]) * 0.01 if self.X is not None else None
        self.episode_history = []
        
        return {
            'weights': self.weights.copy(),
            'mse': self.get_mse(),
            'gradient': self.get_gradient(),
            'best_mse': self.best_mse
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the current weights."""
        return X.dot(self.best_weights if self.best_weights is not None else self.weights)

    def get_weights(self) -> np.ndarray:
        """Get the current weights of the model."""
        return self.best_weights if self.best_weights is not None else self.weights
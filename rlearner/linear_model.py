import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import warnings

@dataclass
class PolicyGradientConfig:
    """Configuration class for PolicyGradientRegression"""
    n_features: int
    learning_rate: float = 0.01
    min_learning_rate: float = 1e-4
    initial_policy_std: float = 0.1
    min_policy_std: float = 0.01
    l2_penalty: float = 0.01
    batch_size: int = 32
    mse_threshold: float = 0.001

class RLEnvironment:
    """Environment that simulates the linear regression problem as an RL task"""
    def __init__(self, X: np.ndarray, y: np.ndarray, config: PolicyGradientConfig):
        self.X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)  # Normalize features
        self.y = y
        self.config = config
        self.reset_state()
        
    def reset_state(self) -> None:
        """Reset environment state"""
        self.current_weights = None
        self.current_mse = float('inf')
        self.best_weights = None
        self.best_mse = float('inf')
        
    def reset(self, n_features: int) -> np.ndarray:
        """Reset environment to initial state"""
        init_scale = np.sqrt(2.0 / n_features)
        self.current_weights = np.random.randn(n_features) * init_scale
        self.current_mse = self.calculate_mse(self.current_weights)
        
        if self.best_weights is None or self.current_mse < self.best_mse:
            self.best_weights = self.current_weights.copy()
            self.best_mse = self.current_mse
            
        return self.get_state()
        
    def calculate_mse(self, weights: np.ndarray) -> float:
        """Calculate Mean Squared Error with L2 regularization"""
        prediction_error = np.mean((self.y - self.X @ weights) ** 2)
        l2_penalty = self.config.l2_penalty * np.sum(weights ** 2)
        return prediction_error + l2_penalty
        
 
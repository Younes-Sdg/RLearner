from rlearner.base import BaseModel
from ..environments.regression import RegressionEnvironmentRL
from ..policies.linear_model_policies import GradientPolicy
import numpy as np

class LinearModelRL(BaseModel):
    """Linear model that utilizes reinforcement learning for weight updates."""

    def __init__(self, learning_rate: float = 0.01, max_steps: int = 50000,
                 mse_threshold: float = 0.001, n_features: int = None):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.mse_threshold = mse_threshold
        self.n_features = n_features
        self.environment = self._init_environment(mse_threshold)
        self.policy = self._init_policy(n_features, learning_rate)

    def _init_environment(self, mse_threshold: float) -> RegressionEnvironmentRL:
        return RegressionEnvironmentRL(mse_threshold)

    def _init_policy(self, n_features: int, learning_rate: float) -> GradientPolicy:
        return GradientPolicy(n_features, learning_rate)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.environment.set_data(X, y)
        state = self.environment.reset()
        
        for step in range(self.max_steps):
            action = self.policy.select_action(state)
            next_state, reward, done = self.environment.step(action)
            self.policy.update(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print(f"Terminated after {step + 1} steps with reward {reward:.4f}")
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.environment.predict(X)

    def get_weights(self) -> np.ndarray:
        return self.environment.get_weights()

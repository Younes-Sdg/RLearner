import numpy as np
from typing import Tuple, List
from enum import Enum

class Policy(Enum):
    GRADIENT = "gradient"
    DIRECT = "direct"

class RLEnvironment:
    """Environment that simulates the linear regression problem as an RL task"""
    def __init__(self, X: np.ndarray, y: np.ndarray, mse_threshold: float = 0.001):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.mse_threshold = mse_threshold
        self.best_mse = float('inf')
        self.history: List[float] = []
        self.reset()
    
    def reset(self) -> np.ndarray:
        self.weights = np.random.randn(self.n_features) * 0.1
        self.history = []
        return self._get_state()
    
    def _get_mse(self) -> float:
        predictions = self.X.dot(self.weights)
        return np.mean((predictions - self.y) ** 2)
    
    def _get_state(self) -> np.ndarray:
        mse = self._get_mse()
        self.history.append(mse)
        mse_change = 0 if len(self.history) < 5 else self.history[-5] - self.history[-1]
        return np.concatenate([self.weights, [mse, mse_change]])
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        old_mse = self._get_mse()
        self.weights += action
        new_mse = self._get_mse()
        
        reward = old_mse - new_mse
        if new_mse < self.best_mse:
            self.best_mse = new_mse
            reward += 0.1
            
        done = new_mse < self.mse_threshold
        return self._get_state(), reward, done

class DirectRLPolicy:
    """Policy that learns purely through trial and error"""
    def __init__(self, n_features: int, learning_rate: float = 0.01):
        self.n_features = n_features
        self.lr = learning_rate
        self.success_directions = np.zeros((0, n_features))  # Store as 2D array
        self.max_directions = 10
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        weights = state[:-2]
        mse = state[-2]
        mse_change = state[-1]
        
        # If we have successful directions, sometimes reuse them
        if len(self.success_directions) > 0 and np.random.random() < 0.3:
            # Randomly combine successful directions
            weights = np.random.random(len(self.success_directions))
            weights = weights / weights.sum()
            base_direction = self.success_directions.T.dot(weights)
            noise = np.random.randn(self.n_features) * self.lr * 0.1
            return base_direction * self.lr + noise
        
        # Otherwise, try new direction
        scale = np.sqrt(mse) * self.lr
        direction = np.random.randn(self.n_features)
        direction = direction / np.linalg.norm(direction) * scale
        
        return direction
    
    def update(self, state: np.ndarray, action: np.ndarray, reward: float):
        if reward > 0:
            direction = action / (np.linalg.norm(action) + 1e-8)
            direction = direction.reshape(1, -1)
            
            if len(self.success_directions) < self.max_directions:
                self.success_directions = np.vstack([self.success_directions, direction])
            else:
                # Replace oldest direction
                self.success_directions = np.vstack([self.success_directions[1:], direction])

class GradientPolicy:
    """Original gradient-based policy"""
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
    
    def select_action(self, state: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights = state[:-2]
        predictions = X.dot(weights)
        gradient = X.T.dot(predictions - y) / len(y)
        noise = np.random.randn(len(weights)) * self.lr * 0.01
        return -gradient * self.lr + noise
    
    def update(self, state: np.ndarray, action: np.ndarray, reward: float):
        pass

class LinearModelRL:
    def __init__(self, learning_rate: float = 0.01, policy: Policy = Policy.GRADIENT):
        self.lr = learning_rate
        self.policy_type = policy
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_steps: int = 1000) -> 'LinearModelRL':
        env = RLEnvironment(X, y)
        state = env.reset()
        
        if self.policy_type == Policy.GRADIENT:
            policy = GradientPolicy(self.lr)
        else:
            policy = DirectRLPolicy(X.shape[1], self.lr)
        
        best_weights = None
        best_mse = float('inf')
        
        for step in range(max_steps):
            if self.policy_type == Policy.GRADIENT:
                action = policy.select_action(state, X, y)
            else:
                action = policy.select_action(state)
            
            next_state, reward, done = env.step(action)
            
            # Track best weights
            current_mse = next_state[-2]
            if current_mse < best_mse:
                best_mse = current_mse
                best_weights = env.weights.copy()
            
            policy.update(state, action, reward)
            
            if step % 100 == 0:
                print(f"Step {step}, MSE: {current_mse:.6f}")
            
            if done:
                print(f"Converged at step {step}")
                break
                
            state = next_state
        
        # Use best weights found
        self.weights = best_weights if best_weights is not None else env.weights
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.weights)
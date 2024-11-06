import numpy as np
from ..base import BasePolicy
from collections import deque

class GradientPolicy(BasePolicy):
    def __init__(self, n_features: int, learning_rate: float = 0.01, epsilon: float = 0.1):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_space = np.array([0.5, 1.0, 2.0])
        
        # Policy network weights
        self.policy_weights = np.random.randn(2 * n_features, len(self.action_space)) * 0.01
        
        # Use deque for experience buffer with fixed size
        self.experience_buffer = deque(maxlen=1000)
        
    def select_action(self, state: dict) -> np.ndarray:
        gradient = state['gradient']
        
        if np.random.random() < self.epsilon:
            lr_multiplier = np.random.choice(self.action_space)
        else:
            state_features = np.concatenate([
                state['weights'].flatten(),
                gradient.flatten()
            ]).reshape(-1, 1)
            
            action_values = (state_features.T @ self.policy_weights).flatten()
            lr_multiplier = self.action_space[np.argmax(action_values)]
        
        return -lr_multiplier * self.learning_rate * gradient

    def update(self, state: dict, action: np.ndarray, reward: float, next_state: dict, done: bool) -> None:
        
        experience = {
            'state': {
                'weights': state['weights'].copy(),
                'gradient': state['gradient'].copy(),
                'mse': state['mse'],
                'best_mse': state['best_mse']
            },
            'action': action.copy(),
            'reward': reward,
            'next_state': {
                'weights': next_state['weights'].copy(),
                'gradient': next_state['gradient'].copy(),
                'mse': next_state['mse'],
                'best_mse': next_state['best_mse']
            },
            'done': done
        }
        
        self.experience_buffer.append(experience)
        
        if len(self.experience_buffer) >= 32:
            self._update_policy()
            
    def _update_policy(self):
        # Sample batch from experience buffer
        batch_indices = np.random.randint(0, len(self.experience_buffer), size=32)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        for experience in batch:
            state = experience['state']
            action = experience['action']
            reward = experience['reward']
            
            state_features = np.concatenate([
                state['weights'].flatten(),
                state['gradient'].flatten()
            ]).reshape(-1, 1)
            
            lr_multiplier = np.linalg.norm(action) / (self.learning_rate * np.linalg.norm(state['gradient']))
            action_idx = np.argmin(np.abs(self.action_space - lr_multiplier))
            
            gradient = np.zeros_like(self.policy_weights)
            gradient[:, action_idx] = state_features.flatten() * reward
            
            self.policy_weights += self.learning_rate * gradient
import numpy as np
from .base import BasePolicy

class GradientPolicy(BasePolicy):
    def __init__(self, n_features: int, learning_rate: float = 0.01):
        """
        Initialize the Gradient policy.
        
        Args:
            n_features (int): Number of input features
            learning_rate (float): Learning rate for updates
        """
        self.n_features = n_features
        self.learning_rate = learning_rate

    def select_action(self, state: dict) -> np.ndarray:
        """
        Select action based on current state using the negative gradient.
        
        Args:
            state (dict): Current state containing weights and gradient
            
        Returns:
            np.ndarray: Action (weight updates)
        """
        gradient = state['gradient']
        return -self.learning_rate * gradient  # Update weights in the direction of the negative gradient

    def update(self, state: dict, action: np.ndarray, reward: float, next_state: dict, done: bool) -> None:
        """
        Update policy based on the transition information.
        
        Args:
            state (dict): Current state
            action (np.ndarray): Action taken
            reward (float): Reward received
            next_state (dict): Next state
            done (bool): Whether episode is complete
        """
        # No additional updates needed for simple gradient descent
        pass

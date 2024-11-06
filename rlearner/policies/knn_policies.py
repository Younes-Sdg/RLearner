import numpy as np
from ..base import BasePolicy

class KNNPolicy(BasePolicy):
    def __init__(self, max_k, learning_rate, epsilon, gamma):
        self.max_k = max_k
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_values = {}

    def select_action(self, state_key):
        """Select k using an epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(1, self.max_k + 1)
        
        if state_key not in self.q_values:
            self.q_values[state_key] = np.zeros(self.max_k)
        
        return np.argmax(self.q_values[state_key]) + 1

    def update(self, state_key, action, reward, next_state_key):
        """Update Q-values based on the action taken and reward received."""
        if state_key not in self.q_values:
            self.q_values[state_key] = np.zeros(self.max_k)
        if next_state_key not in self.q_values:
            self.q_values[next_state_key] = np.zeros(self.max_k)

        old_q = self.q_values[state_key][action - 1]
        next_max_q = np.max(self.q_values[next_state_key])
        
        new_q = old_q + self.learning_rate * (reward + self.gamma * next_max_q - old_q)
        self.q_values[state_key][action - 1] = new_q

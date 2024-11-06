import numpy as np
from ..base import BaseModel, BaseEnvironment, BasePolicy
from rlearner.environments.knn import KNNEnvironment
from rlearner.policies.knn_policies import KNNPolicy

class KNN_RL(BaseModel):
    def __init__(self, max_k=10, learning_rate=0.1, epsilon=0.3, gamma=0.9, max_steps=1000):
        super().__init__(max_steps)
        self.max_k = max_k
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self._init_environment()
        self._init_policy()

    def _init_environment(self):
        """Initialize the KNN environment with the training data."""
        self.environment = KNNEnvironment()

    def _init_policy(self):
        """Initialize the KNN policy for action selection."""
        self.policy = KNNPolicy(self.max_k, self.learning_rate, self.epsilon, self.gamma)

    def fit(self, X_train, y_train, X_val, y_val, episodes=100):
        """Custom fit method for KNN-RL with episodes."""
        self.environment.set_data(X_train, y_train)

        for episode in range(episodes):
            total_reward = 0
            
            for i in range(len(X_val)):
                current_state = X_val[i]
                state_key = self._get_state_key(current_state)
                
                # Choose action (k value) using the policy
                k = self.policy.select_action(state_key)
                
                
                prediction = self.environment.predict(current_state, k)
                reward = self._get_reward(prediction, y_val[i])
                total_reward += reward
                
                # Get next state and update the Q-table
                next_state = X_val[min(i + 1, len(X_val) - 1)]
                next_state_key = self._get_state_key(next_state)
                self.policy.update(state_key, k, reward, next_state_key)
            
            

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            state_key = self._get_state_key(x)
            k = self.policy.select_action(state_key)
            prediction = self.environment.predict(x, k)
            predictions.append(prediction)
        return np.array(predictions)

    def _get_state_key(self, X):
        """Convert a feature vector into a hashable state key for Q-table."""
        return tuple(np.round(X, 2))

    def _get_reward(self, prediction, true_label):
        """Reward based on prediction accuracy."""
        return 1.0 if prediction == true_label else -1.0

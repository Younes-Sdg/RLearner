import numpy as np
from rlearner.base import BaseEnvironment

class KNNEnvironment(BaseEnvironment):
    def __init__(self, X_train=None, y_train=None):
        self.X_train = X_train
        self.y_train = y_train

    def set_data(self, X_train, y_train):
        """Initialize environment with training data."""
        self.X_train = X_train
        self.y_train = y_train

    def reset(self):
        """Reset environment; no specific state management here."""
        
        return None

    def step(self, action):
        """Dummy method to comply with BaseEnvironment. Real use in fit loop."""
        return None, 0.0, False

    def predict(self, x, k):
        """Predict the label of x using k-nearest neighbors."""
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:k]
        neighbor_labels = self.y_train[nearest_indices]
        prediction = np.bincount(neighbor_labels).argmax()
        return prediction

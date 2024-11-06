import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rlearner.models.knn_model import KNN_RL  
from sklearn.neighbors import KNeighborsClassifier

@pytest.fixture
def data():
    """
    Fixture to generate synthetic dataset for testing.
    """
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, 
                                n_redundant=0, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def test_knn_rl_vs_sklearn(data):
    """
    Test to compare KNN-RL with sklearn KNN model on classification accuracy.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    knn_rl = KNN_RL(max_k=10, learning_rate=0.1, epsilon=0.3, gamma=0.9)
    knn_rl.fit(X_train, y_train, X_val, y_val, episodes=50)
    y_pred_rl = knn_rl.predict(X_test)
    
    
    sklearn_knn = KNeighborsClassifier(n_neighbors=5)
    sklearn_knn.fit(X_train, y_train)
    y_pred_sklearn = sklearn_knn.predict(X_test)
    
   
    accuracy_rl = accuracy_score(y_test, y_pred_rl)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    print(f"KNN-RL Accuracy: {accuracy_rl:.3f}")
    print(f"Sklearn KNN Accuracy: {accuracy_sklearn:.3f}")
    
    assert abs(accuracy_rl - accuracy_sklearn) <= 0.05, f"Accuracy difference is too large: KNN-RL={accuracy_rl:.3f}, Sklearn={accuracy_sklearn:.3f}"


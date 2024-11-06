import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from rlearner.models.linear_model import LinearModelRL

@pytest.fixture
def regression_data():
    np.random.seed(42)
    X = np.random.rand(100, 3) * 10  
    true_weights = np.array([2.5, -1.5, 3.0]) 
    noise = np.random.randn(100) * 2
    y = X @ true_weights + noise  
    return X, y, true_weights

def test_linear_model_rl_vs_sklearn(regression_data):
    X, y, true_weights = regression_data
    
    model_rl = LinearModelRL(
        learning_rate=0.01,
        max_steps=5000,
        mse_threshold=0.001,
        n_features=X.shape[1]
    )
    model_rl.fit(X, y)
    predictions_rl = model_rl.predict(X)
    
    mse_rl = mean_squared_error(y, predictions_rl)
    r2_rl = r2_score(y, predictions_rl)

    obtained_weights_rl = model_rl.get_weights()  
    
    for obtained_weight, true_weight in zip(obtained_weights_rl, true_weights):
        assert abs(obtained_weight - true_weight) <= 0.5, f"Weight difference too large: RL={obtained_weight:.4f}, True={true_weight:.4f}"
    
    model_lr = LinearRegression()
    model_lr.fit(X, y)
    predictions_lr = model_lr.predict(X)
    
    mse_lr = mean_squared_error(y, predictions_lr)
    r2_lr = r2_score(y, predictions_lr)
    
    assert abs(mse_rl - mse_lr) <= 0.5, f"MSE difference too large: RL={mse_rl:.4f}, Sklearn={mse_lr:.4f}"
    assert abs(r2_rl - r2_lr) <= 0.5, f"R² Score difference too large: RL={r2_rl:.4f}, Sklearn={r2_lr:.4f}"

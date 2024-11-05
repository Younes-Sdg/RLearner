from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Optional, Tuple

class BaseEnvironment(ABC):
    """Abstract base class for all RL environments in the library."""
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return next state, reward, and done flag."""
        pass


class BasePolicy(ABC):
    """Abstract base class for all policies."""
    
    @abstractmethod
    def select_action(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """Select action based on current state."""
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Update policy based on the transition information."""
        pass


class BaseModel(ABC):
    """Abstract base class for all RL-based models."""
    
    def __init__(self, max_steps: int = 1000):
        self.max_steps = max_steps
        self.environment: Optional[BaseEnvironment] = None
        self.policy: Optional[BasePolicy] = None
    
    @abstractmethod
    def _init_environment(self) -> None:
        """Initialize the specific environment for this model."""
        pass
    
    @abstractmethod
    def _init_policy(self) -> None:
        """Initialize the specific policy for this model."""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Training loop for RL-based models."""
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
        
        return self

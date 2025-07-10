from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Tuple, Union, List

class BaseModel(ABC):

    def __init__(
            self, s0: float,
            x0: Union[float, np.ndarray, List[float]],
            x_dimensions: int = 1
            ):
        self.s0 = s0  
        self.x_dimensions = x_dimensions
        
        # Convert x0 to numpy array if it's a list or scalar
        if isinstance(x0, (float, int)):
            self.x0 = np.array([x0])
        elif isinstance(x0, list):
            self.x0 = np.array(x0)
        else:
            self.x0 = x0
            
        # Ensure x0 has the correct shape based on x_dimensions
        if len(self.x0) != x_dimensions:
            raise ValueError(f"x0 must have {x_dimensions} dimensions, got {len(self.x0)}")
            
        self.pathS = None
        self.pathX = None
        
    @abstractmethod
    def sigma(self, t: float, S: np.ndarray, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def dupire_vol(self, t: float, S: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def update_X(
        self, 
        t: float
    ) -> np.ndarray:
        pass
    
    def generate_brownian_motions(
        self, 
        num_paths: int, 
        num_steps: int, 
        use_sobol: bool = False,
        rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        dt = 1.0 / 252
        if use_sobol:
            from scipy.stats import qmc, norm
            sampler = qmc.Sobol(d=num_steps, scramble=True)
            sobol_samples = sampler.random(n=num_paths)
            normal_samples = norm.ppf(sobol_samples)
            dW_S = normal_samples * np.sqrt(dt)
            
            return dW_S
        else:
            # Standard Monte Carlo with pseudo-random numbers
            dW_S = rng.normal(0, np.sqrt(dt), (num_paths, num_steps))
            return dW_S
    
    def combine_X_dimensions(self, X: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """
        Combine multiple X dimensions into a single value, useful for models that expect a single X value.
        
        Args:
            X: Path-dependent factors, shape (num_particles, x_dimensions) or (num_particles,)
            weights: Optional weights for each dimension, defaults to equal weights
            
        Returns:
            Combined X values, shape (num_particles,)
        """
        if X.ndim == 1:
            return X

        if weights is None:
            weights = np.ones(X.shape[1]) / X.shape[1]
            
        weights = np.array(weights) / np.sum(weights)
        return np.sum(X * weights.reshape(1, -1), axis=1)
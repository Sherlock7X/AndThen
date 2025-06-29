import numpy as np
from base_model import BaseModel
from typing import Callable, Union, List

class PDVModel(BaseModel):
    """PDV model: dS_t/S_t = σ(t,S,X)l(t,S)dW_t^S"""
    
    def __init__(
        self,
        s0: float,
        x0: Union[float, np.ndarray, List[float]],
        dt: float = 1/252,
        Delta: float = 1/12,
        kappa: float = 0.8,
        sigma0: float = 0.2,
        X_type: str = 'VWAP',
        sigma_type: int = 1,
        dupire_vol_interp: Callable = None,
        x_dimensions: int = 1
    ):
        """
        Args:
            X_type: String or list of strings specifying the type of path-dependent factors
                'VWAP': Volume weighted average price
                'min-max': min-max price
                'timelag': time lagged price
            x_dimensions: Number of path-dependent factors to track
            dupire_vol_interp: Dupire volatility after interpolation
        """
        # Handle X_type being a string or list
        if isinstance(X_type, str):
            self.X_type = [X_type] * x_dimensions
        else:
            if len(X_type) != x_dimensions:
                raise ValueError(f"X_type must have {x_dimensions} elements if provided as a list")
            self.X_type = X_type
            
        super().__init__(s0, x0, x_dimensions)
        
        self.dt = dt  # time step
        self.Delta = Delta
        self.sigma_type = sigma_type  # type of volatility function
        self.dupire_vol_interp = dupire_vol_interp
        self.vol_cap = 0.32
        self.vol_floor = 0.08
        self.kappa = kappa
        self.sigma0 = sigma0
        
    
    def dupire_vol(self, t: float, S: np.ndarray) -> np.ndarray:
        """get Dupire volatility σ(t,S)"""
        return self.dupire_vol_interp(t, S)
    
    def update_X(
        self, 
        t: float
    ) -> np.ndarray:
        """update path-dependent factors X
        
        Returns:
            X: Array of shape (num_particles, x_dimensions)
        """
        handlers = {
            'VWAP': self._handle_vwap,
            'min': self._handle_min,
            'max': self._handle_max,
            'timelag': self._handle_timelag
        }
        
        # Initialize result array for all X dimensions
        num_particles = self.pathS.shape[0]
        result = np.zeros((num_particles, self.x_dimensions))
        
        # Process each X dimension with its corresponding handler
        for i in range(self.x_dimensions):
            # Get the appropriate handler for this dimension
            x_type = self.X_type[i]
            handler = handlers.get(x_type)
            
            if handler is None:
                raise ValueError(f"Unsupported X_type: {x_type}")
            
            # Call the handler and store the result for this dimension
            result[:, i] = handler(t)
            
        return result

    def _handle_vwap(self, t: float) -> np.ndarray:
        """Handle VWAP calculation"""
        if t <= self.Delta:
            # Use all available data up to time t
            current_idx = max(1, int(t / self.dt) + 1)
            return np.mean(self.pathS[:, :current_idx], axis=1)
        else:
            # Use data in the window [t-Delta, t]
            window_size = int(self.Delta / self.dt)
            current_idx = int(t / self.dt) + 1
            start_idx = max(0, current_idx - window_size)
            return np.mean(self.pathS[:, start_idx:current_idx], axis=1)

    def _handle_min(self, t: float) -> np.ndarray:
        """Handle min calculation"""
        if t <= self.Delta:
            current_idx = max(1, int(t / self.dt) + 1)
            return np.min(self.pathS[:, :current_idx], axis=1)
        else:
            # Use data in the window [t-Delta, t] for min-max calculation
            window_size = int(self.Delta / self.dt)
            current_idx = int(t / self.dt) + 1
            start_idx = max(0, current_idx - window_size)
            return np.min(self.pathS[:, start_idx:current_idx], axis=1)
        
    def _handle_max(self, t: float) -> np.ndarray:
        """Handle max calculation"""
        if t <= self.Delta:
            current_idx = max(1, int(t / self.dt) + 1)
            return np.max(self.pathS[:, :current_idx], axis=1)
        else:
            # Use data in the window [t-Delta, t] for min-max calculation
            window_size = int(self.Delta / self.dt)
            current_idx = int(t / self.dt) + 1
            start_idx = max(0, current_idx - window_size)
            return np.max(self.pathS[:, start_idx:current_idx], axis=1)

    def _handle_timelag(self, t: float) -> np.ndarray:
        """Handle time lag calculation"""
        if t <= self.Delta:
            # If t is less than Delta, use the initial price
            return self.pathS[:, 0]
        else:
            # Return the price from Delta time ago
            lag_index = int((t - self.Delta) / self.dt) + 1
            return self.pathS[:, lag_index]
    
    def sigma(self, t: float, S: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Volatility function σ(t,S,X)
        
        Args:
            t: time
            S: stock price, shape (num_particles,)
            X: path-dependent factors, shape (num_particles, x_dimensions)
            
        Returns:
            volatility: shape (num_particles,)
        """
        # For backward compatibility, if X is still 1D (num_particles,), reshape it
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # If sigma_type is 1 or 2, we use only the first dimension of X
        if self.sigma_type == 1:
            return self.vol_cap * (S <= X[:, 0]) + self.vol_floor * (S > X[:, 0])
        elif self.sigma_type == 2:
            return self.vol_cap * (abs(S / X[:, 0] - 1) > self.kappa * self.sigma0 * np.sqrt(self.Delta)) + \
                   self.vol_floor * (abs(S / X[:, 0] - 1) <= self.kappa * self.sigma0 * np.sqrt(self.Delta))
        elif self.sigma_type == 3:
            return self.vol_cap * ((S - X[:, 0]) / (X[:, 1] - X[:, 0] + 1e-8) <= 1/2) + \
                   self.vol_floor * ((S - X[:, 0]) / (X[:, 1] - X[:, 0] + 1e-8) > 1/2)
        
if __name__ == "__main__":
    xtype = 'VWAP'  
    sigma_type = 1
    model = PDVModel(s0=100, x0=100, X_type=xtype, sigma_type=sigma_type)
    pathS = np.zeros((10, 50))  # Simulated paths for S
    pathX = np.zeros((10, 50, 1))  # Simulated paths for X
    pathvol = np.zeros((10, 50))  # Simulated paths for volatility
    for i in range(10):
        pathS[i] = np.linspace(100, 105, 50) + np.random.normal(0, 1, 50)
    pathS[:, 0] = 100  # Initial condition for S
    model.pathS = pathS

    for j in range(50):
        X_values = model.update_X(j * model.dt)
        pathX[:, j] = X_values  # Assign to 3D array
        pathvol[:, j] = model.sigma(j * model.dt, pathS[:, j], X_values)
    model.pathX = pathX

    t = 0.1  # Example time
    time_step = int(t / model.dt)
    Xt = pathX[:, time_step]
    St = pathS[:, time_step]
    sigma_t = model.sigma(t, St, Xt)
    print(f"Single dimension - At time {t}, time_step: {time_step}, St: {St}, Xt: {Xt}, σ(t,S,X): {sigma_t}")


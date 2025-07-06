import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, interp1d
from typing import Dict, List, Tuple, Union, Callable
from abc import ABC, abstractmethod
import time

class GridImpliedVol:
    """
    A dummy class for implied volatility interpolation.
    """
    def __init__(self, time_points: np.ndarray, moneyness_points: np.ndarray, vol_data: np.ndarray):
        self.base_vol = 0.2
    
    def get_vol(self, t: float, moneyness: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.base_vol * np.ones_like(np.asarray(moneyness))
    
class GridInterpolator(GridImpliedVol):
    """
    Class for interpolating volatility surfaces.
    use fixed time points to accelerate interpolation. e.g. 0:1:1/252
    use a preset range of moneyness values, e.g. [0.3, 1.6]
    use dictionary to store the interpolation functions for each tenor.
    another class called 'GridImpliedVol' has a function '.get_vol' to get the implied volatilities
    goal: 
    give a (t, moneyness) pair, return the interpolated volatility.
    """
    def __init__(self, time_points: np.ndarray, moneyness_points: np.ndarray, vol_data: np.ndarray):
        """
        Initialize the GridInterpolator with grid points and volatility data.
        
        Parameters:
        -----------
        time_points: np.ndarray
            1D array of time points (in years)
        moneyness_points: np.ndarray
            1D array of moneyness points (strike/spot)
        vol_data: np.ndarray
            2D array of volatility values, shape (len(time_points), len(moneyness_points))
        """
        super().__init__(time_points, moneyness_points, vol_data)
        
        self.time_points = np.asarray(time_points)
        self.moneyness_points = np.asarray(moneyness_points)
        self.vol_data = np.asarray(vol_data)
        
        if self.vol_data.shape != (len(self.time_points), len(self.moneyness_points)):
            raise ValueError(f"Expected vol_data shape {(len(self.time_points), len(self.moneyness_points))}, "
                             f"got {self.vol_data.shape}")
        
        # Create interpolators for each time slice
        self.time_interpolators = {}
        for i, t in enumerate(self.time_points):
            self.time_interpolators[t] = interp1d(
                self.moneyness_points, 
                self.vol_data[i, :],
                bounds_error=False,
                fill_value=(self.vol_data[i, 0], self.vol_data[i, -1])  # Extrapolate with edge values
            )
        
        # Create a 2D regular grid interpolator for arbitrary (t, moneyness) points
        self.grid_interpolator = RegularGridInterpolator(
            (self.time_points, self.moneyness_points),
            self.vol_data,
            bounds_error=False,
            fill_value=None  # Use extrapolation based on closest values
        )
    
    def get_vol(self, t: float, moneyness: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get implied volatility for given time and moneyness.
        Optimized for both scalar and vector inputs.
        
        Parameters:
        -----------
        t: float
            Time to maturity in years
        moneyness: float or np.ndarray
            Moneyness (strike/spot) - can be scalar or vector
            
        Returns:
        --------
        float or np.ndarray
            Interpolated volatility value(s)
        """
        moneyness_array = np.asarray(moneyness)
        
        # Fast path: t exactly matches one of our time points - use 1D interpolation
        if t in self.time_interpolators:
            return self.time_interpolators[t](moneyness_array)
        
        # General case: t is not on the grid - use 2D interpolation
        # Handle scalar case
        if moneyness_array.ndim == 0:
            points = np.array([[t, float(moneyness_array)]])
            return float(self.grid_interpolator(points)[0])
        
        # Handle vector case efficiently - reshape for batch processing
        points = np.column_stack([
            np.full_like(moneyness_array, t), 
            moneyness_array
        ])
        return self.grid_interpolator(points)
    
    def plot_vol_surface(self, title: str = "Implied Volatility Surface"):
        """
        Plot the volatility surface.
        """
        T, K = np.meshgrid(self.time_points, self.moneyness_points, indexing='ij')
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(T, K, self.vol_data, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('Time to Maturity')
        ax.set_ylabel('Moneyness (Strike/Spot)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(title)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.tight_layout()
        
        return fig
    
    def slice_by_time(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a slice of the volatility surface at a specific time.
        
        Parameters:
        -----------
        t: float
            Time to maturity
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            moneyness points and corresponding volatility values
        """
        vols = self.get_vol(t, self.moneyness_points)
        return self.moneyness_points, vols
    
    def slice_by_moneyness(self, moneyness: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a slice of the volatility surface at a specific moneyness.
        
        Parameters:
        -----------
        moneyness: float
            Moneyness (Strike/Spot)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            time points and corresponding volatility values
        """
        vols = np.array([
            self.get_vol(t, moneyness) for t in self.time_points
        ])
        return self.time_points, vols

def generate_sample_data(num_time_points=252, num_moneyness_points=50):
    """Generate sample volatility surface data"""
    # Create time points (in years)
    time_points = np.linspace(0.01, 1.0, num_time_points)
    
    # Create moneyness points
    moneyness_points = np.linspace(0.6, 1.4, num_moneyness_points)
    
    # Create volatility surface with smile effect
    vol_data = np.zeros((len(time_points), len(moneyness_points)))
    
    # Base volatility level
    base_vol = 0.2
    
    # Generate volatility surface with smile and term structure
    for i, t in enumerate(time_points):
        # Term structure effect (volatility decreases with time)
        term_effect = np.exp(-0.2 * t)
        
        for j, m in enumerate(moneyness_points):
            # Smile effect (higher vol for away-from-money options)
            smile = 0.15 * (m - 1.0) ** 2
            
            # Skew effect (put options have higher vol than calls)
            skew = -0.05 * (m - 1.0)
            
            # Combine effects
            vol_data[i, j] = base_vol * term_effect + smile + skew
    
    return time_points, moneyness_points, vol_data

def benchmark_interpolator(grid_interpolator, num_samples=10000):
    """Benchmark the performance of the interpolator"""
    # Generate random test points
    random_times = np.random.uniform(0.01, 2.0, num_samples)
    random_moneyness = np.random.uniform(0.6, 1.4, num_samples)
    
    # Scalar input benchmark
    start_time = time.time()
    for i in range(100):  # Sample a smaller batch for scalar testing
        vol = grid_interpolator.get_vol(random_times[i], random_moneyness[i])
    scalar_time = (time.time() - start_time) / 100
    print(f"Average time per scalar query: {scalar_time*1000:.3f} ms")
    
    # Vector input benchmark
    start_time = time.time()
    vols = grid_interpolator.get_vol(random_times[0], random_moneyness)
    vector_time = time.time() - start_time
    print(f"Time for {num_samples} vector queries: {vector_time*1000:.3f} ms")
    print(f"Average time per vector element: {vector_time*1000/num_samples:.6f} ms")
    
    # Exact time slice benchmark
    start_time = time.time()
    vols = grid_interpolator.get_vol(grid_interpolator.time_points[2], random_moneyness)
    exact_time = time.time() - start_time
    print(f"Time for {num_samples} queries on exact time slice: {exact_time*1000:.3f} ms")
    
    return scalar_time, vector_time, exact_time

def main():
    # Generate sample data
    time_points, moneyness_points, vol_data = generate_sample_data()
    
    # Create the grid interpolator
    grid_interpolator = GridInterpolator(time_points, moneyness_points, vol_data)
    
    # Benchmark performance
    print("Performance Benchmarks:")
    scalar_time, vector_time, exact_time = benchmark_interpolator(grid_interpolator)
    
    # Plot the volatility surface
    fig = grid_interpolator.plot_vol_surface("Sample Implied Volatility Surface")
    
    # Plot some time slices
    plt.figure(figsize=(10, 6))
    for t in [0.1, 0.5, 1.0, 1.5]:
        m, v = grid_interpolator.slice_by_time(t)
        plt.plot(m, v, label=f"T = {t:.2f}")
    plt.xlabel("Moneyness (K/S)")
    plt.ylabel("Implied Volatility")
    plt.title("Volatility Smile Across Different Maturities")
    plt.legend()
    plt.grid(True)
    
    # Plot some moneyness slices
    plt.figure(figsize=(10, 6))
    for m in [0.7, 0.9, 1.0, 1.1, 1.3]:
        t, v = grid_interpolator.slice_by_moneyness(m)
        plt.plot(t, v, label=f"Moneyness = {m:.2f}")
    plt.xlabel("Time to Maturity")
    plt.ylabel("Implied Volatility")
    plt.title("Term Structure Across Different Moneyness Levels")
    plt.legend()
    plt.grid(True)
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
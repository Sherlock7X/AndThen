from typing import Callable, Tuple
import numpy as np

class LocalVolSimulator:
    def __init__(
        self,
        model,
        num_particles: int = 10_000,
        time_steps: int = 252,
        dt: float = 1.0 / 252,
        rng: np.random.Generator = np.random.default_rng()
    ):
        """
        A pure Local Volatility model simulator.
        
        Args:
            model: Local volatility model instance
            num_particles: Number of particles for simulation
            time_steps: Number of time steps for simulation
            dt: Time step size
            rng: Random number generator
        """
        self.model = model
        self.num_particles = num_particles
        self.time_steps = time_steps
        self.dt = dt
        self.rng = rng
        
        # Initialize paths
        self.paths = np.zeros((num_particles, time_steps + 1))
        self.paths[:, 0] = model.s0
    
    def simulate(self) -> np.ndarray:
        """
        Simulate price paths using the local volatility model.
        
        Returns:
            np.ndarray: Simulated price paths with shape (num_particles, time_steps + 1)
        """
        print("Simulating Local Volatility Model")
        
        # Generate Brownian motion increments
        dW = self._generate_brownian_increments()
        
        # Simulate the paths
        for t in range(1, self.time_steps + 1):
            S_prev = self.paths[:, t-1]
            
            # Calculate local volatility at current time and stock price
            local_vol = self.model.dupire_vol((t-1) * self.dt, S_prev)
            
            # Euler-Maruyama step for local volatility model:
            # dS = μS dt + σ(t,S)S dW
            self.paths[:, t] = S_prev * (1 + local_vol * dW[:, t-1])
        
        return self.paths
    
    def _generate_brownian_increments(self) -> np.ndarray:
        """
        Generate Brownian motion increments.
        
        Returns:
            np.ndarray: Brownian motion increments with shape (num_particles, time_steps)
        """
        return self.rng.normal(0, np.sqrt(self.dt), (self.num_particles, self.time_steps))

    def plot_paths(self, num_paths: int = 50, figsize=(12, 8)):
        """
        Plot sample price paths from the simulation.
        
        Args:
            num_paths: Number of paths to plot
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        
        if num_paths > self.num_particles:
            num_paths = self.num_particles
        
        time_grid = np.arange(0, self.time_steps + 1) * self.dt
        
        plt.figure(figsize=figsize)
        
        # Randomly select paths to plot
        indices = np.random.choice(self.num_particles, num_paths, replace=False)
        for idx in indices:
            plt.plot(time_grid, self.paths[idx], alpha=0.5)
            
        plt.title('Local Volatility Model - Sample Paths')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.grid(True)
        
        return plt.gcf()
    
    def compute_statistics(self):
        """
        Compute statistics of the simulated paths.
        
        Returns:
            dict: Dictionary containing statistics
        """
        final_prices = self.paths[:, -1]
        
        stats = {
            "mean": np.mean(final_prices),
            "std": np.std(final_prices),
            "median": np.median(final_prices),
            "min": np.min(final_prices),
            "max": np.max(final_prices),
            "quantiles": {
                "5%": np.percentile(final_prices, 5),
                "25%": np.percentile(final_prices, 25),
                "75%": np.percentile(final_prices, 75),
                "95%": np.percentile(final_prices, 95)
            }
        }
        
        return stats

if __name__ == "__main__":
    # Example usage
    import time
    import matplotlib.pyplot as plt
    
    # Create a simple local volatility model class
    class SimpleLocalVolModel:
        def __init__(self, s0=100.0, r=0.02, q=0.01, base_vol=0.2, skew=-0.1):
            self.s0 = s0
            self.r = r  # risk-free rate
            self.q = q  # dividend yield
            self.base_vol = base_vol  # base volatility level
            self.skew = skew  # volatility skew parameter
            
        def dupire_vol(self, t, S):
            """Simple local volatility function with skew"""
            # Time-dependent component (term structure)
            time_factor = np.exp(-0.05 * t)
            
            # Price-dependent component (volatility skew)
            moneyness = S / self.s0
            skew_factor = 1.0 + self.skew * (moneyness - 1.0)
            
            # Ensure volatility stays positive
            return self.base_vol * time_factor * np.maximum(0.1, skew_factor)
    
    # Create the model and simulator
    model = SimpleLocalVolModel(s0=100.0, base_vol=0.2, skew=-0.1)
    simulator = LocalVolSimulator(model=model, num_particles=10_000, time_steps=252)
    
    # Run simulation and measure time
    start_time = time.time()
    paths = simulator.simulate()
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
    
    # Print statistics
    stats = simulator.compute_statistics()
    print("\nSimulation Statistics:")
    print(f"Mean terminal price: {stats['mean']:.2f}")
    print(f"Standard deviation: {stats['std']:.2f}")
    print(f"Min/Max prices: {stats['min']:.2f} / {stats['max']:.2f}")
    print(f"5% / 95% quantiles: {stats['quantiles']['5%']:.2f} / {stats['quantiles']['95%']:.2f}")
    
    # Plot sample paths
    plt.figure(figsize=(10, 6))
    simulator.plot_paths(num_paths=100)
    plt.show()
    
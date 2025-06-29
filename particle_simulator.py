from typing import Callable, Tuple
import numpy as np

class ParticleMonteCarlo:
    def __init__(
        self,
        model,
        num_particles: int = 10_000,
        time_steps: int = 252,
        dt: float = 1.0 / 252,
        bins: int = 200,
        rng: np.random.Generator = np.random.default_rng(),
        method: str = 'bin'  # Method for computing conditional expectation ('bin' or 'kernel')
    ):
        """
        Args:
            model: Model instance with path-dependent factors
            num_particles: Number of particles for simulation
            time_steps: Number of time steps for simulation
            dt: Time step size
            bins: Number of bins for discretizing the stock price range
            rng: Random number generator
        """
        self.model = model
        self.num_particles = num_particles
        self.time_steps = time_steps
        self.dt = dt
        self.rng = rng
        self.method = method
        
        self.model.pathS = np.zeros((num_particles, time_steps + 1))
        self.model.pathS[:, 0] = model.s0
        
        # Initialize pathX with appropriate dimensions
        x_dimensions = getattr(model, 'x_dimensions', 1)
        self.model.pathX = np.zeros((num_particles, time_steps + 1, x_dimensions))
        
        # Initialize each path with x0
        for i in range(x_dimensions):
            self.model.pathX[:, 0, i] = model.x0[i]
        
        self.leverage_surface = np.ones((time_steps, bins))
        
        self.bins = bins
        self.bin_edges = np.linspace(0.5 * model.s0, 1.5 * model.s0, bins + 1)
        self.leverage_surface = np.zeros((time_steps, bins))

    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        dW_S = self.model.generate_brownian_motions(
            self.num_particles, self.time_steps, rng=self.rng
        )
        
        for t in range(1, self.time_steps + 1):
            S_prev = self.model.pathS[:, t-1]
            
            # Extract X_prev based on its dimensions
            if self.model.pathX.ndim == 3:  # Vector-valued X
                X_prev = self.model.pathX[:, t-1, :]
            else:  # Backwards compatibility for scalar X
                X_prev = self.model.pathX[:, t-1]
            
            sigma_sq = self.model.sigma(t-1, S_prev, X_prev) ** 2
            if self.method == 'kernel':
                cond_exp = self._kernel_compute_conditional_expectation(S_prev, sigma_sq)
            elif self.method == 'bin':
                cond_exp = self._bin_compute_conditional_expectation(S_prev, sigma_sq)
            
            l_t = self._compute_leverage(t-1, S_prev, cond_exp)
            
            self.model.pathS[:, t] = S_prev * (
                1 + self.model.sigma(t-1, S_prev, X_prev) * l_t * dW_S[:, t-1]
            )
            
            # Update pathX based on the new model
            X_new = self.model.update_X(t * self.dt)
            
            # Handle different shapes of X_new
            if X_new.ndim == 2:  # If update_X returns (num_particles, x_dimensions)
                if self.model.pathX.ndim == 3:  # Vector-valued X
                    self.model.pathX[:, t, :] = X_new
                else:  # Legacy scalar X case
                    if X_new.shape[1] == 1:
                        self.model.pathX[:, t] = X_new[:, 0]
                    else:
                        # Reshape pathX to accommodate vector X if necessary
                        x_dimensions = X_new.shape[1]
                        new_pathX = np.zeros((self.num_particles, self.time_steps + 1, x_dimensions))
                        new_pathX[:, :t, 0] = self.model.pathX[:, :t]  # Copy existing data to first dimension
                        new_pathX[:, t, :] = X_new  # Set new value for all dimensions
                        self.model.pathX = new_pathX
            else:  # Legacy case where update_X returns (num_particles,)
                if self.model.pathX.ndim == 3:  # Vector-valued X with single dimension
                    self.model.pathX[:, t, 0] = X_new
                else:  # Legacy scalar X
                    self.model.pathX[:, t] = X_new
            
            self._update_leverage_surface(t-1, S_prev, l_t)
        
        return self.model.pathS, self.model.pathX

    def _bin_compute_conditional_expectation(
        self,
        S: np.ndarray,
        sigma_sq: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        bin_indices = np.digitize(S, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.bins - 1)
        
        cond_exp = np.zeros_like(S)
        for i in range(self.bins):
            mask = (bin_indices == i)
            if np.any(mask):
                cond_exp[mask] = np.mean(sigma_sq[mask])
        
        return cond_exp
    
    def _kernel_compute_conditional_expectation(
        self,
        S: np.ndarray,
        sigma_sq: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute conditional expectation E[sigma^2 | S] using kernel method
        
        Args:
            S: Array of stock prices
            sigma_sq: Array of squared volatilities
        
        Returns:
            bin_centers: Centers of the bins for the stock prices
            cond_exp: Conditional expectation E[sigma^2 | S] for each bin
        """
        # Define the kernel bandwidth - use Scott's rule
        n = len(S)
        std_S = np.std(S)
        
        # Handle case when standard deviation is very small or zero
        if std_S < 1e-8:
            bandwidth = 1e-3 * np.mean(S)  # Use 0.1% of mean price as bandwidth
        else:
            bandwidth = 1.06 * std_S * n**(-1/5)
        
        # Compute bin centers
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
        # Initialize array to store conditional expectations
        cond_exp = np.zeros_like(bin_centers)
        
        # For each bin center, compute weighted average of sigma_sq
        for i, center in enumerate(bin_centers):
            # Compute kernel weights with check to avoid divide by zero
            kernel_weights = np.exp(-0.5 * ((S - center) / bandwidth)**2)
            kernel_sum = np.sum(kernel_weights)
            
            # Normalize weights, handling case when all weights might be near-zero
            if kernel_sum < 1e-10:
                cond_exp[i] = np.mean(sigma_sq)  # Fall back to simple mean
            else:
                kernel_weights = kernel_weights / kernel_sum
                cond_exp[i] = np.sum(kernel_weights * sigma_sq)
    
        # Interpolate to get conditional expectation for each particle
        from scipy.interpolate import interp1d
        interpolator = interp1d(
            bin_centers, 
            cond_exp, 
            bounds_error=False, 
            fill_value=(cond_exp[0], cond_exp[-1])
        )
        
        particle_cond_exp = interpolator(S)
        
        return particle_cond_exp

    def _compute_leverage(
        self,
        t: int,
        S: np.ndarray,
        cond_exp_sigma_sq: np.ndarray
    ) -> np.ndarray:
        
        sigma_dup = self.model.dupire_vol(t * self.dt, S)
        return sigma_dup / (np.sqrt(cond_exp_sigma_sq) + 1e-6)  # Avoid division by zero

    def _update_leverage_surface(
        self,
        t: int,
        S: np.ndarray,
        l_t: np.ndarray
    ) -> None:
        
        bin_indices = np.digitize(S, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.bins - 1)
        
        for i in range(self.bins):
            mask = (bin_indices == i)
            if np.sum(mask) > 0:
                self.leverage_surface[t, i] = np.mean(l_t[mask])

if __name__ == "__main__":
    # Example usage
    from pdv_model import PDVModel
    import time
    PDV = PDVModel(s0=100, x0=100)
    PDV.dupire_vol_interp = lambda t, S: 0.2 # dummy Dupire volatility function
    particle_mc = ParticleMonteCarlo(model=PDV, num_particles=30_000, time_steps=252, method='kernel')
    start_time = time.time()
    pathS, pathX = particle_mc.simulate()
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds")

    import matplotlib.pyplot as plt
    # Plot leverage surface
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # Create bin centers
    S_vals = (particle_mc.bin_edges[:-1] + particle_mc.bin_edges[1:]) / 2  
    t_vals = np.arange(particle_mc.time_steps)

    leverage = particle_mc.leverage_surface

    # Create a single figure with two subplots
    fig = plt.figure(figsize=(18, 8))
    
    # First subplot - 2D heatmap
    ax1 = fig.add_subplot(121)
    
    # For pcolormesh with shading='flat', we need grid dimensions to be (N+1, M+1)
    S_edges = particle_mc.bin_edges  # These are already the edges
    t_edges = np.arange(particle_mc.time_steps + 1)  # Add one more edge
    
    S_grid, T_grid = np.meshgrid(S_edges, t_edges)
    
    pcm = ax1.pcolormesh(T_grid, S_grid, leverage, cmap='viridis')
    fig.colorbar(pcm, ax=ax1, label='Leverage')
    ax1.set_title('Leverage Surface (2D)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Stock Price (S)')
    
    # Second subplot - 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    
    # For 3D surface plot, we need the centers, not the edges
    S_centers, T_centers = np.meshgrid(S_vals, t_vals)
    
    # Plot the surface using the filtered data
    surf = ax2.plot_surface(S_centers, T_centers, leverage, 
                        cmap=cm.viridis, edgecolor='none', alpha=0.8)
    
    # Add labels and title
    ax2.set_xlabel('Stock Price (S)')
    ax2.set_ylabel('Time Step')
    ax2.set_zlabel('Leverage Factor')
    ax2.set_title('Leverage Surface (3D)')
    
    # Add color bar
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Leverage')
    
    # Adjust view angle for better visualization
    ax2.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig('ParticleMonteCarlo/figures/bin_leverage_surface_multi.png', dpi=300)
    plt.show()
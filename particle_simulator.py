from typing import Callable, Tuple
import numpy as np
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod

class LeverageSurface:
    """
    Manages the leverage surface, including its calculation, storage, and interpolation.
    """
    def __init__(self, time_steps: int, bins: int, s0: float):
        self.time_steps = time_steps
        self.bins = bins
        self.s0 = s0
        self.leverage_surface = np.full((time_steps, bins), np.nan)
        self.leverage_interpolation = {}
        self._create_bin_edges()

    def _create_bin_edges(self):
        """Creates non-uniform bin edges for stock prices, denser around ATM."""
        min_price = 0.5 * self.s0
        max_price = 1.5 * self.s0
        alpha = 1.5
        points = np.linspace(-1, 1, self.bins + 1)
        transformed_points = np.tanh(alpha * points) / np.tanh(alpha)
        self.bin_edges = self.s0 + transformed_points * (max_price - min_price) / 2
        self.bin_edges[0] = min_price
        self.bin_edges[-1] = max_price
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def update(self, t: int, S: np.ndarray, l_t: np.ndarray):
        """Update the leverage surface and interpolation function for a time step."""
        # Update surface
        bin_indices = np.digitize(S, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.bins - 1)
        for i in range(self.bins):
            mask = (bin_indices == i)
            if np.sum(mask) > 0:
                self.leverage_surface[t, i] = np.mean(l_t[mask])
        
        # Update interpolation
        sort_indices = np.argsort(S)
        S_sorted, l_t_sorted = S[sort_indices], l_t[sort_indices]
        unique_S, unique_indices = np.unique(S_sorted, return_index=True)
        if len(unique_S) < len(S_sorted):
            unique_l_t = np.array([np.mean(l_t_sorted[S_sorted == s]) for s in unique_S])
            S_sorted, l_t_sorted = unique_S, unique_l_t
        
        if len(S_sorted) >= 2:
            self.leverage_interpolation[t] = interp1d(
                S_sorted, l_t_sorted, kind='linear', bounds_error=False,
                fill_value=(l_t_sorted[0], l_t_sorted[-1])
            )
        else:
            mean_leverage = np.mean(l_t) if len(l_t) > 0 else 1.0
            self.leverage_interpolation[t] = lambda x: np.full_like(x, mean_leverage)

    def get_leverage(self, t: int, S: np.ndarray) -> np.ndarray:
        """Get interpolated leverage values for a given time step and stock prices."""
        if t in self.leverage_interpolation:
            return self.leverage_interpolation[t](S)
        else:
            bin_indices = np.digitize(S, self.bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, self.bins - 1)
            return self.leverage_surface[t, bin_indices]

class ConditionalExpectation(ABC):
    """Abstract base class for conditional expectation computation strategies."""
    def __init__(self, bin_edges: np.ndarray, bin_centers: np.ndarray):
        self.bin_edges = bin_edges
        self.bin_centers = bin_centers
        self.bins = len(bin_centers)

    @abstractmethod
    def compute(self, S: np.ndarray, sigma_sq: np.ndarray) -> np.ndarray:
        """Computes E[sigma^2 | S]."""
        pass

class BinConditionalExpectation(ConditionalExpectation):
    """Computes conditional expectation using binning."""
    def compute(self, S: np.ndarray, sigma_sq: np.ndarray) -> np.ndarray:
        bin_indices = np.digitize(S, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.bins - 1)
        cond_exp = np.zeros_like(S)
        for i in range(self.bins):
            mask = (bin_indices == i)
            if np.any(mask):
                cond_exp[mask] = np.mean(sigma_sq[mask])
        return cond_exp

class KernelConditionalExpectation(ConditionalExpectation):
    """Computes conditional expectation using a kernel method."""
    def compute(self, S: np.ndarray, sigma_sq: np.ndarray) -> np.ndarray:
        n = len(S)
        std_S = np.std(S)
        bandwidth = 1e-3 * np.mean(S) if std_S < 1e-8 else 5 * std_S * n**(-1/5)
        
        cond_exp_bins = np.zeros_like(self.bin_centers)
        for i, center in enumerate(self.bin_centers):
            kernel_weights = np.exp(-0.5 * ((S - center) / bandwidth)**2)
            kernel_sum = np.sum(kernel_weights)
            if kernel_sum < 1e-10:
                cond_exp_bins[i] = np.mean(sigma_sq)
            else:
                cond_exp_bins[i] = np.sum(kernel_weights * sigma_sq) / kernel_sum
    
        interpolator = interp1d(
            self.bin_centers, cond_exp_bins, bounds_error=False, 
            fill_value=(cond_exp_bins[0], cond_exp_bins[-1])
        )
        return interpolator(S)

class ParticleMonteCarlo:
    def __init__(
        self,
        model,
        num_paths: int = 1_000,
        num_particles: int = 10_000,
        time_steps: int = 252,
        dt: float = 1.0 / 252,
        bins: int = 100,
        rng: np.random.Generator = np.random.default_rng(),
        method: str = 'kernel'  # Method for computing conditional expectation ('bin' or 'kernel')
    ):
        """
        Monte Carlo simulator for Path-Dependent Volatility models.
        
        Args:
            model: PDV Model instance with path-dependent factors
            num_paths: Number of paths for final simulation (after leverage calibration)
            num_particles: Number of particles for leverage surface initialization
            time_steps: Number of time steps for simulation
            dt: Time step size
            bins: Number of bins for discretizing the stock price range
            rng: Random number generator
            method: Method for computing conditional expectation ('bin' or 'kernel')
        """
        self.model = model
        self.num_paths = num_paths
        self.num_particles = num_particles
        self.time_steps = time_steps
        self.dt = dt
        self.rng = rng
        
        # Initialize the leverage surface manager
        self.leverage_manager = LeverageSurface(time_steps, bins, model.s0)
        
        # Select the conditional expectation computation strategy based on the method
        if method == 'kernel':
            self.cond_exp_computer = KernelConditionalExpectation(
                self.leverage_manager.bin_edges, self.leverage_manager.bin_centers
            )
        elif method == 'bin':
            self.cond_exp_computer = BinConditionalExpectation(
                self.leverage_manager.bin_edges, self.leverage_manager.bin_centers
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Initialize with particle count for leverage calibration phase
        self._initialize_paths(self.num_particles)
        
        self.leverage_exists = False  # Flag to indicate if leverage interpolation exists

    def _initialize_paths(self, size: int) -> None:
        """
        Initialize path arrays with given size.
        
        Args:
            size: Number of paths/particles to initialize
        """
        self.model.pathS = np.zeros((size, self.time_steps + 1))
        self.model.pathS[:, 0] = self.model.s0
        
        # Initialize pathX with appropriate dimensions
        x_dimensions = getattr(self.model, 'x_dimensions', 1)
        self.model.pathX = np.zeros((size, self.time_steps + 1, x_dimensions))
        
        # Initialize each path with x0
        for i in range(x_dimensions):
            self.model.pathX[:, 0, i] = self.model.x0[i]

    def initialize_leverage(self) -> None:
        """
        Run a preliminary simulation to initialize the leverage surface.
        This method uses a larger number of particles (num_particles) to get an accurate
        leverage surface, then prepares for the main simulation with num_paths.
        """
        if not self.leverage_exists:
            print(f"Using {self.num_particles} particles to initialize leverage surface")
            self._calibrate_leverage()
            self.leverage_exists = True
            
            # Reset paths for the main simulation
            self._initialize_paths(self.num_paths)

    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the PDV model simulation.
        
        If this is the first simulation, it will first initialize the leverage surface
        using a preliminary simulation, then run the main simulation using the computed
        leverage functions.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the simulated stock price paths 
            and state variable paths.
        """
        print("Simulating PDVModel")
        if not self.leverage_exists:
            self.initialize_leverage()
        
        self._simulate_paths()
        return self.model.pathS, self.model.pathX

    def _calibrate_leverage(self) -> None:
        """
        Internal method to calibrate the leverage surface using a high number of particles.
        """
        # Get current number of paths being simulated
        current_size = self.model.pathS.shape[0]
        
        dW_S = self.model.generate_brownian_motions(
            current_size, self.time_steps, rng=self.rng
        )
        
        for t in range(1, self.time_steps + 1):
            S_prev = self.model.pathS[:, t-1]
            
            # Extract X_prev based on its dimensions
            if self.model.pathX.ndim == 3:  # Vector-valued X
                X_prev = self.model.pathX[:, t-1, :]
            else:  # Backwards compatibility for scalar X
                X_prev = self.model.pathX[:, t-1]
            
            sigma_sq = self.model.sigma(t-1, S_prev, X_prev) ** 2
            
            cond_exp = self.cond_exp_computer.compute(S_prev, sigma_sq)
            
            l_t = self._compute_leverage(t-1, S_prev, cond_exp)
            
            self.leverage_manager.update(t-1, S_prev, l_t)
            
            # Evolve stock price using the computed leverage
            self.model.pathS[:, t] = S_prev * (
                1 + self.model.sigma(t-1, S_prev, X_prev) * l_t * dW_S[:, t-1]
            )
                
            self._update_pathX(t)

    def _simulate_paths(self) -> None:
        """
        Internal method to simulate paths using the calibrated leverage surface.
        """
        # Get current number of paths being simulated
        current_size = self.model.pathS.shape[0]
        
        dW_S = self.model.generate_brownian_motions(
            current_size, self.time_steps, rng=self.rng
        )
        
        for t in range(1, self.time_steps + 1):
            S_prev = self.model.pathS[:, t-1]
            
            # Extract X_prev based on its dimensions
            if self.model.pathX.ndim == 3:  # Vector-valued X
                X_prev = self.model.pathX[:, t-1, :]
            else:  # Backwards compatibility for scalar X
                X_prev = self.model.pathX[:, t-1]

            # Directly use the interpolated leverage
            local_vol = self.model.sigma(t-1, S_prev, X_prev)
            l_t_interp = self.leverage_manager.get_leverage(t-1, S_prev)
            self.model.pathS[:, t] = S_prev * (
                1 + local_vol * l_t_interp * dW_S[:, t-1]
            )
                
            self._update_pathX(t)

    def _update_pathX(self, t: int) -> None:
        """Helper method to update the state variable paths."""
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
                    new_pathX = np.zeros((self.model.pathS.shape[0], self.time_steps + 1, x_dimensions))
                    new_pathX[:, :t, 0] = self.model.pathX[:, :t]  # Copy existing data to first dimension
                    new_pathX[:, t, :] = X_new  # Set new value for all dimensions
                    self.model.pathX = new_pathX
        else:  # Legacy case where update_X returns (num_particles,)
            if self.model.pathX.ndim == 3:  # Vector-valued X with single dimension
                self.model.pathX[:, t, 0] = X_new
            else:  # Legacy scalar X
                self.model.pathX[:, t] = X_new

    def _compute_leverage(
        self,
        t: int,
        S: np.ndarray,
        cond_exp_sigma_sq: np.ndarray
    ) -> np.ndarray:
        
        sigma_dup = self.model.dupire_vol(t * self.dt, S)
        return sigma_dup / (np.sqrt(cond_exp_sigma_sq) + 1e-6)  # Avoid division by zero

if __name__ == "__main__":
    # Example usage
    from pdv_model import PDVModel
    import time
    
    # Create PDV model
    pdv_model = PDVModel(s0=100, x0=100)
    pdv_model.dupire_vol_interp = lambda t, S: 0.2 # dummy Dupire volatility function
    
    # Create and run particle simulator
    particle_mc = ParticleMonteCarlo(
        model=pdv_model,
        num_paths=1_000,      # Number of paths for final simulation 
        num_particles=10_000, # Number of particles for leverage calibration
        time_steps=252,
        bins=100,
        method='kernel'
    )
    
    start_time = time.time()
    pathS, pathX = particle_mc.simulate()
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds")

    import matplotlib.pyplot as plt
    # Plot leverage surface
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    s0 = pdv_model.s0
    min_val = 50
    max_val = 150
    
    n_points = 200
    alpha_viz = 1.5
    points_viz = np.linspace(-1, 1, n_points)
    transformed_points_viz = np.tanh(alpha_viz * points_viz) / np.tanh(alpha_viz)
    S_vals = s0 + transformed_points_viz * (max_val - min_val) / 2
    

    n_time_points = 50
    t_vals = np.linspace(1, particle_mc.time_steps - 1, n_time_points, dtype=int)  # Uniformly sample 50 time points
    
    # Create a single figure with two subplots
    fig = plt.figure(figsize=(18, 8))
    
    # First subplot - 2D heatmap with interpolation
    ax1 = fig.add_subplot(121)
    
    # Create a higher resolution 2D grid for better visualization
    S_grid_fine, T_grid_fine = np.meshgrid(S_vals, t_vals)
    
    # Use get_leverage to compute interpolated leverage values
    leverage_interp = np.zeros_like(S_grid_fine)
    for i, t in enumerate(t_vals):
        leverage_interp[i, :] = particle_mc.leverage_manager.get_leverage(t, S_vals)
    
    # Plot the interpolated 2D heatmap
    pcm = ax1.pcolormesh(T_grid_fine, S_grid_fine, leverage_interp, cmap='viridis', shading='auto')
    fig.colorbar(pcm, ax=ax1, label='Leverage')
    ax1.set_title('Interpolated Leverage Surface (2D)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Stock Price (S)')
    
    # Second subplot - 3D surface with interpolation
    ax2 = fig.add_subplot(122, projection='3d')
    
    # For 3D surface plot, use the same interpolated grid
    S_grid_3d, T_grid_3d = np.meshgrid(S_vals, t_vals)
    
    # Plot the surface using the interpolated data
    surf = ax2.plot_surface(S_grid_3d, T_grid_3d, leverage_interp, 
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
    
    time_slice_positions = [0.05, 0.25, 0.5, 0.75, 0.99]
    time_slices = [max(1, int(pos * (particle_mc.time_steps - 1))) for pos in time_slice_positions]
    plt.figure(figsize=(14, 8))
    
    for i, t in enumerate(time_slices):
        leverage_slice = particle_mc.leverage_manager.get_leverage(t, S_vals)
        plt.plot(S_vals, leverage_slice, label=f'T={t/(particle_mc.time_steps-1):.2f}', linewidth=2)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Stock Price', fontsize=12)
    plt.ylabel('Leverage Factor', fontsize=12)
    plt.title('Leverage Factor Cross-Sections at Different Time Points', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    # Add visualizations for stock price paths (pathS) and state variable paths (pathX)
    # Randomly select 10 paths to display
    np.random.seed(42)  # Fix random seed for reproducibility
    path_indices = np.random.choice(particle_mc.num_paths, size=10, replace=False)
    
    # Create time grid for plotting
    time_grid = np.arange(particle_mc.time_steps + 1) * particle_mc.dt
    
    # Plot stock price paths
    plt.figure(figsize=(14, 6))
    for idx in path_indices:
        plt.plot(time_grid, pathS[idx], alpha=0.7, linewidth=1)
    
    plt.axhline(y=pdv_model.s0, color='r', linestyle='--', label='Initial Price S0')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('Stock Price (S)', fontsize=12)
    plt.title('10 Randomly Selected Stock Price Paths', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    # Plot state variable paths (if scalar)
    if pathX.ndim == 2:
        plt.figure(figsize=(14, 6))
        for idx in path_indices:
            plt.plot(time_grid, pathX[idx], alpha=0.7, linewidth=1)
        
        plt.axhline(y=pdv_model.x0, color='r', linestyle='--', label='Initial State X0')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time (t)', fontsize=12)
        plt.ylabel('State Variable (X)', fontsize=12)
        plt.title('10 Randomly Selected State Variable Paths', fontsize=14)
        plt.legend()
        plt.tight_layout()
    # If X is multi-dimensional, create a subplot for each dimension
    elif pathX.ndim == 3:
        x_dim = pathX.shape[2]
        fig, axs = plt.subplots(x_dim, 1, figsize=(14, 4*x_dim), sharex=True)
        
        # If there's only one dimension, axs won't be an array and needs special handling
        if x_dim == 1:
            axs = [axs]
            
        for dim in range(x_dim):
            for idx in path_indices:
                axs[dim].plot(time_grid, pathX[idx, :, dim], alpha=0.7, linewidth=1)
            
            # Add initial state line
            if isinstance(pdv_model.x0, (list, tuple, np.ndarray)) and len(pdv_model.x0) > dim:
                axs[dim].axhline(y=pdv_model.x0[dim], color='r', linestyle='--', 
                               label=f'Initial State X0[{dim}]')
            
            axs[dim].grid(True, alpha=0.3)
            axs[dim].set_ylabel(f'State Variable X[{dim}]', fontsize=12)
            axs[dim].set_title(f'10 Random Paths for Dimension {dim}', fontsize=12)
            axs[dim].legend()
        
        # Set x-axis label for the bottom subplot
        axs[-1].set_xlabel('Time (t)', fontsize=12)
        plt.tight_layout()

    plt.show()


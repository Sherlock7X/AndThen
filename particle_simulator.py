from typing import Callable, Tuple
import numpy as np

class ParticleMonteCarlo:
    def __init__(
        self,
        model,
        num_paths: int = 1_000,
        num_particles: int = 10_000,
        time_steps: int = 252,
        dt: float = 1.0 / 252,
        bins: int = 200,
        rng: np.random.Generator = np.random.default_rng(),
        method: str = 'bin'  # Method for computing conditional expectation ('bin' or 'kernel')
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
        self.method = method
        
        # Initialize with particle count for leverage calibration phase
        self._initialize_paths(self.num_particles)
        
        self.bins = bins
        
        # 创建非均匀的价格边界，在ATM附近更密集
        # 使用双曲正切（tanh）函数创建非线性分布
        s0 = model.s0  # 当前价格（ATM）
        min_price = 0.5 * s0
        max_price = 1.5 * s0
        
        # 使用双曲正切函数创建非均匀分布的边界点，在ATM附近更密集
        # 参数控制密度：alpha越大，ATM附近的点越密集
        alpha = 1.5
        points = np.linspace(-1, 1, bins + 1)
        
        # 应用变换：在0（对应ATM）附近压缩，在边缘拉伸
        transformed_points = np.tanh(alpha * points) / np.tanh(alpha)
        
        # 将转换后的点映射到价格范围
        self.bin_edges = s0 + transformed_points * (max_price - min_price) / 2
        
        # 确保边界点严格递增且覆盖整个范围
        self.bin_edges[0] = min_price
        self.bin_edges[-1] = max_price
        
        # 初始化杠杆表面
        self.leverage_surface = np.full((time_steps, bins), np.nan)
        self.leverage_interpolation = {}
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
            # Run a preliminary simulation to initialize leverage surface
            self._PDV_simulate()
            self.leverage_exists = True
            
            # Reset paths for the main simulation with num_paths
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
            # First run: initialize leverage then perform main simulation
            self.initialize_leverage()
            self._PDV_simulate()
        else:
            # Subsequent runs: just perform simulation with existing leverage
            self._PDV_simulate()
        return self.model.pathS, self.model.pathX

    def _PDV_simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal method to simulate the Path-Dependent Volatility model.
        This implements the particle method to simulate path-dependent volatility
        with leverage adjustment.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Stock price paths and state variable paths
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
            if self.method == 'kernel':
                cond_exp = self._kernel_compute_conditional_expectation(S_prev, sigma_sq)
            elif self.method == 'bin':
                cond_exp = self._bin_compute_conditional_expectation(S_prev, sigma_sq)
            
            l_t = self._compute_leverage(t-1, S_prev, cond_exp)
            
            # Always update leverage surface and interpolation
            self._update_leverage_surface(t-1, S_prev, l_t)
            self._update_leverage_interpolation(t-1, S_prev, l_t)
            
            if self.leverage_exists:
                # If we have already initialized leverage, use it directly
                local_vol = self.model.sigma(t-1, S_prev, X_prev)
                l_t_interp = self.get_leverage(t-1, S_prev)
                self.model.pathS[:, t] = S_prev * (
                    1 + local_vol * l_t_interp * dW_S[:, t-1]
                )
            else:
                # First run, use computed leverage from this iteration
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
            # surface smoothness depends on the bandwidth parameter
            bandwidth = 5 * std_S * n**(-1/5)
        
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

    def _update_leverage_interpolation(
        self,
        t: int,
        S: np.ndarray,
        l_t: np.ndarray
    ) -> None:
        """
        Update the leverage interpolation function for time step t.
        This creates a callable interpolation function that can be used
        to compute leverage values for any stock price at time t.
        
        Args:
            t: Time step index
            S: Array of stock prices
            l_t: Array of leverage values
        """
        from scipy.interpolate import interp1d
        
        # Sort S and l_t by S values to ensure proper interpolation
        sort_indices = np.argsort(S)
        S_sorted = S[sort_indices]
        l_t_sorted = l_t[sort_indices]
        
        # Remove duplicate S values for interpolation by averaging the corresponding l_t values
        unique_S, unique_indices = np.unique(S_sorted, return_index=True)
        if len(unique_S) < len(S_sorted):
            unique_l_t = np.zeros_like(unique_S)
            for i, s in enumerate(unique_S):
                mask = (S_sorted == s)
                unique_l_t[i] = np.mean(l_t_sorted[mask])
            S_sorted = unique_S
            l_t_sorted = unique_l_t
        
        # Ensure we have enough points for interpolation
        if len(S_sorted) >= 2:
            self.leverage_interpolation[t] = interp1d(
                S_sorted, 
                l_t_sorted,
                kind='linear',
                bounds_error=False,
                fill_value=(l_t_sorted[0], l_t_sorted[-1])
            )
        else:
            mean_leverage = np.mean(l_t) if len(l_t) > 0 else 1.0
            self.leverage_interpolation[t] = lambda x: np.full_like(x, mean_leverage)

    def get_leverage(self, t: int, S: np.ndarray) -> np.ndarray:
        """
        Get interpolated leverage values for the given time step and stock prices.
        
        Args:
            t: Time step index
            S: Array of stock prices
            
        Returns:
            Interpolated leverage values for the given stock prices
        """
        if t in self.leverage_interpolation:
            # Use the stored interpolation function
            return self.leverage_interpolation[t](S)
        else:
            # Fall back to bin-based lookup if no interpolation is available
            bin_indices = np.digitize(S, self.bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, self.bins - 1)
            return self.leverage_surface[t, bin_indices]

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
        num_particles=50_000, # Number of particles for leverage calibration
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
    t_vals = np.linspace(1, particle_mc.time_steps - 1, n_time_points, dtype=int)  # 均匀采样50个时间点
    
    # Create a single figure with two subplots
    fig = plt.figure(figsize=(18, 8))
    
    # First subplot - 2D heatmap with interpolation
    ax1 = fig.add_subplot(121)
    
    # Create a higher resolution 2D grid for better visualization
    S_grid_fine, T_grid_fine = np.meshgrid(S_vals, t_vals)
    
    # Use get_leverage to compute interpolated leverage values
    leverage_interp = np.zeros_like(S_grid_fine)
    for i, t in enumerate(t_vals):
        leverage_interp[i, :] = particle_mc.get_leverage(t, S_vals)
    
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
        leverage_slice = particle_mc.get_leverage(t, S_vals)
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
    

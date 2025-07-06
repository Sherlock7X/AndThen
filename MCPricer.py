import numpy as np
from typing import Dict, Union, List, Tuple, Optional
from BasePricer import BasePricer
from particle_simulator import ParticleMonteCarlo
from LocalVolSimulator import LocalVolSimulator

class MCPricer(BasePricer):
    """
    Monte Carlo pricer for options using simulation-based pricing.
    This pricer works with models that support path simulation.
    """
    
    def __init__(
            self,
            model,
            risk_free_rate: float = 0.0,
            dividend_yield: float = 0.0,
            use_antithetic: bool = True,
            use_control_variate: bool = False,
            use_sobol: bool = False,
            rng: np.random.Generator = np.random.default_rng()
        ):
        """
        Initialize the Monte Carlo pricer.
        
        Args:
            model: Model instance that supports path simulation
            risk_free_rate: Annual risk-free interest rate (decimal)
            dividend_yield: Annual dividend yield (decimal)
            use_antithetic: Whether to use antithetic variates for variance reduction
            use_control_variate: Whether to use control variates for variance reduction
            use_sobol: Whether to use Sobol sequences for sampling
            rng: Random number generator
        """
        super().__init__(model, risk_free_rate, dividend_yield, use_antithetic, use_control_variate)
        self.use_sobol = use_sobol
        self.rng = rng
        self.simulator = None
        self.paths = None
        self.path_x = None
        
    def setup_simulation(
            self, 
            num_paths: int = 1000,
            num_particles: int = 10000,
            time_steps: int = 252,
            dt: float = 1/252,
            bins: int = 200,
            method: str = 'bin',
            use_local_vol_simulator: bool = None
        ) -> None:
        """
        Setup the Monte Carlo simulation environment.
        
        Args:
            num_paths: Number of paths for final simulation
            num_particles: Number of particles for leverage calibration (PDV model only)
            time_steps: Number of time steps for simulation
            dt: Time step size (default: 1/252 for daily steps)
            bins: Number of bins for discretizing the stock price range
            method: Method for computing conditional expectation ('bin' or 'kernel')
            use_local_vol_simulator: Whether to use LocalVolSimulator. If None,
                                    will auto-detect based on model properties.
        """
        self.num_paths = num_paths
        self.time_steps = time_steps
        self.dt = dt
        
        # Double the number of paths if using antithetic variates
        effective_paths = num_paths * 2 if self.use_antithetic else num_paths
        
        # Auto-detect if we should use LocalVolSimulator
        if use_local_vol_simulator is None:
            # Check if model has local_vol method (needed for LocalVolSimulator)
            use_local_vol_simulator = hasattr(self.model, 'local_vol') and not hasattr(self.model, 'sigma')
        
        # Initialize the appropriate simulator
        if use_local_vol_simulator:
            print("Using LocalVolSimulator for local volatility model")
            self.simulator = LocalVolSimulator(
                model=self.model,
                num_particles=effective_paths,  # For LocalVolSimulator, particles = paths
                time_steps=time_steps,
                dt=dt,
                rng=self.rng
            )
        else:
            print("Using ParticleMonteCarlo for path-dependent volatility model")
            # For PDV model, we need both num_paths and num_particles
            self.simulator = ParticleMonteCarlo(
                model=self.model,
                num_paths=effective_paths,  # For final simulation
                num_particles=num_particles,  # For leverage calibration
                time_steps=time_steps,
                dt=dt,
                bins=bins,
                rng=self.rng,
                method=method
            )
        
    def run_simulation(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run the simulation to generate paths.
        
        Returns:
            If using a local volatility model: just pathS (ndarray)
            Otherwise: Tuple containing (pathS, pathX)
        """
        if self.simulator is None:
            raise ValueError("Simulation environment not set up. Call setup_simulation first.")
        
        # Check if we're using a local volatility model (LocalVolSimulator)
        if isinstance(self.simulator, LocalVolSimulator):
            # Local volatility model only returns paths, not path_x
            self.paths = self.simulator.simulate()
            self.path_x = None
            return self.paths
        else:
            # Path-dependent volatility model returns both paths and path_x
            self.paths, self.path_x = self.simulator.simulate()
            return self.paths, self.path_x
    
    def price(
            self, 
            option_type: str, 
            strike: float, 
            expiry: float,
            payoff_func: str = 'european',
            **kwargs
        ) -> Dict[str, float]:
        """
        Price an option using Monte Carlo simulation.
        
        Args:
            option_type: Type of option ('call', 'put', etc.)
            strike: Strike price
            expiry: Time to expiry in years
            payoff_func: Type of payoff function ('european', 'digital', or custom function)
            
        Returns:
            Dictionary containing the price and error estimates
        """
        if self.paths is None:
            self.run_simulation()
            
        # Convert expiry to the corresponding time step
        expiry_step = min(int(expiry / self.dt), self.time_steps)
        
        # Get terminal stock prices
        terminal_prices = self.paths[:, expiry_step]
        
        # Calculate payoffs
        if payoff_func == 'european':
            payoffs = self.payoff_european(option_type, terminal_prices, strike)
        elif payoff_func == 'digital':
            payoffs = self.payoff_digital(option_type, terminal_prices, strike)
        elif callable(payoff_func):
            # Custom payoff function
            if self.path_x is not None:
                # For path-dependent models with state variables
                payoffs = payoff_func(terminal_prices, self.path_x[:, expiry_step], strike)
            else:
                # For local volatility models with no state variables
                payoffs = payoff_func(terminal_prices, None, strike)
        else:
            raise ValueError(f"Unsupported payoff function: {payoff_func}")
            
        # Apply discount factor
        discounted_payoffs = self.discount(payoffs, expiry)
        
        # Calculate price (mean of discounted payoffs)
        price = np.mean(discounted_payoffs)
        
        # Calculate standard error
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
        
        return {
            'price': price,
            'std_error': std_error,
            'conf_interval_95': [price - 1.96 * std_error, price + 1.96 * std_error]
        }
        
    def price_barrier_option(
            self,
            option_type: str,
            barrier_type: str,
            strike: float,
            barrier: float,
            expiry: float
        ) -> Dict[str, float]:
        """
        Price a barrier option using Monte Carlo simulation.
        
        Args:
            option_type: Type of option ('call', 'put')
            barrier_type: Type of barrier ('up-and-in', 'up-and-out', 'down-and-in', 'down-and-out')
            strike: Strike price
            barrier: Barrier level
            expiry: Time to expiry in years
            
        Returns:
            Dictionary containing the price and error estimates
        """
        if self.paths is None:
            self.run_simulation()
            
        # Convert expiry to the corresponding time step
        expiry_step = min(int(expiry / self.dt), self.time_steps)
        
        # Get paths up to expiry
        paths_to_expiry = self.paths[:, :expiry_step+1]
        
        # Check barrier conditions
        if barrier_type == 'up-and-in':
            barrier_triggered = np.max(paths_to_expiry, axis=1) >= barrier
        elif barrier_type == 'up-and-out':
            barrier_triggered = np.max(paths_to_expiry, axis=1) < barrier
        elif barrier_type == 'down-and-in':
            barrier_triggered = np.min(paths_to_expiry, axis=1) <= barrier
        elif barrier_type == 'down-and-out':
            barrier_triggered = np.min(paths_to_expiry, axis=1) > barrier
        else:
            raise ValueError(f"Unsupported barrier type: {barrier_type}")
            
        # Get terminal prices and calculate base payoffs
        terminal_prices = paths_to_expiry[:, -1]
        base_payoffs = self.payoff_european(option_type, terminal_prices, strike)
        
        # Apply barrier condition
        payoffs = base_payoffs * barrier_triggered
        
        # Apply discount factor
        discounted_payoffs = self.discount(payoffs, expiry)
        
        # Calculate price and error
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
        
        return {
            'price': price,
            'std_error': std_error,
            'conf_interval_95': [price - 1.96 * std_error, price + 1.96 * std_error]
        }
        
    def price_asian_option(
            self,
            option_type: str,
            strike: float,
            expiry: float,
            averaging_type: str = 'arithmetic',
            averaging_freq: str = 'daily'
        ) -> Dict[str, float]:
        """
        Price an Asian option using Monte Carlo simulation.
        
        Args:
            option_type: Type of option ('call', 'put')
            strike: Strike price
            expiry: Time to expiry in years
            averaging_type: Type of averaging ('arithmetic', 'geometric')
            averaging_freq: Frequency of averaging ('daily', 'weekly', 'monthly')
            
        Returns:
            Dictionary containing the price and error estimates
        """
        if self.paths is None:
            self.run_simulation()
            
        # Convert expiry to the corresponding time step
        expiry_step = min(int(expiry / self.dt), self.time_steps)
        
        # Get paths up to expiry
        paths_to_expiry = self.paths[:, :expiry_step+1]
        
        # Determine sampling frequency
        if averaging_freq == 'daily':
            sampling_steps = 1
        elif averaging_freq == 'weekly':
            sampling_steps = int(5 / self.dt)  # Assuming 5 days per week
        elif averaging_freq == 'monthly':
            sampling_steps = int(21 / self.dt)  # Assuming 21 days per month
        else:
            raise ValueError(f"Unsupported averaging frequency: {averaging_freq}")
            
        # Select prices based on frequency
        sampled_prices = paths_to_expiry[:, ::sampling_steps]
        
        # Calculate average prices
        if averaging_type == 'arithmetic':
            avg_prices = np.mean(sampled_prices, axis=1)
        elif averaging_type == 'geometric':
            avg_prices = np.exp(np.mean(np.log(sampled_prices), axis=1))
        else:
            raise ValueError(f"Unsupported averaging type: {averaging_type}")
            
        # Calculate payoffs
        payoffs = self.payoff_european(option_type, avg_prices, strike)
        
        # Apply discount factor
        discounted_payoffs = self.discount(payoffs, expiry)
        
        # Calculate price and error
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
        
        return {
            'price': price,
            'std_error': std_error,
            'conf_interval_95': [price - 1.96 * std_error, price + 1.96 * std_error]
        }
    
    def price_forward_start(
            self,
            option_type: str,
            strike: float,
            start_time: float,  # Time when the option actually starts
            expiry: float,      # Total time to expiry from now (includes start_time)
            **kwargs
        ) -> Dict[str, float]:
        """
        Price a forward start option using Monte Carlo simulation.
        
        A forward start option begins at a future date (start_time) with a strike price
        that's determined as a percentage of the underlying price at that future date.
        
        Args:
            option_type: Type of option ('call', 'put')
            strike_pct: Strike price as a percentage of the underlying price at start_time (e.g., 1.0 for ATM)
            start_time: Time when the option begins (years from now)
            expiry: Total time to expiry from now (years, includes start_time)
            
        Returns:
            Dictionary containing the price and error estimates
        """
        if self.paths is None:
            self.run_simulation()
            
        # Convert times to corresponding time steps
        start_step = min(int(start_time / self.dt), self.time_steps)
        expiry_step = min(int(expiry / self.dt), self.time_steps)
        
        # Ensure start_step is before expiry_step
        if start_step >= expiry_step:
            raise ValueError("Start time must be before expiry time")
            
        # Get prices at the start date and terminal date
        start_prices = self.paths[:, start_step]
        terminal_prices = self.paths[:, expiry_step] / start_prices

        
        # Calculate payoffs (vectorized)
        if option_type.lower() == 'call':
            payoffs = np.maximum(terminal_prices - strike, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(strike - terminal_prices, 0)
        else:
            raise ValueError(f"Unsupported option type: {option_type}")
            
        # Apply discount factor (from now until expiry)
        discounted_payoffs = self.discount(payoffs, expiry)
        
        # Calculate price and error
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
        
        return {
            'price': price,
            'std_error': std_error,
            'conf_interval_95': [price - 1.96 * std_error, price + 1.96 * std_error]
        }

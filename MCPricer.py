import numpy as np
from typing import Dict, Union, List, Tuple, Optional
from BasePricer import BasePricer
from particle_simulator import ParticleMonteCarlo

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
            num_paths: int = 10000, 
            time_steps: int = 252,
            dt: float = 1/252,
            bins: int = 200,
            method: str = 'bin'
        ) -> None:
        """
        Setup the Monte Carlo simulation environment.
        
        Args:
            num_paths: Number of paths for simulation
            time_steps: Number of time steps for simulation
            dt: Time step size (default: 1/252 for daily steps)
            bins: Number of bins for discretizing the stock price range
            method: Method for computing conditional expectation ('bin' or 'kernel')
        """
        self.num_paths = num_paths
        self.time_steps = time_steps
        self.dt = dt
        
        # Double the number of paths if using antithetic variates
        effective_paths = num_paths * 2 if self.use_antithetic else num_paths
        
        # Initialize the simulator
        self.simulator = ParticleMonteCarlo(
            model=self.model,
            num_particles=effective_paths,
            time_steps=time_steps,
            dt=dt,
            bins=bins,
            rng=self.rng,
            method=method
        )
        
    def run_simulation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the simulation to generate paths.
        
        Returns:
            Tuple containing (pathS, pathX)
        """
        if self.simulator is None:
            raise ValueError("Simulation environment not set up. Call setup_simulation first.")
            
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
            payoffs = payoff_func(terminal_prices, self.path_x[:, expiry_step], strike)
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

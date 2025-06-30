from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Dict, List, Union, Optional, Tuple

class BasePricer(ABC):
    """
    Abstract base class for option pricing. This class defines the interface
    for pricing options with different models and methods.
    """
    
    def __init__(
            self,
            model,
            risk_free_rate: float = 0.0,
            dividend_yield: float = 0.0,
            use_antithetic: bool = True,
            use_control_variate: bool = False
        ):
        """
        Initialize the pricer with a model and pricing parameters.
        
        Args:
            model: The model that will be used for pricing (must implement certain interface)
            risk_free_rate: Annual risk-free interest rate (decimal)
            dividend_yield: Annual dividend yield (decimal)
            use_antithetic: Whether to use antithetic variates for variance reduction
            use_control_variate: Whether to use control variates for variance reduction
        """
        self.model = model
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.use_antithetic = use_antithetic
        self.use_control_variate = use_control_variate
        
    @abstractmethod
    def price(
            self, 
            option_type: str, 
            strike: float, 
            expiry: float,
            **kwargs
        ) -> Dict[str, float]:
        """
        Price an option.
        
        Args:
            option_type: Type of option ('call', 'put', etc.)
            strike: Strike price
            expiry: Time to expiry in years
            
        Returns:
            Dictionary containing the price and possibly other metrics
        """
        pass
    
    @abstractmethod
    def setup_simulation(
            self, 
            num_paths: int, 
            time_steps: int, 
            **kwargs
        ) -> None:
        """
        Setup the simulation environment.
        
        Args:
            num_paths: Number of paths for simulation
            time_steps: Number of time steps
        """
        pass
    
    def discount(self, cash_flow: np.ndarray, time: float) -> np.ndarray:
        """
        Discount a cash flow.
        
        Args:
            cash_flow: Cash flow array to discount
            time: Time in years
            
        Returns:
            Discounted cash flow
        """
        return cash_flow * np.exp(-self.risk_free_rate * time)
    
    def payoff_european(
            self, 
            option_type: str, 
            spot: np.ndarray, 
            strike: float
        ) -> np.ndarray:
        """
        Calculate the payoff for a European option.
        
        Args:
            option_type: Type of option ('call', 'put')
            spot: Spot price array
            strike: Strike price
            
        Returns:
            Option payoff array
        """
        if option_type.lower() == 'call':
            return np.maximum(spot - strike, 0)
        elif option_type.lower() == 'put':
            return np.maximum(strike - spot, 0)
        else:
            raise ValueError(f"Unsupported option type: {option_type}")
    
    def payoff_digital(
            self, 
            option_type: str, 
            spot: np.ndarray, 
            strike: float
        ) -> np.ndarray:
        """
        Calculate the payoff for a digital option.
        
        Args:
            option_type: Type of option ('call', 'put')
            spot: Spot price array
            strike: Strike price
            
        Returns:
            Option payoff array
        """
        if option_type.lower() == 'call':
            return (spot > strike).astype(float)
        elif option_type.lower() == 'put':
            return (spot < strike).astype(float)
        else:
            raise ValueError(f"Unsupported option type: {option_type}")
    
    def calc_greeks(
            self,
            option_type: str,
            strike: float,
            expiry: float,
            bump_size: float = 0.01
        ) -> Dict[str, float]:
        """
        Calculate option Greeks using bump-and-revalue.
        
        Args:
            option_type: Type of option ('call', 'put')
            strike: Strike price
            expiry: Time to expiry in years
            bump_size: Size of the bump for finite differences
            
        Returns:
            Dictionary containing the Greeks
        """
        # Store original spot
        original_spot = self.model.s0
        
        # Base price
        base_result = self.price(option_type, strike, expiry)
        base_price = base_result['price']
        
        # Delta - bump spot up
        self.model.s0 = original_spot * (1 + bump_size)
        price_up = self.price(option_type, strike, expiry)['price']
        
        # Delta - bump spot down
        self.model.s0 = original_spot * (1 - bump_size)
        price_down = self.price(option_type, strike, expiry)['price']
        
        # Reset spot
        self.model.s0 = original_spot
        
        # Calculate delta and gamma
        delta = (price_up - price_down) / (2 * bump_size * original_spot)
        gamma = (price_up - 2 * base_price + price_down) / ((bump_size * original_spot) ** 2)
        
        # Restore initial state and return results
        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
        }

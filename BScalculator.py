from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np

def black_scholes_price(sigma, S, K, T, r, q, option_type):
    """
    Calculates the Black-Scholes option price.

    Args:
        sigma (float): Volatility.
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiration (in years).
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        option_type (str): 'call' or 'put'.

    Returns:
        float: The Black-Scholes price of the option.
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price

def implied_volatility(target_price, S, K, T, r, q, option_type):
    """
    Calculates the implied volatility for a given option price.

    This function uses the Brent's method to find the root of the Black-Scholes
    pricing formula, which corresponds to the implied volatility.

    Args:
        target_price (float): The market price of the option.
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiration (in years).
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        option_type (str): 'call' or 'put'.

    Returns:
        float: The implied volatility, or np.nan if a solution cannot be found
               or if an arbitrage opportunity is detected.
    """
    # Calculate the intrinsic value of the option
    if option_type.lower() == 'call':
        intrinsic_value = max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
    else:
        intrinsic_value = max(0, K * np.exp(-r * T) - S * np.exp(-q * T))

    # No-arbitrage check: The option price cannot be below its intrinsic value.
    if target_price < intrinsic_value:
        return np.nan  # Arbitrage opportunity detected

    # If the price is very close to the intrinsic value, volatility is near zero.
    # This handles deep in-the-money or out-of-the-money options.
    if abs(target_price - intrinsic_value) < 1e-6:
        return 1e-6

    try:
        # Define the objective function for the root-finding algorithm.
        # The goal is to find sigma such that the BS price equals the target price.
        def objective(sigma):
            if sigma < 0:  # Ensure volatility is non-negative
                return np.inf
            return black_scholes_price(sigma, S, K, T, r, q, option_type) - target_price
        
        # Use Brent's method to find the implied volatility.
        # The search range [1e-6, 5.0] is a practical choice for most scenarios.
        implied_vol = brentq(objective, 1e-6, 5.0)
        return implied_vol
    except ValueError:
        # If brentq fails to find a root within the given range,
        # it raises a ValueError.
        return np.nan
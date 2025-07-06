from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np

def black_scholes_price(sigma, S, K, T, r, q, option_type):
    """
    计算Black-Scholes期权价格
    
    参数:
        sigma: 波动率
        S: 当前股价
        K: 行权价
        T: 到期时间
        r: 无风险利率
        q: 股息率
        option_type: 'call' 或 'put'
    """
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price

def implied_volatility(target_price, S, K, T, r, q, option_type):
    """
    计算隐含波动率
    
    参数:
        target_price: 目标期权价格
        S, K, T, r, q, option_type: 同上
    """
    try:
        # 定义用于求解的函数
        def objective(sigma):
            return black_scholes_price(sigma, S, K, T, r, q, option_type) - target_price
        
        # 使用Brent方法求解隐含波动率
        implied_vol = brentq(objective, 0.0001, 2.0)
        return implied_vol
    except:
        # 如果求解失败，返回NaN
        return np.nan
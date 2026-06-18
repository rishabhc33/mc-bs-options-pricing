import numpy as np
from scipy.stats import norm

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float) -> tuple:
    """
    Calculates the theoretical fair value of European call and put options 
    using the analytical Black-Scholes formula.

    Parameters:
    S : float       - Current spot price
    K : float       - Option Strike Price
    T : float       - Time to expiry in years
    r : float       - Risk-free interest rate
    sigma : float   - Annual volatility
    -----------
    Returns:
    call_price : float  - Analytical present value of the Call option
    put_price : float   - Analytical present value of the Put option
    """

    if T<=0: 
        return max(S-K, 0.0), max(K-S, 0.0) 
    
    d1 = np.log(S/K) + (r+sigma**2/2)*T
    d1 /= sigma * np.sqrt(T)    
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    put_price = K*np.exp(-r*T) * norm.cdf(-d2) - S*norm.cdf(-d1)

    return call_price, put_price
    
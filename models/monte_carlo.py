import numpy as np

def simulate_gbm(S: float, T:float, r: float, sigma: float, tsteps: int, num_paths: int):
    """
    Simulates Geometric Brownian Motion (GBM) paths using vectorized numpy operations.

    Parameters:
    S : float       - Current spot price
    T : float       - Time to expiry in years
    r : float       - Risk-free interest rate
    sigma : float   - Annual volatility
    tsteps : int     - Number of time steps
    num_paths : int   - Number of simulated paths to generate
    """

    dt = T / tsteps
    Z = np.random.normal(0, 1, (num_paths, tsteps))

    drift = (r - 0.5*sigma**2)*dt
    diffusion = sigma * np.sqrt(dt)

    path_exp = np.cumsum(drift + diffusion*Z, axis=1)
    paths = np.zeros((num_paths, tsteps+1))
    paths[:, 0] = S
    paths[:, 1:] = S * np.exp(path_exp)

    return paths

def monte_carlo_price(paths: np.ndarray, K: float, r: float, T: float):
    """
    Calculates the fair price of European call and put options by averaging the 
    discounted final payoffs from simulated Monte Carlo paths.
    
    Parameters:
    paths : np.ndarray  - 2D array of simulated asset paths, output from simulate_gbm
    K : float           - Option Strike Price
    r : float           - Risk-free interest rate )
    T : float           - Time to expiry in years
    --------
    Returns:
    call_price : float  - Estimated present value of the Call option
    put_price : float   - Estimated present value of the Put option
    """

    final_price = paths[:, -1]
    call_payoffs = np.maximum(final_price-K, 0)
    put_payoffs = np.maximum(K-final_price, 0)
    discount_factor = np.exp(-r*T)

    call_price = float(np.mean(call_payoffs)*discount_factor)
    put_price = float(np.mean(put_payoffs)*discount_factor)

    return call_price, put_price

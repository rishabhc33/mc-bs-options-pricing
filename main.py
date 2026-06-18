import argparse
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from models.black_scholes import black_scholes_price
from models.monte_carlo import simulate_gbm, monte_carlo_price

def market_parameters(ticker_symbol: str) -> tuple:
    """
    Fetches the live spot price of a given stock ticker and 
    calculates its annual historical volatility.
    """
    
    print(f"\nFetching live market data for {ticker_symbol} from Yahoo Finance...")
    ticker = yf.Ticker(ticker_symbol)
    
    history = ticker.history(period="1d")
    if history.empty:
        raise ValueError(f"Could not retrieve data for {ticker_symbol}. Check the ticker symbol.")
    spot_price = float(history['Close'].iloc[-1])

    vol_data = ticker.history(period="1y")
    vol_data['Returns'] = np.log(vol_data['Close'] / vol_data['Close'].shift(1))
    daily_vol = vol_data['Returns'].std()
    annual_vol = float(daily_vol * np.sqrt(252))
    
    return spot_price, annual_vol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Black Scholes & Monte Carlo Option Pricing Engine")
    parser.add_argument(
        "-t", "--ticker", 
        type=str, 
        default="AAPL", 
        help="Stock Ticker"
    )
    args = parser.parse_args()
    ticker = args.ticker.upper()

    try:
        spot_price, volatility = market_parameters(ticker)
        print(f"Current Spot Price: ${spot_price:.2f}")
        print(f"Annual Volatility: {volatility*100:.2f}%")
    except Exception as e:
        print(f"Error fetching data: {e}")
        exit()

    strike_price = spot_price * 1.10  # 10% out of the money
    tte = 0.5   # 6 months
    risk_free_rate = 0.05  # 5% 
    tsteps = 126 # 126 trading days in half an year
    path_scenarios = [100, 500, 1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000]

    print(f"Evaluating Call Option at Strike: ${strike_price:.2f}\n")

    # Black Scholes
    bs_call, _ = black_scholes_price(spot_price, strike_price, tte, risk_free_rate, volatility)
    print(f"Black Scholes Call Price: ${bs_call:.4f}\n")

    # Monte Carlo
    print(f"Running MC Simulations")
    mc_call_prices = []
    for n in path_scenarios:
        paths = simulate_gbm(spot_price, tte, risk_free_rate, volatility, tsteps, n)
        mc_call, _ = monte_carlo_price(paths, strike_price, risk_free_rate, tte)
        mc_call_prices.append(mc_call)
        print(f"Paths: {n:5d} | MC Call Price: ${mc_call:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(path_scenarios, mc_call_prices, marker='o', linestyle='-', color='#1f77b4', label=f'{ticker} MC Estimate')
    plt.axhline(y=bs_call, color='#d62728', linestyle='--', linewidth=2, label=f'BS Exact (${bs_call:.4f})')
    plt.title(f"Monte Carlo Convergence vs. Black-Scholes ({ticker} Live Data)")
    plt.xlabel("Number of Simulated Paths ($N$)")
    plt.ylabel("Calculated Call Option Price ($)")
    plt.xscale('log')
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()
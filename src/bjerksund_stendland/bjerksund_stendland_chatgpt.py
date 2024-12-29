import math
from scipy.stats import norm

def bjerksund_stensland_call(S, K, T, r, q, sigma):
    """
    Calculate the price of an American call option using the Bjerksund-Stensland model.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free rate (annualized)
        q (float): Dividend yield (annualized)
        sigma (float): Volatility (annualized)

    Returns:
        float: American call option price
    """
    h = 1 - math.exp(-r * T)
    beta = (0.5 - q / sigma**2) + math.sqrt(((q / sigma**2) - 0.5)**2 + 2 * r / sigma**2)
    B0 = max(K, (beta / (beta - 1)) * K)
    B_inf = beta / (beta - 1) * K
    t1 = 0.5 * (math.sqrt(5) - 1) * T

    def call_price(S, K, r, q, sigma, T):
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    if S >= B0:
        return S - K

    phi = -1

    def F(S, B, T):
        return call_price(S, B, r, q, sigma, T) + (phi * (S / B)**beta) * (B - K)

    A2 = (B_inf / (beta - 1)) * (1 - math.exp(-h * (beta - 1) * T))
    A1 = K * (1 - math.exp(-r * T))

    if S < B0:
        return F(S, B0, T) + A2 - A1
    else:
        return S - K

# Example Usage
S = 100  # Current stock price
K = 95   # Strike price
T = 1    # Time to maturity (1 year)
r = 0.05 # Risk-free rate (5%)
q = 0.02 # Dividend yield (2%)
sigma = 0.2  # Volatility (20%)

price = bjerksund_stensland_call(S, K, T, r, q, sigma)
print(f"American Call Option Price: {price:.2f}")
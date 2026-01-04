import math
from scipy.stats import norm

def black_scholes_greeks(S, K, t, r, sigma, option_type="call"):
    """
    Calculate delta and vega for an option.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the stock
        option_type (str): "call" for call option, "put" for put option

    Returns:
        dict: Delta and Vega values
    """
    d1 = (math.log(S / K) + (r + (sigma**2) / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
    elif option_type.lower() == "put":
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Vega is the same for call and put options
    vega = S * norm.pdf(d1) * math.sqrt(t)

    return {"Delta": delta, "Vega": vega}

# Example usage
S = 254.77  # Current stock price
K = 250  # Strike price
t = 0.0199  # Time to maturity (1 year)
# https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_long_term_rate&field_tdr_date_value_month=202412
r = 0.0487  # Risk-free interest rate (5%)
sigma = 0.1924  # Volatility (20%)

greeks = black_scholes_greeks(S, K, t, r, sigma, option_type="call")
print(f"Call Option Delta: {greeks['Delta']:.4f}")
print(f"Option Vega: {greeks['Vega']:.4f}")
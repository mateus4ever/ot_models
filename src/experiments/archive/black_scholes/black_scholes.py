import math
from scipy.stats import norm

def black_scholes(S, K, t, r, sigma, option_type="call"):
    """
    Calculate the Black-Scholes option price.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the stock
        option_type (str): "call" for call option, "put" for put option

    Returns:
        float: Option price
    """
    # Calculate d1 and d2
    d1 = (math.log(S / K) + (r + (sigma ** 2) / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    if option_type.lower() == "call":
        # Call option price
        price = S * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    elif option_type.lower() == "put":
        # Put option price
        price = K * math.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price


# Example usage
S = 255.65  # Current stock price
K = 240  # Strike price
t = 0.0199  # Time to maturity (1 year)
# https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_long_term_rate&field_tdr_date_value_month=202412
r = 0.0487  # Risk-free interest rate (5%)
# sigma = 0.1924  # Volatility (20%)
sigma = 0.190966  # Volatility (20%)

call_price = black_scholes(S, K, t, r, sigma, option_type="call")
put_price = black_scholes(S, K, t, r, sigma, option_type="put")

print(f"Call Option Price: {call_price:.2f}")
print(f"Put Option Price: {put_price:.2f}")
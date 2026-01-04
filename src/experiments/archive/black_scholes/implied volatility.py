import math
from scipy.stats import norm

def black_scholes(S, K, t, r, sigma, option_type="call"):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    elif option_type == "put":
        return K * math.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

def vega(S, K, t, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    return S * norm.pdf(d1) * math.sqrt(t)

def implied_volatility(S, K, t, r, market_price, option_type="call", tol=1e-5, max_iter=100):
    sigma = 0.2  # Initial guess for volatility
    for i in range(max_iter):
        price = black_scholes(S, K, t, r, sigma, option_type)
        vega_value = vega(S, K, t, r, sigma)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega_value  # Newton-Raphson update
    raise ValueError("Implied volatility did not converge")

S = 255.65  # Current stock price
K = 250  # Strike price
t = 0.0199  # Time to maturity (1 year)
# https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_long_term_rate&field_tdr_date_value_month=202412
r = 0.0487  # Risk-free interest rate (5%)
market_price = 6.62  # Observed market price of the option
option_type = "call"

# Calculate implied volatility
iv = implied_volatility(S, K, t, r, market_price, option_type)
print(f"Implied Volatility: {iv:.4%}")
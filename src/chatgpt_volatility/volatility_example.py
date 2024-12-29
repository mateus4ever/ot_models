import numpy as np
import yfinance as yf
from scipy.stats import norm

# Step 1: Download historical stock price data
ticker = "AAPL"  # Example stock (Apple Inc.)
data = yf.download(ticker, start="2023-12-01", end="2024-12-03")
data["Daily Returns"] = data["Adj Close"].pct_change()

# Step 2: Calculate Historical Volatility
# Annualized volatility = std_dev of daily returns * sqrt(trading days in a year)
daily_volatility = data["Daily Returns"].std()
annualized_volatility = daily_volatility * np.sqrt(252)  # 252 trading days in a year

# Step 3: Black-Scholes Option Pricing
# Parameters
S = data["Adj Close"].iloc[-1]  # Current stock price (last close)
#K = S * 1.01  # Strike price (5% above current price)
K = 240  # Strike price

T = 183 / 365  # Time to expiration (30 days)
r = 0.03  # Risk-free rate (3%)s
sigma = annualized_volatility  # Historical volatility

# Black-Scholes formula
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Calculate the option price
option_price = black_scholes_call(S, K, T, r, sigma)

# Results
historical_volatility = annualized_volatility * 100  # As a percentage
result = {
    "Stock Price (S)": round(S, 2),
    "Strike Price (K)": round(K, 2),
    "Time to Expiry (T)": round(T, 3),
    "Annualized Volatility (%)": round(historical_volatility, 2),
    "Option Price (Call)": round(option_price, 2),
}
print(result)


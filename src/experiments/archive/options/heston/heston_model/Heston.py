import yfinance as yf
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Fetch historical data for Apple
ticker = "AAPL"
data = yf.Ticker(ticker)
hist_data = data.history(period="1y")  # 1 year of historical data
current_price = hist_data['Close'].iloc[-1]

# Calculate historical volatility (as a proxy for initial variance v0)
log_returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
v0 = np.var(log_returns.dropna())

# Define Heston parameters (these can be adjusted or estimated)
kappa = 2.0  # Mean-reversion speed
theta = 0.04  # Long-run variance
sigma = 0.3   # Volatility of volatility
rho = -0.6    # Correlation between price and volatility
r = 0.03      # Risk-free rate
q = 0.01      # Dividend yield

# Heston model characteristic function
def heston_char_func(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho):
    i = complex(0, 1)
    x = np.log(S0 / K)
    d = np.sqrt((rho * sigma * phi * i - kappa) ** 2 - sigma ** 2 * (2 * phi * i + phi ** 2))
    g = (kappa - rho * sigma * phi * i - d) / (kappa - rho * sigma * phi * i + d)
    C = (kappa * theta / sigma ** 2) * ((kappa - rho * sigma * phi * i - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = ((kappa - rho * sigma * phi * i - d) / sigma ** 2) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    char_func = np.exp(C + D * v0 + i * phi * x)
    return char_func

# Heston integrand
def heston_integrand(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho, P_num):
    i = complex(0, 1)
    char_func = heston_char_func(phi - (i if P_num == 1 else 0), S0, K, T, r, q, v0, kappa, theta, sigma, rho)
    integrand = (np.exp(-i * phi * np.log(K)) * char_func).real / (phi ** 2 + 1)
    return integrand

# Heston option price
def heston_option_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho, option_type="call"):
    P1 = 0.5 + (1 / np.pi) * quad(lambda phi: heston_integrand(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho, 1), 0, 100)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda phi: heston_integrand(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho, 2), 0, 100)[0]
    if option_type == "call":
        return np.exp(-q * T) * S0 * P1 - np.exp(-r * T) * K * P2
    elif option_type == "put":
        return np.exp(-r * T) * K * (1 - P2) - np.exp(-q * T) * S0 * (1 - P1)

# Calculate option prices
strike_price = 240  # Example strike price
time_to_maturity = 0.5  # Example time to maturity (in years)
call_price = heston_option_price(current_price, strike_price, time_to_maturity, r, q, v0, kappa, theta, sigma, rho, "call")
put_price = heston_option_price(current_price, strike_price, time_to_maturity, r, q, v0, kappa, theta, sigma, rho, "put")

# Output results
print(f"Call Option Price: ${call_price:.2f}")
print(f"Put Option Price: ${put_price:.2f}")

# Plot historical stock prices
plt.figure(figsize=(10, 6))
plt.plot(hist_data['Close'], label='AAPL Historical Prices')
plt.title('Apple Inc. Historical Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()

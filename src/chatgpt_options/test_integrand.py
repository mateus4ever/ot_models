import numpy as np
import pandas as pd
import yfinance as yf
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Fetch historical stock data
ticker = "AAPL"
stock = yf.Ticker(ticker)

# Choose an expiration date
expiration_dates = stock.options
chosen_expiration_date = expiration_dates[0]

# Fetch option chain
options_chain = stock.option_chain(chosen_expiration_date)
calls = options_chain.calls

# Extract strikes and market prices
strikes = calls['strike'].to_numpy()
market_prices = calls['lastPrice'].to_numpy()

# Example input values
S0 = stock.history(period="1d")['Close'].iloc[-1]
r = 0.03  # Risk-free rate
q = 0.01  # Dividend yield
T = (pd.to_datetime(chosen_expiration_date) - pd.Timestamp.now()).days / 365  # Time to maturity in years

# Initial Heston parameters
v0 = 0.04  # Initial variance
kappa = 2.0  # Mean-reversion speed
theta = 0.04  # Long-run variance
sigma = 0.3  # Volatility of volatility
rho = -0.7  # Correlation between stock price and variance


# Define Heston characteristic function
def heston_char_func(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho):
    i = complex(0, 1)
    x = np.log(S0 / K)
    d = np.sqrt((rho * sigma * phi * i - kappa) ** 2 - sigma ** 2 * (2 * phi * i + phi ** 2))
    g = (kappa - rho * sigma * phi * i - d) / (kappa - rho * sigma * phi * i + d)
    C = (kappa * theta / sigma ** 2) * (
                (kappa - rho * sigma * phi * i - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = ((kappa - rho * sigma * phi * i - d) / sigma ** 2) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    char_func = np.exp(C + D * v0 + i * phi * x)
    return char_func


# Define the integrand for P1 and P2
def heston_integrand(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho, P_num):
    i = complex(0, 1)
    char_func = heston_char_func(phi - (i if P_num == 1 else 0), S0, K, T, r, q, v0, kappa, theta, sigma, rho)
    integrand = (np.exp(-i * phi * np.log(K)) * char_func).real / (phi ** 2 + 1)
    return integrand


# Plot the integrand to analyze instabilities
def analyze_integrand(S0, K, T, r, q, v0, kappa, theta, sigma, rho):
    phis = np.linspace(0.1, 100, 1000)
    integrand_values = [heston_integrand(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho, 1) for phi in phis]

    plt.figure(figsize=(10, 6))
    plt.plot(phis, integrand_values)
    plt.title(f"Heston Integrand Analysis for Strike {K}")
    plt.xlabel("Phi")
    plt.ylabel("Integrand Value")
    plt.grid(True)
    plt.show()


# Heston pricing function with embedded debugging
def heston_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho, option_type="call"):
    try:
        # Analyze integrand for potential issues
        analyze_integrand(S0, K, T, r, q, v0, kappa, theta, sigma, rho)

        # Split integration if necessary
        integral_1 = quad(
            lambda phi: heston_integrand(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho, 1),
            0, 50, limit=50
        )[0]
        integral_2 = quad(
            lambda phi: heston_integrand(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho, 1),
            50, 100, limit=50
        )[0]
        P1 = 0.5 + (1 / np.pi) * (integral_1 + integral_2)

        integral_1 = quad(
            lambda phi: heston_integrand(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho, 2),
            0, 50, limit=50
        )[0]
        integral_2 = quad(
            lambda phi: heston_integrand(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho, 2),
            50, 100, limit=50
        )[0]
        P2 = 0.5 + (1 / np.pi) * (integral_1 + integral_2)
    except Exception as e:
        print(f"Integration failed: {e}")
        return None

    if option_type == "call":
        return np.exp(-q * T) * S0 * P1 - np.exp(-r * T) * K * P2
    elif option_type == "put":
        return np.exp(-r * T) * K * (1 - P2) - np.exp(-q * T) * S0 * (1 - P1)


# Example usage
strike_price = strikes[0]
price = heston_price(S0, strike_price, T, r, q, v0, kappa, theta, sigma, rho)
print(f"Price for strike {strike_price}: {price}")

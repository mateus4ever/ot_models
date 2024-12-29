import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def heston_model(S, K, T, r, kappa, theta, sigma, rho, v0, num_paths, num_steps):
    """
    Simulates asset price paths under the Heston model and calculates option prices.

    Parameters:
        S: np.ndarray
            Historical asset price array (used as initial prices for simulation)
        K: float
            Strike price of the option
        T: float
            Time to maturity (in years)
        r: float
            Risk-free interest rate
        kappa: float
            Mean reversion rate of variance
        theta: float
            Long-term variance level
        sigma: float
            Volatility of variance (volatility of volatility)
        rho: float
            Correlation between asset and variance
        v0: float
            Initial variance
        num_paths: int
            Number of simulated paths
        num_steps: int
            Number of time steps per path

    Returns:
        price: float
            Simulated European call option price
    """
    if len(S) == 0:
        raise ValueError("The historical asset price array is empty. Please check the input data.")

    S0 = S[-1]  # Use the last historical price as the starting price
    dt = T / num_steps
    simulated_S = np.zeros((num_paths, num_steps + 1))
    v = np.zeros((num_paths, num_steps + 1))
    simulated_S[:, 0] = S0
    v[:, 0] = v0

    for t in range(1, num_steps + 1):
        Z1 = np.random.normal(size=num_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=num_paths)

        v[:, t] = (
            np.maximum(
                v[:, t - 1]
                + kappa * (theta - v[:, t - 1]) * dt
                + sigma * np.sqrt(np.maximum(v[:, t - 1], 0)) * np.sqrt(dt) * Z2,
                0,
            )
        )

        simulated_S[:, t] = (
                simulated_S[:, t - 1]
                * np.exp(
            (r - 0.5 * v[:, t]) * dt
            + np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * Z1
        )
        )

    payoff = np.maximum(simulated_S - K, 0)  # Use historical prices for payoff calculation
    price = np.exp(-r * T) * np.mean(payoff)

    return price, simulated_S


# Download historical prices from Yahoo Finance
symbol = "AAPL"  # Example: Apple Inc.
data = yf.download(symbol, start="2024-01-01", end="2024-12-22")

if data.empty:
    print(f"Warning: No data was retrieved for symbol {symbol}. Using fallback prices.")
    closing_prices = np.array([150, 152, 154, 153, 155, 157, 156])  # Example fallback data
else:
    closing_prices = data['Close'].values


S0 = 248.0    # Initial stock price
K = 250.0     # Strike price
r = 0.03      # Risk-free rate
T = 1.0       # Time to maturity
kappa = 3.0   # Mean reversion rate
theta = 0.05  # Long-term average volatility
sigma = 0.3   # Volatility of volatility
rho = -0.6    # Correlation coefficient
v0 = 0.05     # Initial volatility


# Parameters
# K = 150  # Strike price
# T = 1.0  # Time to maturity (1 year)
# r = 0.03  # Risk-free rate
# kappa = 2.0  # Mean reversion speed
# theta = 0.04  # Long-term variance
# sigma = 0.6  # Volatility of variance (vol of vol)
# rho = -0.7  # Correlation between asset and variance
# v0 = 0.04  # Initial variance
num_paths = 100000  # Number of Monte Carlo paths
num_steps = 252  # Number of time steps (daily)




# Simulate Heston model
price, simulated_S = heston_model(closing_prices, K, T, r, kappa, theta, sigma, rho, v0, num_paths, num_steps)

# Output results
print(f"Simulated European Call Option Price: {price:.2f}")

# Plot simulated paths
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(simulated_S[i, :], lw=1)
plt.title(f"Simulated Asset Price Paths under Heston Model ({symbol})")
plt.xlabel("Time Step")
plt.ylabel("Asset Price")
plt.show()

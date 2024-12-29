# Reinitializing libraries due to reset
import numpy as np
import matplotlib.pyplot as plt

# Define strike prices
strike_prices = np.linspace(80, 120, 100)  # Strike prices from 80 to 120

# Generate implied volatilities
# Example 1: Flat skew (constant implied volatility)
iv_flat = np.full_like(strike_prices, 0.2)

# Example 2: Negative skew (typical for equity markets)
iv_negative_skew = 0.2 + 0.001 * (100 - strike_prices)

# Example 3: Positive skew (typical for commodities)
iv_positive_skew = 0.2 + 0.001 * (strike_prices - 100)

# Example 4: Volatility smile
iv_smile = 0.2 + 0.002 * np.abs(strike_prices - 100)

# Plot the results
plt.figure(figsize=(12, 8))

# Flat skew
plt.plot(strike_prices, iv_flat, label="Flat Skew", linestyle="--")

# Negative skew
plt.plot(strike_prices, iv_negative_skew, label="Negative Skew (Equity Market)")

# Positive skew
plt.plot(strike_prices, iv_positive_skew, label="Positive Skew (Commodity Market)")

# Volatility smile
plt.plot(strike_prices, iv_smile, label="Volatility Smile")

# Formatting
plt.title("Volatility Skew Examples")
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.legend()
plt.grid(True)
plt.show()
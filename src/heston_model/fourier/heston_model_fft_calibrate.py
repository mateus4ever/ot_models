import numpy as np
import scipy.integrate as integrate
import scipy.fftpack as fft
import matplotlib.pyplot as plt


# Heston model characteristic function (closed-form solution)
def heston_char_func(u, S0, K, T, r, kappa, theta, sigma_v, rho, v0):
    i = 1j
    x0 = np.log(S0)
    d = np.sqrt((rho * sigma_v * i * u) ** 2 + (u ** 2 + i * u) * sigma_v ** 2)
    g1 = (u ** 2 + i * u) * np.exp(i * u * x0) * np.exp(r * T)
    g2 = np.exp(-r * T) * np.exp(-i * u * np.log(K))
    phi = g1 / g2
    return phi


# Heston model option price using FFT
def heston_fft(S0, K, T, r, kappa, theta, sigma_v, rho, v0, N, M, price_type='call'):
    # Discretization of the strike prices (ensure they match the frequency resolution)
    strike_prices = np.linspace(50, 100, M)  # Define M strike prices
    K_values = np.exp(strike_prices)  # Convert strike prices to exponential scale

    # FFT variables
    u = np.arange(1, N + 1)  # Frequencies for FFT
    du = 2 * np.pi / (N * (K_values[1] - K_values[0]))  # Frequency spacing
    lambda_ = np.linspace(0, N - 1, N) * du  # Frequencies for FFT

    # Apply FFT, match strike prices with corresponding frequencies
    phi_values = np.array([heston_char_func(l, S0, K_values, T, r, kappa, theta, sigma_v, rho, v0) for l in lambda_])

    # Perform FFT: Ensure strike prices and characteristic function values align in dimensions
    fft_values = fft.fft(np.exp(-1j * lambda_ * np.log(K_values[0])) * phi_values) / (2 * np.pi)

    # Calculate option prices
    option_prices = np.real(fft_values)

    return option_prices


# Example parameters for Heston model
S0 = 100  # Spot price
K = 100  # Strike price
T = 1  # Time to maturity (1 year)
r = 0.05  # Risk-free rate
kappa = 2.0  # Rate of mean reversion
theta = 0.04  # Long-term variance (theta)
sigma_v = 0.3  # Volatility of volatility
rho = -0.5-  # Correlation between asset price and volatility
v0 = 0.2  # Initial variance

# FFT parameters
N = 128 # Number of FFT points
M = 128  # Number of strike prices (adjusted for compatibility)

# Calculate the option price
option_prices = heston_fft(S0, K, T, r, kappa, theta, sigma_v, rho, v0, N, M)

# Plot the option price as a function of strike price
plt.plot(option_prices, label='Option Prices (FFT)')
plt.xlabel('Strike Price')
plt.ylabel('Option Price')
plt.title('European Call Option Prices using Heston Model with FFT')
plt.legend()
plt.show()

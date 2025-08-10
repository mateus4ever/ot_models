import numpy as np
from scipy.integrate import quad

# Parameters
S1_0 = 100  # Current price of asset 1
S2_0 = 95   # Current price of asset 2
K = 5       # Strike price of the spread
T = 1.0     # Time to maturity (in years)
r = 0.03    # Risk-free rate
q1 = 0.01   # Dividend yield for asset 1
q2 = 0.02   # Dividend yield for asset 2
vol1 = 0.2  # Volatility of asset 1
vol2 = 0.25 # Volatility of asset 2
rho = 0.5   # Correlation between asset 1 and asset 2

# Joint characteristic function for the spread option
def joint_char_func(u, S1_0, S2_0, T, r, q1, q2, vol1, vol2, rho):
    i = complex(0, 1)
    mu1 = np.log(S1_0) + (r - q1 - 0.5 * vol1**2) * T
    mu2 = np.log(S2_0) + (r - q2 - 0.5 * vol2**2) * T
    var1 = vol1**2 * T
    var2 = vol2**2 * T
    cov = rho * vol1 * vol2 * T
    joint_mu = i * u * (mu1 - mu2)
    joint_var = -0.5 * (u**2 * (var1 + var2 - 2 * cov))
    return np.exp(joint_mu + joint_var)

# Fourier transform of the payoff
def integrand(u, S1_0, S2_0, K, T, r, q1, q2, vol1, vol2, rho):
    i = complex(0, 1)
    char_func = joint_char_func(u - i, S1_0, S2_0, T, r, q1, q2, vol1, vol2, rho)
    payoff_transform = np.exp(-i * u * np.log(K)) * char_func
    return (payoff_transform / (i * u)).real

# Price the spread option using Fourier inversion
def price_spread_option(S1_0, S2_0, K, T, r, q1, q2, vol1, vol2, rho):
    integral = quad(lambda u: integrand(u, S1_0, S2_0, K, T, r, q1, q2, vol1, vol2, rho), 0, 100)[0]
    price = np.exp(-r * T) * (0.5 + (1 / np.pi) * integral)
    return price

# Calculate the option price
spread_option_price = price_spread_option(S1_0, S2_0, K, T, r, q1, q2, vol1, vol2, rho)
print(f"Spread Option Price: {spread_option_price:.2f}")
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from datetime import datetime
import yfinance as yf

def heston_price(S0, K, T, r, v0, theta, kappa, sigma_v, rho):
    """
    Calculate the price of a European call option using the Heston model.

    Args:
    S0: Spot price
    K: Strike price
    T: Time to maturity
    r: Risk-free interest rate
    v0: Initial variance
    theta: Long-run variance
    kappa: Mean-reversion speed
    sigma_v: Volatility of volatility
    rho: Correlation between the asset price and volatility

    Returns:
    Price of the European call option under the Heston model
    """

    # Define the characteristic function of the Heston model
    def characteristic_function(phi, T, v0, theta, kappa, sigma_v, rho, r):
        i = complex(0, 1)
        u = i * phi + 0.5
        real_part = -0.5 * (v0 * u ** 2) * T
        imaginary_part = -i * np.log(S0 / K) * phi

        # Apply a numerical stability trick to avoid overflow
        try:
            exp_real = np.exp(real_part)
            exp_imaginary = np.exp(imaginary_part)
        except OverflowError:
            # In case of overflow, handle it by approximating the large values
            exp_real = np.inf if real_part > 700 else np.exp(real_part)
            exp_imaginary = 0 if imaginary_part < -700 else np.exp(imaginary_part)

        return exp_imaginary * exp_real

    # Numerically integrate the characteristic function to get the option price
    N = 100  # Number of integration points
    summand = 0
    phi_max = 100  # Limit phi to a reasonable range to avoid overflow

    for k in range(1, N + 1):
        phi = k * np.pi / T
        if phi > phi_max:
            break
        summand += characteristic_function(phi, T, v0, theta, kappa, sigma_v, rho, r)

    option_price = np.real(summand)  # Real part of the summation gives the option price
    return option_price


def compute_residuals(S0, strikes, maturities, r, theta, market_prices):
    """
    Compute the residuals between market prices and model prices.

    Args:
    S0: Spot price
    strikes: List of strike prices
    maturities: List of maturities
    r: Risk-free interest rate
    theta: Parameters of the Heston model
    market_prices: Market option prices

    Returns:
    residuals: Difference between model prices and market prices
    """
    model_prices = np.array([heston_price(S0, K, T, r, *theta) for K, T in zip(strikes, maturities)])
    residuals = market_prices - model_prices
    return residuals


def levenberg_marquardt(S0, strikes, maturities, r, market_prices, initial_guess):
    """
    Perform Levenberg-Marquardt optimization to calibrate the Heston model.

    Args:
    S0: Spot price
    strikes: List of strike prices
    maturities: List of maturities
    r: Risk-free interest rate
    market_prices: Market option prices
    initial_guess: Initial guess for the model parameters

    Returns:
    Optimized parameters for the Heston model
    """
    theta = initial_guess
    residuals = compute_residuals(S0, strikes, maturities, r, theta, market_prices)

    max_iterations = 100
    tolerance = 1e-6
    mu = 1e-3  # Damping factor

    for k in range(max_iterations):
        # Compute the Jacobian matrix and residuals
        jacobian = compute_jacobian(S0, strikes, maturities, r, theta, market_prices)

        # Compute the normal equations
        delta_theta = np.linalg.inv(jacobian.T @ jacobian + mu * np.eye(len(theta))) @ jacobian.T @ residuals

        # Update parameters
        theta_new = theta + delta_theta
        residuals_new = compute_residuals(S0, strikes, maturities, r, theta_new, market_prices)

        # If the new residuals are smaller, accept the new parameters
        if np.linalg.norm(residuals_new) < np.linalg.norm(residuals):
            theta = theta_new
            residuals = residuals_new
            mu = mu * 0.5  # Decrease damping factor
        else:
            mu = mu * 2  # Increase damping factor

        # Check if stopping criterion is met
        if np.linalg.norm(delta_theta) < tolerance:
            break

    return theta


def compute_jacobian(S0, strikes, maturities, r, theta, market_prices):
    """
    Compute the Jacobian matrix of the residuals with respect to the parameters.

    Args:
    S0: Spot price
    strikes: List of strike prices
    maturities: List of maturities
    r: Risk-free interest rate
    theta: Parameters of the Heston model [v0, theta, kappa, sigma_v, rho]
    market_prices: Market option prices

    Returns:
    jacobian: Jacobian matrix of the residuals with respect to the parameters
    """
    epsilon = 1e-5  # Small perturbation for finite differences
    jacobian = np.zeros((len(strikes), len(theta)))  # Initialize the Jacobian matrix

    # Compute residuals for the current parameters
    residuals_base = compute_residuals(S0, strikes, maturities, r, theta, market_prices)

    # Compute the Jacobian matrix using finite differences
    for i in range(len(theta)):
        # Perturb the i-th parameter
        perturbed_theta = theta.copy()
        perturbed_theta[i] += epsilon

        # Compute residuals with perturbed parameters
        residuals_perturbed = compute_residuals(S0, strikes, maturities, r, perturbed_theta, market_prices)

        # Approximate the derivative (finite difference)
        jacobian[:, i] = (residuals_perturbed - residuals_base) / epsilon

    return jacobian


# Example usage:
# Example usage:
# S0 = 100  # Spot price of the asset
# strikes = [95, 100, 105]  # Strike prices
# maturities = [0.5, 0.5, 0.5]  # Maturities (in years)
# r = 0.03  # Risk-free rate
# market_prices = [7.2, 5.5, 3.8]  # Market option prices
# initial_guess = [0.04, 0.04, 2.0, 0.3, -0.5]  # Initial guess for Heston parameters

# Fetch Apple option data from Yahoo Finance
ticker = "AAPL"
options_data = yf.Ticker(ticker).options  # Get all available expiration dates
expiration = options_data[9]  # Use the first expiration date

# Fetch the options chain for the selected expiration date
options_chain = yf.Ticker(ticker).option_chain(expiration)
calls = options_chain.calls

# Use data for the strikes and market prices
strikes = calls['strike'].values
market_prices = calls['lastPrice'].values

start_date = datetime(2024, 12, 15)
end_date = datetime(2025, 6, 20)
# Calculate the number of days between the two dates
days_difference = (end_date - start_date).days
perc_year = days_difference/365

maturities = np.full(79, perc_year)

# Risk-free rate (approximate)
r = 0.03

# Spot price (current price of AAPL)
S = yf.Ticker(ticker).history(period="1d")['Close'][-1]
initial_guess = [0.04, 0.04, 2.0, 0.3, -0.5]  # Initial guess for Heston parameters

# Perform calibration
calibrated_parameters = levenberg_marquardt(S, strikes, maturities, r, market_prices, initial_guess)
print(f"Calibrated parameters: {calibrated_parameters}")

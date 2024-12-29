# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import quad

# Step 1: Download Option Chain Data from Yahoo Finance
ticker = "AAPL"  # Example stock: Apple Inc.
stock = yf.Ticker(ticker)

# Get available option expiration dates
expiration_dates = stock.options
print(f"Available expiration dates: {expiration_dates}")

# Choose an expiration date
chosen_expiration_date = expiration_dates[0]  # Example: choose the first expiration date

# Fetch the option chain for the chosen expiration date
options_chain = stock.option_chain(chosen_expiration_date)

# Extract call option data
calls = options_chain.calls
print(f"Sample calls data:\n{calls.head()}")

# Filter for strikes and market prices
strikes = calls['strike'].to_numpy()
market_prices = calls['lastPrice'].to_numpy()

# Step 2: Define Heston Model Functions
def heston_char_func(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho):
    i = complex(0, 1)
    x = np.log(S0 / K)
    d = np.sqrt((rho * sigma * phi * i - kappa) ** 2 - sigma ** 2 * (2 * phi * i + phi ** 2))
    g = (kappa - rho * sigma * phi * i - d) / (kappa - rho * sigma * phi * i + d)
    C = (kappa * theta / sigma ** 2) * ((kappa - rho * sigma * phi * i - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = ((kappa - rho * sigma * phi * i - d) / sigma ** 2) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    char_func = np.exp(C + D * v0 + i * phi * x)
    return char_func

def heston_integrand(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho, P_num):
    i = complex(0, 1)
    char_func = heston_char_func(phi - (i if P_num == 1 else 0), S0, K, T, r, q, v0, kappa, theta, sigma, rho)
    integrand = (np.exp(-i * phi * np.log(K)) * char_func).real / (phi ** 2 + 1)
    return integrand

def heston_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho, option_type="call"):
    P1 = 0.5 + (1 / np.pi) * quad(lambda phi: heston_integrand(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho, 1), 0, 100)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda phi: heston_integrand(phi, S0, K, T, r, q, v0, kappa, theta, sigma, rho, 2), 0, 100)[0]
    if option_type == "call":
        return np.exp(-q * T) * S0 * P1 - np.exp(-r * T) * K * P2
    elif option_type == "put":
        return np.exp(-r * T) * K * (1 - P2) - np.exp(-q * T) * S0 * (1 - P1)

# Step 3: Define Objective Function for Calibration
def objective(params, market_prices, strikes, T, S0, r, q):
    kappa, theta, sigma, rho, v0 = params
    error = 0
    for i in range(len(market_prices)):
        model_price = heston_price(S0, strikes[i], T, r, q, v0, kappa, theta, sigma, rho, "call")
        error += (market_prices[i] - model_price) ** 2
    return error / len(market_prices)

# Step 4: Perform Optimization
# Example input values (replace with actual values)
S0 = stock.history(period="1d")['Close'].iloc[-1]  # Last closing price
r = 0.03  # Risk-free rate
q = 0.01  # Dividend yield
T = (pd.to_datetime(chosen_expiration_date) - pd.Timestamp.now()).days / 365  # Time to maturity in years

# Initial guesses for parameters
initial_params = [2.0, 0.04, 0.3, -0.7, 0.04]  # [kappa, theta, sigma, rho, v0]
bounds = [(0.01, 5.0), (0.001, 1.0), (0.01, 1.0), (-1.0, 1.0), (0.001, 0.1)]

# Optimize
result = minimize(objective, initial_params, args=(market_prices, strikes, T, S0, r, q), bounds=bounds, method="L-BFGS-B")
optimized_params = result.x
print("Optimized Parameters:", optimized_params)

# Step 5: Validate and Compare Model Prices
model_prices = [
    heston_price(S0, strikes[i], T, r, q, optimized_params[4], *optimized_params[:4], "call")
    for i in range(len(strikes))
]
print("Market Prices:", market_prices)
print("Model Prices:", model_prices)

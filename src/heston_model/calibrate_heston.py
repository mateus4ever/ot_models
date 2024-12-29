
import yfinance as yf
import numpy as np
from scipy import optimize
from scipy.integrate import quad
import math

i = 1j

def fetch_apple_data(start_date='2020-01-01', end_date='2023-01-01'):
    data = yf.download('AAPL', start=start_date, end=end_date)
    return data

def heston_char_func(u, S0, K, T, r, kappa, theta, sigma_v, rho, v0):
    x0 = np.log(S0)
    d = np.sqrt((rho * sigma_v * i * u) ** 2 + (u ** 2 + i * u) * sigma_v ** 2)
    g1 = np.exp(i * u * x0)
    g2 = np.exp(-i * u * np.log(S0))
    phi = g1 / g2
    return phi

def heston_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0, market_prices):
    def integrand(u, K_val):
        char_func_val = heston_char_func(u, S0, K_val, T, r, kappa, theta, sigma_v, rho, v0)
        return np.real(np.exp(-i * u * np.log(K_val)) * char_func_val) / (u ** 2 + 1)

    prices = []
    for K_val in K:
        integral_result, _ = quad(integrand, 0, np.inf, args=(K_val,))
        prices.append(integral_result)

    return np.array(prices)

def objective_function(params, *args):
    S0, K, T, r, market_prices = args
    kappa, theta, sigma_v, rho, v0 = params
    model_price = heston_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0, market_prices)
    return np.sum((model_price - market_prices) ** 2)

def calibrate_heston_model(stock_ticker='AAPL', start_date='2020-01-01', end_date='2023-01-01'):
    stock_data = fetch_apple_data(start_date=start_date, end_date=end_date)
    S0 = stock_data['Close'].iloc[-1]

    market_prices =  stock_data['Close'].values
    K = np.array([100, 110, 120])
    T = 1
    r = 0.05
    v0 = 0.2

    initial_guess = [2.0, 0.04, 0.3, -0.5, v0]

    result = optimize.minimize(objective_function, initial_guess, args=(S0, K, T, r, market_prices),
                               bounds=((0.01, 5), (0, 1), (0, 2), (-1, 1), (0.01, 2)))

    kappa, theta, sigma_v, rho, v0 = result.x

    print(f"Calibrated parameters for {stock_ticker}:")
    print(f"Kappa: {kappa}")
    print(f"Theta: {theta}")
    print(f"Sigma_v: {sigma_v}")
    print(f"Rho: {rho}")
    print(f"V0: {v0}")

calibrate_heston_model(stock_ticker='AAPL', start_date='2020-01-01', end_date='2023-01-01')
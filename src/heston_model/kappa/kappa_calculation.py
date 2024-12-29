from datetime import datetime

import yfinance as yf
import QuantLib as ql
import numpy as np
from scipy.optimize import minimize


# Define a function to calculate option prices using the Heston model
def heston_price(S, K, T, r, v0, theta, kappa, sigma_v, rho):
    # Ensure parameters are valid (e.g., no negative volatilities)
    if v0 <= 0 or theta <= 0 or kappa <= 0 or sigma_v <= 0:
        return np.nan  # Return NaN if any parameter is invalid

    # Set up the QuantLib environment
    today = ql.Date(15, 12, 2024)  # today's date
    ql.Settings.instance().evaluationDate = today

    # Risk-free rate (using flat forward for simplicity)
    riskFreeCurve = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))

    # Dividend yield (assuming zero dividend yield)
    dividendCurve = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))

    # Spot price of the underlying asset
    spotHandle = ql.QuoteHandle(ql.SimpleQuote(S))

    # Set up the Heston process
    heston_process = ql.HestonProcess(riskFreeCurve, dividendCurve, spotHandle, v0, theta, kappa, sigma_v, rho)

    # Create the Heston model
    heston_model = ql.HestonModel(heston_process)

    # Pricing engine
    engine = ql.AnalyticHestonEngine(heston_model)

    # European option
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(today + int(T * 365))  # Calculate maturity in days
    option = ql.EuropeanOption(payoff, exercise)
    option.setPricingEngine(engine)

    # Return the option price
    return option.NPV()


# Function to calculate the total error between model prices and market prices
def objective(params, S, market_prices, strikes, maturities, r):
    v0, theta, kappa, sigma_v, rho = params
    model_prices = [heston_price(S, K, T, r, v0, theta, kappa, sigma_v, rho) for K, T in zip(strikes, maturities)]

    # Check if any model price is NaN (invalid parameter set)
    if any(np.isnan(price) for price in model_prices):
        return np.inf  # Return a large number if parameters are invalid

    # Calculate the sum of squared errors (optimization criteria)
    return np.sum((np.array(model_prices) - np.array(market_prices)) ** 2)


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

# Initial guess for the parameters [v0, theta, kappa, sigma_v, rho]
initial_params = [0.2261, 0.05, 2.0, 0.3, -0.5]

# Use scipy.optimize to minimize the objective function and fit the model
result = minimize(objective, initial_params, args=(S, market_prices, strikes,maturities, r), method='L-BFGS-B',
                  bounds=[
                      (0.0001, 2.0),  # v0
                      (0.0001, 2.0),  # theta
                      (0.01, 10.0),  # kappa (mean-reversion speed)
                      (0.01, 2.0),  # sigma_v (volatility of volatility)
                      (-1.0, 1.0)  # rho (correlation between asset price and volatility)
                  ])

# Print the fitted parameters
v0, theta, kappa, sigma_v, rho = result.x
print(f"Fitted parameters:\nv0: {v0}, theta: {theta}, kappa: {kappa}, sigma_v: {sigma_v}, rho: {rho}")

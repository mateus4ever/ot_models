import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

# Step 1: Fetch Apple stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2015-01-01', end='2024-12-11')

# Step 2: Calculate daily returns
data['Return'] = data['Adj Close'].pct_change() * 100  # Convert to percentage

# Step 3: Fit an EGARCH model
returns = data['Return'].dropna()
model = arch_model(returns, vol='EGARCH', p=1, q=1, mean='constant', dist='normal')
results = model.fit(disp='off')

# Step 4: Initialize forecasting process
forecast_horizon = 20
last_resid = results.resid[-1]  # Last residual
last_variance = results.conditional_volatility[-1] ** 2  # Last conditional variance

forecasted_volatility = []

# Forecast iteratively
for i in range(forecast_horizon):
    # Compute next-day variance using the EGARCH equation
    variance = results.params['omega'] + \
               results.params['alpha[1]'] * (abs(last_resid / np.sqrt(last_variance)) - np.sqrt(2 / np.pi)) + \
               results.params['beta[1]'] * np.log(last_variance)

    next_variance = np.exp(variance)
    next_volatility = np.sqrt(next_variance)

    # Store results
    forecasted_volatility.append(next_volatility)

    # Update for the next step
    last_variance = next_variance
    last_resid = 0  # For out-of-sample forecasts, assume mean return is 0

# Display results
for i, vol in enumerate(forecasted_volatility, start=1):
    print(f"Day {i} Volatility (Daily, Estimated): {vol:.2f}%")

# Optional: Calculate cumulative and annualized volatility
cumulative_variance = sum(v ** 2 for v in forecasted_volatility)
total_cumulative_volatility = np.sqrt(cumulative_variance)
annualized_volatility_10_days = total_cumulative_volatility * np.sqrt(252 / 10)

print(f"Annualized Volatility (Next 10 Days, Estimated): {annualized_volatility_10_days:.2f}%")

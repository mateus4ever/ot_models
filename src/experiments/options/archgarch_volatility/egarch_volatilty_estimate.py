import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

# Step 1: Fetch Apple stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2015-01-01', end='2025-01-02')

# Step 2: Calculate daily returns
data['Return'] = data['Adj Close'].pct_change() * 100  # Convert to percentage

# Step 3: Fit an EGARCH model
returns = data['Return'].dropna()
model = arch_model(returns, vol='EGARCH', p=1, q=1, mean='constant', dist='normal')
results = model.fit(disp='off')

# Step 4: Forecast next-day volatility
forecast = results.forecast(horizon=1)
next_day_variance = forecast.variance.iloc[-1, 0]
next_day_volatility = np.sqrt(next_day_variance)  # Convert variance to volatility

# Calculate annualized volatility
annualized_volatility = next_day_volatility * np.sqrt(252)
# Display results
print(f"Next Day Volatility (Daily, Estimated): {next_day_volatility:.2f}%")
print(f"Next Day Volatility (Annualized, Estimated): {annualized_volatility:.2f}%")

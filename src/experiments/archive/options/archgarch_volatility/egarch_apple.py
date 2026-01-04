import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model

# Step 1: Fetch Apple stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2015-01-01', end='2023-12-31')

# Step 2: Calculate daily returns
data['Return'] = data['Adj Close'].pct_change() * 100  # Convert to percentage

# Step 3: Calculate realized volatility (rolling standard deviation of returns)
data['Realized Volatility'] = data['Return'].rolling(window=21).std()

# Step 4: Fit an EGARCH model to the returns
returns = data['Return'].dropna()
model = arch_model(returns, vol='EGARCH', p=1, q=1, mean='constant', dist='normal')
results = model.fit(disp='off')

# Step 5: Extract EGARCH conditional volatility
data['EGARCH Volatility'] = np.nan  # Initialize the column with NaNs
data.loc[returns.index, 'EGARCH Volatility'] = results.conditional_volatility

# Step 6: Align data and drop missing values
comparison_df = data[['Realized Volatility', 'EGARCH Volatility']].dropna()

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Realized Volatility'], label='Realized Volatility', alpha=0.7)
plt.plot(comparison_df['EGARCH Volatility'], label='EGARCH Volatility', alpha=0.7)
plt.title(f"Comparison of EGARCH and Realized Volatility for {ticker}")
plt.legend()
plt.show()

# Calculate correlation and error metrics
correlation = comparison_df.corr()
mae = (comparison_df['EGARCH Volatility'] - comparison_df['Realized Volatility']).abs().mean()
rmse = np.sqrt(((comparison_df['EGARCH Volatility'] - comparison_df['Realized Volatility'])**2).mean())

# Display metrics
result = {
    "Correlation": correlation,
    "Mean Absolute Error (MAE)": mae,
    "Root Mean Squared Error (RMSE)": rmse
}

# Print the results
print(result)
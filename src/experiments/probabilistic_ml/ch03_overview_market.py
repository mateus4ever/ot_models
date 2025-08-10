# Import necessary Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

# Set plotting style
plt.style.use('seaborn')

# Define start and end dates for S&P 500 ('SPY') historical data
start = datetime(1993, 2, 1)
end = datetime(2022, 10, 15)

# Fetch historical price data for SPY
equity = yf.Ticker('SPY').history(start=start, end=end)

# Compute daily percentage returns and remove NaN values
equity['Returns'] = equity['Close'].pct_change(1) * 100
equity = equity.dropna()

# Plot the histogram of SPY's daily returns
plt.figure(figsize=(10, 5))
plt.hist(equity['Returns'], bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.title('Distribution of S&P 500 Daily Percentage Returns Over the Past 30 Years')
plt.xlabel('Daily Percentage Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Display descriptive statistics of S&P 500 daily returns
desc_stats = equity['Returns'].describe().round(2)
print(f"Descriptive statistics of S&P 500 percentage returns:\n{desc_stats}")

# Compute and display skewness and kurtosis of returns
skewness = equity['Returns'].skew()
kurtosis = equity['Returns'].kurtosis()
print(f"\nThe skewness of S&P 500 returns is: {skewness:.2f} and the kurtosis is: {kurtosis:.2f}")

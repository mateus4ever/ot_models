
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

# Estimate the mean and standard deviation from SPY's 30-year historical data
mean = equity['Returns'].mean()
vol = equity['Returns'].std()
sample = equity['Returns'].count()

# Use NumPy's random number generator to sample from a normal distribution
# with the above estimates of its mean and standard deviation
# Create a new column called 'Simulated' and generate the same number of
# random samples from NumPy's normal distribution as the actual data sample
# you've imported above for SPY
equity['Simulated'] = np.random.normal(mean, vol, sample)

# Visualize and summarize SPY's simulated daily price returns.
plt.figure(figsize=(10, 5))
plt.hist(equity['Simulated'], bins=50, color='green', alpha=0.7, edgecolor='black')
plt.title('Distribution of S&P 500 Simulated Daily Percentage Returns')
plt.xlabel('Daily Percentage Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Display descriptive statistics of SPY's simulated daily returns
desc_stats_simulated = equity['Simulated'].describe().round(2)
print(f"Descriptive statistics of S&P 500 stock's simulated percentage returns:\n{desc_stats_simulated}")

# Compute and display skewness and kurtosis of the simulated daily price returns
skewness_simulated = equity['Simulated'].skew().round(2)
kurtosis_simulated = equity['Simulated'].kurtosis().round(2)
print(f"\nThe skewness of S&P 500 simulated returns is: {skewness_simulated} and the kurtosis is: {kurtosis_simulated}.")

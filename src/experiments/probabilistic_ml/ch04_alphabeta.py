# Import relevant Python packages
import statsmodels.api as sm
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from pandas_datareader import data as pdr

# Set plot style
plt.style.use('seaborn')

# Define the date range for data collection
start = datetime(2017, 8, 3)
end = datetime(2022, 8, 6)

# Import financial data
# S&P 500 index as a proxy for the market
market = yf.Ticker('SPY').history(start=start, end=end)
# Fetch S&P 500 (use ^GSPC instead of SPY)
# market = pdr.get_data_yahoo("^GSPC", start=start, end=end)

# Apple Inc. (AAPL) stock data
stock = yf.Ticker('AAPL').history(start=start, end=end)

# 10-year US Treasury Note as a proxy for the risk-free rate
riskfree_rate = yf.Ticker('^TNX').history(start=start, end=end)

# Create a DataFrame to hold daily returns of securities
daily_returns = pd.DataFrame()
daily_returns['market'] = market['Close'].pct_change(1) * 100
daily_returns['stock'] = stock['Close'].pct_change(1) * 100

# Compounded daily risk-free rate based on 360 days (used in bond markets)
daily_returns['riskfree'] = (1 + riskfree_rate['Close']) ** (1/360) - 1

# Plot and summarize the distribution of daily returns for SPY (market)
plt.hist(daily_returns['market'])
plt.title('Distribution of Market (SPY) Daily Returns')
plt.xlabel('Daily Percentage Returns')
plt.ylabel('Frequency')
plt.show()

# Analyze descriptive statistics for market returns
print("Descriptive Statistics of the Market's daily percentage returns:\n{}"
      .format(daily_returns['market'].describe()))

# Plot and summarize the distribution of daily returns for AAPL (stock)
plt.hist(daily_returns['stock'])
plt.title('Distribution of Apple Inc. (AAPL) Daily Returns')
plt.xlabel('Daily Percentage Returns')
plt.ylabel('Frequency')
plt.show()

# Analyze descriptive statistics for Apple stock returns
print("Descriptive Statistics of Apple's daily percentage returns:\n{}"
      .format(daily_returns['stock'].describe()))

# Plot and summarize the distribution of the risk-free rate
plt.hist(daily_returns['riskfree'])
plt.title('Distribution of the Risk-Free Rate (TNX) Daily Returns')
plt.xlabel('Daily Percentage Returns')
plt.ylabel('Frequency')
plt.show()

# Analyze descriptive statistics for risk-free rate
print("Descriptive Statistics of the 10-year Treasury Note daily percentage returns:\n{}"
      .format(daily_returns['riskfree'].describe()))

# Examine missing rows in the dataset
missing_rows = market.index.difference(riskfree_rate.index)

# Fill missing rows with the previous day's risk-free rate (generally stable)
daily_returns = daily_returns.ffill()

# Drop NaN values from the first row due to percentage calculations
daily_returns = daily_returns.dropna()

# Check for null values
print("Null values in dataset:\n", daily_returns.isnull().sum())

# Display the first five rows of the dataset
print("First five rows of the dataset:\n", daily_returns.head())

# Market Model for AAPL based on daily excess returns
# Calculate daily excess returns for AAPL
y = daily_returns['stock'] - daily_returns['riskfree']

# Calculate daily excess returns for the market
x = daily_returns['market'] - daily_returns['riskfree']

# Scatter plot of the excess returns
plt.scatter(x, y)
plt.title('AAPL vs. Market (SPY) Excess Returns')
plt.xlabel('SPY Daily Excess Returns')
plt.ylabel('AAPL Daily Excess Returns')
plt.show()

# Add a constant term for the intercept in the regression model
x = sm.add_constant(x)

# Fit an Ordinary Least Squares (OLS) regression model
market_model = sm.OLS(y, x).fit()

# Plot the line of best fit
plt.scatter(x.iloc[:, 1], y, label="Data")
plt.plot(x.iloc[:, 1], x @ market_model.params, color='red', label="Best Fit Line")
plt.title('Market Model of AAPL')
plt.xlabel('SPY Daily Excess Returns')
plt.ylabel('AAPL Daily Excess Returns')
plt.legend()
plt.show()

# Display the alpha and beta of AAPL's market model
alpha = round(market_model.params['const'], 4)
beta = round(market_model.params[1], 4)
print(f"According to AAPL's Market Model, the security had a realized Alpha of {alpha}% and Beta of {beta}")

# Summarize and analyze the statistics of the linear regression
print("The Market Model of AAPL is summarized below:\n")
print(market_model.summary())

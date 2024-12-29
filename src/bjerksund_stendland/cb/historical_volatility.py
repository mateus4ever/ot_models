import yfinance as yf
import numpy as np

def calculate_historical_volatility(prices, trading_days=252):
    """
    Calculate the historical volatility of a stock.

    Args:
        prices (list or numpy array): Array of adjusted closing prices.
        trading_days (int): Number of trading days in a year (default is 252).

    Returns:
        float: Annualized historical volatility.
    """
    # Calculate daily log returns
    returns = np.log(prices[1:] / prices[:-1])
    # Calculate the standard deviation of returns
    daily_vol = np.std(returns)
    # Annualize the volatility
    annual_vol = daily_vol * np.sqrt(trading_days)
    return annual_vol

# Fetch historical data for Apple (AAPL)
data = yf.download("AAPL", start="2024-05-27", end="2024-12-27")
prices = data['Adj Close'].values

# Calculate historical volatility
historical_vol = calculate_historical_volatility(prices)

# Print the result
print(f"Historical Volatility (Annualized) for AAPL in 2023: {historical_vol:.4f}")

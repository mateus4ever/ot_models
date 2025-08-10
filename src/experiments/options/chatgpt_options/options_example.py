# Re-import necessary libraries due to the reset
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the stock ticker and option parameters
ticker = "AAPL"  # Apple Inc. as an example
option_type = "call"  # Could be 'call' or 'put'

# Fetch the options data
stock = yf.Ticker(ticker)

# Get the available options expiration dates
# expiration_dates = stock.options
# expiration_dates = stock.options[:3]
expiration_dates = ['2025-01-17']

plt.figure(figsize=(10, 6))

for expiration_date in expiration_dates:
    # Choose an expiration date for demonstration (taking the first one)

    # Get the options chain for the selected expiration date
    options_chain = stock.option_chain(expiration_date)

    # Select the desired option type (calls or puts)
    options_data = options_chain.calls if option_type == "call" else options_chain.puts
    # Plotting the option prices against strike prices
    plt.plot(options_data['strike'], options_data['lastPrice'], marker='o', label=f'{option_type.capitalize() + expiration_date} Options')
    plt.title(f'{ticker} {option_type.capitalize()} Options Prices ({expiration_date})')

plt.xlabel('Strike Price')
plt.ylabel('Last Price')
plt.legend()
plt.grid()
plt.show()

print ("test")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_capital = 10_000
short_window = 10
long_window = 50
momentum_threshold = 0.02  # buy signal if return > 2%

# Load Tesla EOD data (make sure 'tesla_eod.csv' is in the same folder)
df = pd.read_csv("tesla_eod.csv", parse_dates=["Datetime"])
df.set_index("Datetime", inplace=True)
df = df.sort_index()

# Compute indicators
df["short_ma"] = df["Close"].rolling(window=short_window).mean()
df["long_ma"] = df["Close"].rolling(window=long_window).mean()
df["momentum"] = df["Close"].pct_change(periods=5)

# Generate signals
df["signal"] = 0
df.loc[(df["short_ma"] > df["long_ma"]) & (df["momentum"] > momentum_threshold), "signal"] = 1
df.loc[(df["short_ma"] < df["long_ma"]) | (df["momentum"] < -momentum_threshold), "signal"] = -1
df["signal"] = df["signal"].fillna(0)

# Backtest
position = 0
cash = initial_capital
portfolio_value = []
positions = []
buy_prices = []
sell_prices = []

for i in range(len(df)):
    price = df["Close"].iloc[i]
    signal = df["signal"].iloc[i]

    # Buy signal
    if signal == 1 and position == 0:
        position = cash / price
        cash = 0
        buy_prices.append((df.index[i], price))
    # Sell signal
    elif signal == -1 and position > 0:
        cash = position * price
        position = 0
        sell_prices.append((df.index[i], price))

    current_value = cash + position * price
    portfolio_value.append(current_value)
    positions.append(position)

# Add results to DataFrame
df["PortfolioValue"] = portfolio_value
df["Position"] = positions

# Plot portfolio value
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["PortfolioValue"], label="Portfolio Value", color="blue")
plt.title("Momentum Strategy Portfolio Value")
plt.xlabel("Date")
plt.ylabel("USD")
plt.legend()

# Annotate buy/sell points
for date, price in buy_prices:
    plt.scatter(date, price, marker="^", color="green", label="Buy")
for date, price in sell_prices:
    plt.scatter(date, price, marker="v", color="red", label="Sell")
plt.grid()
plt.tight_layout()
plt.show()

# Summary
print(f"Final Portfolio Value: ${df['PortfolioValue'].iloc[-1]:,.2f}")
print(f"Number of Buys: {len(buy_prices)}")
print(f"Number of Sells: {len(sell_prices)}")

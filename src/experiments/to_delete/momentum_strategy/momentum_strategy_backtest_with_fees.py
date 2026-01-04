import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Parameters
initial_capital = 10_000
momentum_window = 10
fee_rate = 0.001  # 0.1% per trade
position_fraction = 0.2  # invest 20% of capital

# Load Tesla data
df = pd.read_csv("tesla_eod.csv", parse_dates=["Datetime"])
df.set_index("Datetime", inplace=True)
df.sort_index(inplace=True)
df["Price"] = df["Close"]

# Compute momentum
df["Momentum"] = df["Price"].pct_change(momentum_window)

# Initialize portfolio variables
capital = initial_capital
position = 0
portfolio_values = []
buy_prices = []
sell_prices = []
transaction_count = 0
success_count = 0
fail_count = 0
entry_price = 0

# Backtesting loop
for i in range(len(df)):
    price = df["Price"].iloc[i]
    date = df.index[i]
    signal = df["Momentum"].iloc[i]

    # Buy condition
    if signal > 0.05 and position == 0:
        investment = capital * position_fraction
        shares = investment // (price * (1 + fee_rate))
        if shares > 0:
            cost = shares * price * (1 + fee_rate)
            capital -= cost
            position = shares
            entry_price = price
            transaction_count += 1
            buy_prices.append((date, price))

    # Sell condition
    elif signal < -0.05 and position > 0:
        proceeds = position * price * (1 - fee_rate)
        capital += proceeds
        pnl = (price - entry_price) * position
        if pnl > 0:
            success_count += 1
        else:
            fail_count += 1
        sell_prices.append((date, price))
        position = 0
        transaction_count += 1

    # Track portfolio value
    position_value = position * price
    total_value = capital + position_value
    portfolio_values.append(total_value)

# Store in DataFrame
df["PortfolioValue"] = portfolio_values

# Summary
final_value = capital + position * df["Price"].iloc[-1]
print(f"Initial capital:       ${initial_capital:,.2f}")
print(f"Final portfolio value: ${final_value:,.2f}")
print(f"Total return:          {((final_value / initial_capital - 1) * 100):.2f}%")
print(f"Total transactions:    {transaction_count}")
print(f"  Successful:          {success_count}")
print(f"  Unsuccessful:        {fail_count}")

# Calculate Sharpe Ratio
df["DailyReturn"] = df["PortfolioValue"].pct_change()
mean_return = df["DailyReturn"].mean()
volatility = df["DailyReturn"].std()
sharpe_ratio = mean_return / volatility * np.sqrt(252)  # annualized for daily data

print(f"Sharpe Ratio:          {sharpe_ratio:.2f}")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["PortfolioValue"], label="Portfolio Value", color="blue")

# Plot buy/sell without duplicate legend
for date, price in buy_prices:
    plt.scatter(date, price, marker="^", color="green", zorder=5)
for date, price in sell_prices:
    plt.scatter(date, price, marker="v", color="red", zorder=5)

# Custom legend
portfolio_line = mlines.Line2D([], [], color='blue', label='Portfolio Value')
buy_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=8, label='Buy')
sell_marker = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=8, label='Sell')
plt.legend(handles=[portfolio_line, buy_marker, sell_marker], loc="upper left")

plt.title("Momentum Strategy on Tesla (20% Capital, Fees Included)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

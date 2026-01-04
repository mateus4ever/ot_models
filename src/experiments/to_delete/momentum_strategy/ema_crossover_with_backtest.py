import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Parameters
initial_capital = 10_000
fee_rate = 0.001
position_fraction = 0.2
ema_short = 20
ema_long = 50

# Load data (replace with your target stock file)
df = pd.read_csv("spy_eod.csv", parse_dates=["Datetime"])
df.set_index("Datetime", inplace=True)
df.sort_index(inplace=True)
df["Price"] = df["Close"]

# Compute EMAs
df["EMA_Short"] = df["Price"].ewm(span=ema_short, adjust=False).mean()
df["EMA_Long"] = df["Price"].ewm(span=ema_long, adjust=False).mean()

# Strategy variables
capital = initial_capital
position = 0
entry_price = 0
portfolio_values = []
buy_prices = []
sell_prices = []
transaction_count = 0
success_count = 0
fail_count = 0

# Backtest loop
for i in range(1, len(df)):
    price = df["Price"].iloc[i]
    ema_short_prev = df["EMA_Short"].iloc[i - 1]
    ema_long_prev = df["EMA_Long"].iloc[i - 1]
    ema_short_now = df["EMA_Short"].iloc[i]
    ema_long_now = df["EMA_Long"].iloc[i]

    # Buy condition: EMA short crosses above EMA long
    if position == 0 and ema_short_prev < ema_long_prev and ema_short_now > ema_long_now:
        investment = capital * position_fraction
        shares = investment // (price * (1 + fee_rate))
        if shares > 0:
            cost = shares * price * (1 + fee_rate)
            capital -= cost
            position = shares
            entry_price = price
            transaction_count += 1
            buy_prices.append((df.index[i], price))

    # Sell condition: EMA short crosses below EMA long
    elif position > 0 and ema_short_prev > ema_long_prev and ema_short_now < ema_long_now:
        proceeds = position * price * (1 - fee_rate)
        capital += proceeds
        pnl = (price - entry_price) * position
        if pnl > 0:
            success_count += 1
        else:
            fail_count += 1
        sell_prices.append((df.index[i], price))
        position = 0
        transaction_count += 1

    position_value = position * price
    total_value = capital + position_value
    portfolio_values.append(total_value)

# Store results
df = df.iloc[:len(portfolio_values)].copy()
df["PortfolioValue"] = portfolio_values
final_value = capital + position * df["Price"].iloc[-1]

# Summary
print(f"Initial capital:       ${initial_capital:,.2f}")
print(f"Final portfolio value: ${final_value:,.2f}")
print(f"Total return:          {((final_value / initial_capital - 1) * 100):.2f}%")
print(f"Total transactions:    {transaction_count}")
print(f"  Successful:          {success_count}")
print(f"  Unsuccessful:        {fail_count}")

# Sharpe ratio
returns = df["PortfolioValue"].pct_change().dropna()
sharpe = returns.mean() / returns.std() * np.sqrt(252)
print(f"Sharpe Ratio:          {sharpe:.2f}")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["PortfolioValue"], label="Strategy Portfolio", color="blue")

# Buy/sell markers
for date, price in buy_prices:
    plt.scatter(date, price, marker="^", color="green", zorder=5)
for date, price in sell_prices:
    plt.scatter(date, price, marker="v", color="red", zorder=5)

# Benchmark comparison (SPY buy-and-hold)
df["BuyHold"] = initial_capital * (df["Price"] / df["Price"].iloc[0])
plt.plot(df.index, df["BuyHold"], label="Buy & Hold Benchmark", linestyle="--", color="gray")

# Custom legend
portfolio_line = mlines.Line2D([], [], color='blue', label='Strategy Portfolio')
buy_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=8, label='Buy')
sell_marker = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=8, label='Sell')
benchmark_line = mlines.Line2D([], [], color='gray', linestyle='--', label='Buy & Hold')
plt.legend(handles=[portfolio_line, buy_marker, sell_marker, benchmark_line], loc="upper left")

plt.title("EMA Crossover Strategy vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

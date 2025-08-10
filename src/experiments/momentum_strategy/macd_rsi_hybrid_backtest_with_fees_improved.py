import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Parameters
initial_capital = 10_000
fee_rate = 0.001
position_fraction = 0.2
stop_loss_pct = 0.05
trailing_stop_pct = 0.03

# Load Tesla EOD data
#df = pd.read_csv("tesla_eod.csv", parse_dates=["Datetime"])
df = pd.read_csv("aapl_eod.csv", parse_dates=["Datetime"])
df.set_index("Datetime", inplace=True)
df.sort_index(inplace=True)
df["Price"] = df["Close"]

# Indicators: MACD and RSI
exp1 = df["Price"].ewm(span=12, adjust=False).mean()
exp2 = df["Price"].ewm(span=26, adjust=False).mean()
df["MACD"] = exp1 - exp2
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

# Corrected RSI calculation
window = 14
delta = df["Price"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
roll_up = gain.rolling(window=window).mean()
roll_down = loss.rolling(window=window).mean()
RS = roll_up / roll_down
df["RSI"] = 100.0 - (100.0 / (1.0 + RS))

# Additional trend filter: 50-day SMA
df["SMA50"] = df["Price"].rolling(window=50).mean()

# Strategy variables
capital = initial_capital
position = 0
entry_price = 0
peak_price = 0
portfolio_values = []
buy_prices = []
sell_prices = []
transaction_count = 0
success_count = 0
fail_count = 0

# Backtest loop
for i in range(1, len(df)):
    price = df["Price"].iloc[i]
    macd = df["MACD"].iloc[i]
    signal = df["Signal"].iloc[i]
    rsi = df["RSI"].iloc[i]
    sma50 = df["SMA50"].iloc[i]

    macd_prev = df["MACD"].iloc[i - 1]
    signal_prev = df["Signal"].iloc[i - 1]
    rsi_prev = df["RSI"].iloc[i - 1]

    macd_crossover = macd_prev < signal_prev and macd > signal
    rsi_rising = rsi > rsi_prev
    macd_slope = macd - macd_prev
    rsi_slope = rsi - rsi_prev

    # Buy condition: MACD crossover + rising RSI + above SMA50 + upward slopes
    if (position == 0 and macd_crossover and rsi_rising and price > sma50
        and macd_slope > 0 and rsi_slope > 0):
        investment = capital * position_fraction
        shares = investment // (price * (1 + fee_rate))
        if shares > 0:
            cost = shares * price * (1 + fee_rate)
            capital -= cost
            position = shares
            entry_price = price
            peak_price = price
            transaction_count += 1
            buy_prices.append((df.index[i], price))

    # Update peak price for trailing stop
    if position > 0:
        peak_price = max(peak_price, price)

    # Exit condition
    if position > 0:
        pnl = (price - entry_price) * position
        stop_loss_triggered = price < entry_price * (1 - stop_loss_pct)
        trailing_stop_triggered = price < peak_price * (1 - trailing_stop_pct)
        macd_bearish = macd_prev > signal_prev and macd < signal
        rsi_falling = rsi < rsi_prev

        if stop_loss_triggered or trailing_stop_triggered or (macd_bearish and rsi_falling):
            proceeds = position * price * (1 - fee_rate)
            capital += proceeds
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
plt.plot(df.index, df["PortfolioValue"], label="Portfolio Value", color="blue")

for date, price in buy_prices:
    plt.scatter(date, price, marker="^", color="green", zorder=5)
for date, price in sell_prices:
    plt.scatter(date, price, marker="v", color="red", zorder=5)

portfolio_line = mlines.Line2D([], [], color='blue', label='Portfolio Value')
buy_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=8, label='Buy')
sell_marker = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=8, label='Sell')
plt.legend(handles=[portfolio_line, buy_marker, sell_marker], loc="upper left")

plt.title("Improved Hybrid Strategy: MACD+RSI+SMA+Stop Systems+Momentum")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Strategy Overview:
# This is a hybrid trend-following strategy using exponential moving average (EMA) crossover with an adaptive volatility filter and a trailing stop mechanism.
# EMA is preferred over SMA for its responsiveness to recent price changes.
# The leveraged ETF 'UPRO' is used to amplify returns (and risks).
# The volatility window of 14 is a commonly used short-term period (similar to RSI default).
# Entry signals occur when the short EMA crosses above the long EMA, provided volatility is above its 90-day median.
#
# Why use an adaptive volatility filter?
# Volatility is a proxy for market uncertainty and participation. High volatility periods often coincide with stronger directional moves.
# The filter blocks trades when volatility is below its median over the past 90 days — this helps avoid entering trades in quiet, range-bound markets.
# The volatility filter complements the trend signal (EMA crossover) by acting as a regime selector — ensuring the trend is active enough.
#
# Note:
# This script does not use the actual VIX index. Instead, it implements a volatility proxy based on the rolling standard deviation
# of price changes. This is a common practice in quantitative strategies where intraday or custom volatility filters are required.
# You can think of it as a localized, ETF-specific equivalent to the VIX.

# Parameters
initial_capital = 10_000
fee_rate = 0.001  # 0.1% transaction fee
position_fraction = 0.2  # Invest 20% of capital per trade
ema_short = 20  # Short-term EMA window
ema_long = 50   # Long-term EMA window
volatility_window = 14  # Window for recent volatility estimation
trailing_stop_pct = 0.05  # 5% trailing stop

# Load data
# UPRO: Leveraged ETF that tracks 3x the daily performance of the S&P 500 index
# Chosen to enhance returns compared to standard SPY ETF

df = pd.read_csv("spy_eod.csv", parse_dates=["Datetime"])
df.set_index("Datetime", inplace=True)
df.sort_index(inplace=True)
df["Price"] = df["Close"]

# Compute indicators
# EMA crossover is used to detect trend direction
# Volatility filter ensures trades only occur during sufficiently active periods
df["EMA_Short"] = df["Price"].ewm(span=ema_short, adjust=False).mean()
df["EMA_Long"] = df["Price"].ewm(span=ema_long, adjust=False).mean()
df["Volatility"] = df["Price"].pct_change().rolling(volatility_window).std()
df["MedianVolatility"] = df["Volatility"].rolling(90).median()

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
volatility_blocks = 0
signal_opportunities = 0
volatility_block_log = []  # Stores timestamps where trades were blocked due to low volatility

# Backtest loop
for i in range(1, len(df)):
    price = df["Price"].iloc[i]
    ema_short_prev = df["EMA_Short"].iloc[i - 1]
    ema_long_prev = df["EMA_Long"].iloc[i - 1]
    ema_short_now = df["EMA_Short"].iloc[i]
    ema_long_now = df["EMA_Long"].iloc[i]
    volatility = df["Volatility"].iloc[i]
    median_vol = df["MedianVolatility"].iloc[i]

    # Trailing stop: sell if price drops 5% from peak
    if position > 0:
        peak_price = max(peak_price, price)
        if price < peak_price * (1 - trailing_stop_pct):
            proceeds = position * price * (1 - fee_rate)
            capital += proceeds
            pnl = (price - entry_price) * position
            success_count += int(pnl > 0)
            fail_count += int(pnl <= 0)
            sell_prices.append((df.index[i], price))
            position = 0
            transaction_count += 1

    # Entry condition: short EMA crosses above long EMA AND volatility is sufficient
    elif position == 0:
        if ema_short_prev < ema_long_prev and ema_short_now > ema_long_now:
            signal_opportunities += 1
            if volatility > median_vol:
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
            else:
                # Volatility filter blocked this opportunity
                volatility_blocks += 1
                volatility_block_log.append(df.index[i])

    # Exit condition: short EMA crosses below long EMA
    elif position > 0 and ema_short_prev > ema_long_prev and ema_short_now < ema_long_now:
        proceeds = position * price * (1 - fee_rate)
        capital += proceeds
        pnl = (price - entry_price) * position
        success_count += int(pnl > 0)
        fail_count += int(pnl <= 0)
        sell_prices.append((df.index[i], price))
        position = 0
        transaction_count += 1

    # Track portfolio value
    position_value = position * price
    total_value = capital + position_value
    portfolio_values.append(total_value)

# Store results
# Ensure portfolio and price data are aligned
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
print(f"Volatility blocks:     {volatility_blocks}")
if signal_opportunities > 0:
    block_pct = 100 * volatility_blocks / signal_opportunities
    print(f"Volatility block rate: {block_pct:.2f}% of {signal_opportunities} opportunities")

# Sharpe ratio (risk-adjusted return)
returns = df["PortfolioValue"].pct_change().dropna()
if returns.std() != 0:
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
else:
    sharpe = 0  # Avoid division by zero if returns have no variance
print(f"Sharpe Ratio:          {sharpe:.2f}")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["PortfolioValue"], label="Strategy Portfolio", color="blue")

# Buy/sell markers
for date, price in buy_prices:
    plt.scatter(date, price, marker="^", color="green", zorder=5)
for date, price in sell_prices:
    plt.scatter(date, price, marker="v", color="red", zorder=5)

# Benchmark: Buy-and-Hold performance for comparison
df["BuyHold"] = initial_capital * (df["Price"] / df["Price"].iloc[0])
plt.plot(df.index, df["BuyHold"], label="Buy & Hold Benchmark", linestyle="--", color="gray")

# Custom legend
portfolio_line = mlines.Line2D([], [], color='blue', label='Strategy Portfolio')
buy_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=8, label='Buy')
sell_marker = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=8, label='Sell')
benchmark_line = mlines.Line2D([], [], color='gray', linestyle='--', label='Buy & Hold')
plt.legend(handles=[portfolio_line, buy_marker, sell_marker, benchmark_line], loc="upper left")

plt.title("EMA Crossover Strategy with Adaptive Volatility vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

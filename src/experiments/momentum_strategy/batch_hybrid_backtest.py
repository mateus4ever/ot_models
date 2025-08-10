import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# Strategy Parameters
initial_capital = 10_000
fee_rate = 0.001
position_fraction = 0.2
stop_loss_pct = 0.05
trailing_stop_pct = 0.03

def run_strategy(filename):
    df = pd.read_csv(filename, parse_dates=["Datetime"])
    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)
    df["Price"] = df["Close"]

    # Indicators
    exp1 = df["Price"].ewm(span=12, adjust=False).mean()
    exp2 = df["Price"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    window = 14
    delta = df["Price"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(window=window).mean()
    roll_down = loss.rolling(window=window).mean()
    RS = roll_up / roll_down
    df["RSI"] = 100.0 - (100.0 / (1.0 + RS))

    df["SMA50"] = df["Price"].rolling(window=50).mean()

    capital = initial_capital
    position = 0
    entry_price = 0
    peak_price = 0
    portfolio_values = []

    for i in range(1, len(df)):
        price = df["Price"].iloc[i]
        macd, signal, rsi = df["MACD"].iloc[i], df["Signal"].iloc[i], df["RSI"].iloc[i]
        sma50 = df["SMA50"].iloc[i]
        macd_prev = df["MACD"].iloc[i - 1]
        signal_prev = df["Signal"].iloc[i - 1]
        rsi_prev = df["RSI"].iloc[i - 1]

        macd_crossover = macd_prev < signal_prev and macd > signal
        rsi_rising = rsi > rsi_prev

        if position == 0 and macd_crossover and rsi_rising and price > sma50:
            investment = capital * position_fraction
            shares = investment // (price * (1 + fee_rate))
            if shares > 0:
                cost = shares * price * (1 + fee_rate)
                capital -= cost
                position = shares
                entry_price = price
                peak_price = price

        if position > 0:
            peak_price = max(peak_price, price)
            stop_loss = price < entry_price * (1 - stop_loss_pct)
            trailing_stop = price < peak_price * (1 - trailing_stop_pct)
            macd_bearish = macd_prev > signal_prev and macd < signal
            rsi_falling = rsi < rsi_prev

            if stop_loss or trailing_stop or (macd_bearish and rsi_falling):
                proceeds = position * price * (1 - fee_rate)
                capital += proceeds
                position = 0

        position_value = position * price
        total_value = capital + position_value
        portfolio_values.append(total_value)

    df = df.iloc[:len(portfolio_values)].copy()
    df["PortfolioValue"] = portfolio_values
    final_value = df["PortfolioValue"].iloc[-1]
    returns = df["PortfolioValue"].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252)

    return {
        "ticker": filename.split("_")[0].upper(),
        "final_value": final_value,
        "return_pct": (final_value / initial_capital - 1) * 100,
        "sharpe_ratio": sharpe
    }

# Run batch
results = []
for file in ["aapl_eod.csv", "msft_eod.csv", "spy_eod.csv", "qqq_eod.csv", "tsla_eod.csv"]:
    if os.path.exists(file):
        result = run_strategy(file)
        results.append(result)
    else:
        print(f"File not found: {file}")

# Display results
df_results = pd.DataFrame(results)
print("\n📊 Backtest Summary:")
print(df_results.sort_values(by="sharpe_ratio", ascending=False).to_string(index=False))

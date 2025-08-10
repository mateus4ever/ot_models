import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

plt.rcParams['figure.figsize'] = (18, 9)
plt.style.use('fivethirtyeight')

# --- Load data --------------------------------------------------------------
spy  = pd.read_csv("spy.csv", parse_dates=["datetime"], index_col="datetime")
qqq  = pd.read_csv("qqq.csv", parse_dates=["datetime"], index_col="datetime")

# Keep only the close columns and align on the DatetimeIndex
prices = pd.DataFrame({
    'SPY': spy['close'],
    'QQQ': qqq['close']
}).dropna()                        # drop rows where either price is missing

# --- Compute spread & correlation ------------------------------------------
prices['spread'] = prices['SPY'] - prices['QQQ']     # SPY minus QQQ
corr = prices['SPY'].corr(prices['QQQ'])

# --- Plot ------------------------------------------------------------------
plt.plot(prices.index, prices['spread'])
plt.title('SPY – QQQ spread over time')
plt.ylabel('Price difference ($)')
plt.xlabel('Date')
plt.show()

print(f"Correlation between SPY and QQQ: {corr:.4f}")

# Calculate the z-score of the spread
prices['z-score'] = (prices['spread'] - prices['spread'].mean()) / prices['spread'].std()

# Set the z-score threshold
threshold = 1

# Create columns for long and short signals
prices['long_entry'] = 0
prices['short_entry'] = 0
prices['long_exit'] = 0
prices['short_exit'] = 0

# Generate trading signals based on z-score
prices.loc[prices['z-score'] <= -threshold, 'long_entry'] = 1
prices.loc[prices['z-score'] >= threshold, 'short_entry'] = 1
prices.loc[prices['z-score'] * prices['z-score'].shift(1) < 0, 'long_exit'] = 1
prices.loc[prices['z-score'] * prices['z-score'].shift(1) < 0, 'short_exit'] = 1

#Plotting
plt.figure(figsize=(12,6))
prices['z-score'].plot()
plt.axhline(threshold, color='red', linestyle='--')
plt.axhline(-threshold, color='red', linestyle='--')
plt.ylabel('z-score')
plt.title('s-score with threshold=+/-1')
#plt.grid()
plt.show()


plt.figure(figsize=(15,7))

# Plot spread
prices['spread'].plot(label='spread', color='b')

# Plot buy signals
buy_signals = prices['spread'][prices['long_entry'] == 1]
sell_signals = prices['spread'][prices['short_entry'] == 1]
plt.plot(buy_signals, color='g', linestyle='None', marker='^', markersize=5, label='Buy Signal')
plt.plot(sell_signals, color='r', linestyle='None', marker='v', markersize=5, label='Sell Signal')

# Customize and show the plot
plt.legend()

plt.show()

plt.rcParams['figure.figsize'] = (14, 7)
plt.style.use('fivethirtyeight')

# ------------------------------------------------------------------
# back-test helper
# ------------------------------------------------------------------
def realistic_backtest(df,
                       initial_capital=100_000,
                       trade_size=20_000,
                       transaction_cost=0.001,
                       max_holding_days=10,
                       cooldown_days=3,
                       entry_z=1.5,
                       exit_z=0.5):
    capital = initial_capital
    position = 0
    entry_price = 0.0
    entry_index = None
    last_exit_date = None
    trade_log = []

    df = df.copy()
    df['returns'] = 0.0
    df['capital'] = capital

    # --- Define entry/exit signals
    df['long_entry'] = (df['z-score'].shift(1) < -entry_z).astype(int)
    df['short_entry'] = (df['z-score'].shift(1) > entry_z).astype(int)
    df['long_exit'] = (df['z-score'].shift(1) > -exit_z).astype(int)
    df['short_exit'] = (df['z-score'].shift(1) < exit_z).astype(int)

    for i in range(1, len(df)):
        today = df.index[i]
        spread = df['spread'].iloc[i]
        prev_spread = df['spread'].iloc[i - 1]

        # Check if cooldown is in effect
        if last_exit_date is not None and (today - last_exit_date).days < cooldown_days:
            df.at[today, 'capital'] = capital
            continue

        # --- While in position
        if position != 0:
            holding_days = (today - entry_index).days
            if position == 1:
                pnl = spread - prev_spread
            else:
                pnl = prev_spread - spread

            capital += pnl * trade_size / entry_price
            df.at[today, 'returns'] = pnl
            df.at[today, 'capital'] = capital

            should_exit = (
                    (position == 1 and df['long_exit'].iloc[i] == 1) or
                    (position == -1 and df['short_exit'].iloc[i] == 1) or
                    holding_days >= max_holding_days
            )

            if should_exit:
                capital -= trade_size * transaction_cost
                trade_log.append({
                    'exit_date': today,
                    'exit_spread': spread,
                    'pnl': capital - initial_capital,
                    'holding_days': holding_days,
                    'direction': 'long' if position == 1 else 'short'
                })
                position = 0
                entry_price = 0
                entry_index = None
                last_exit_date = today

        # --- Flat, check entry
        elif position == 0:
            if df['long_entry'].iloc[i] == 1:
                position = 1
                entry_price = spread
                entry_index = today
                capital -= trade_size * transaction_cost
                trade_log.append({'entry_date': today, 'entry_spread': spread, 'direction': 'long'})

            elif df['short_entry'].iloc[i] == 1:
                position = -1
                entry_price = spread
                entry_index = today
                capital -= trade_size * transaction_cost
                trade_log.append({'entry_date': today, 'entry_spread': spread, 'direction': 'short'})

        df.at[today, 'capital'] = capital

    trades = pd.DataFrame(trade_log)
    return df, trades

def summarize_trades(trades, initial_capital):
    trades = trades.copy()

    # Separate entry and exit rows
    entries = trades[trades['entry_date'].notna()].reset_index(drop=True)
    exits = trades[trades['exit_date'].notna()].reset_index(drop=True)

    if len(exits) == 0:
        print("No completed trades.")
        return

    # Merge entries and exits by order
    full_trades = pd.merge(entries, exits, left_index=True, right_index=True, suffixes=('_entry', '_exit'))

    full_trades['holding_days'] = full_trades['holding_days_exit']
    full_trades['pnl'] = full_trades['pnl_exit']

    # Summary stats
    total_trades = len(full_trades)
    long_trades = (full_trades['direction_entry'] == 'long').sum()
    short_trades = (full_trades['direction_entry'] == 'short').sum()
    avg_days = full_trades['holding_days'].mean()
    avg_pnl = full_trades['pnl'].mean()
    win_rate = (full_trades['pnl'] > 0).mean() * 100
    total_pnl = full_trades['pnl'].sum()

    print("\n📊 TRADE SUMMARY")
    print(f"Total trades:       {total_trades}")
    print(f"  Long trades:      {long_trades}")
    print(f"  Short trades:     {short_trades}")
    print(f"Avg holding period: {avg_days:.1f} days")
    print(f"Avg PnL per trade:  ${avg_pnl:.2f}")
    print(f"Total PnL:          ${total_pnl:.2f}")
    print(f"Win rate:           {win_rate:.1f}%")
    print(f"Ending capital:     ${initial_capital + total_pnl:,.2f}")

prices['long_entry'] = prices['z-score'].shift(1) < -2
prices['short_entry'] = prices['z-score'].shift(1) > 2
prices['long_exit']  = prices['z-score'].shift(1) > 0
prices['short_exit'] = prices['z-score'].shift(1) < 0

df = prices.astype(int)  # convert signals to 0/1
results, trades = realistic_backtest(df)
summarize_trades(trades, initial_capital=100_000)

print(f"Final capital: ${results['capital'].iloc[-1]:,.2f}")
print(f"Total return: {(results['capital'].iloc[-1] - 100_000):.2f}")
print(trades.tail())

# Optional: save trade log
trades.to_csv("trades.csv")
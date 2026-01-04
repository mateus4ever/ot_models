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
def backtest(prices, roll=30):
    """
    prices must contain columns:
        spread, long_entry, short_entry, long_exit, short_exit
    All lower-case.
    """
    res = prices.copy()

    # 1) z-score (rolling or full-sample; choose one)
    # --- option A: full-sample
    # res['zscore'] = zscore(res['spread'], nan_policy='omit')
    # --- option B: rolling (preferred for trading)
    """"
    removing the lookahead-bias
    res['zscore'] = (
        res['spread'] - res['spread'].rolling(roll).mean()
    ) / res['spread'].rolling(roll).std()
    """
    roll_mean = res['spread'].rolling(roll, min_periods=roll).mean().shift(1)
    roll_std = res['spread'].rolling(roll, min_periods=roll).std().shift(1)
    res['zscore'] = (res['spread'] - roll_mean) / roll_std

    # 2) bookkeeping columns
    res['returns'] = 0.0
    res['profit']  = 0.0
    position = 0     #  1 = long spread | –1 = short spread | 0 = flat
    profit   = 0.0

    for i in range(1, len(res)):
        # accumulate P/L while in a position ------------------------
        if position == 1:   # long spread
            pnl = res.iloc[i]['spread'] - res.iloc[i-1]['spread']
            res.iloc[i, res.columns.get_loc('returns')] = pnl
            profit += pnl
            if res.iloc[i]['long_exit'] == 1:
                position = 0

        elif position == -1:  # short spread
            pnl = res.iloc[i-1]['spread'] - res.iloc[i]['spread']
            res.iloc[i, res.columns.get_loc('returns')] = pnl
            profit += pnl
            if res.iloc[i]['short_exit'] == 1:
                position = 0

        res.iloc[i, res.columns.get_loc('profit')] = profit

        # check for new entries ------------------------------------
        if position == 0:
            if res.iloc[i]['long_entry'] == 1:
                position = 1
            elif res.iloc[i]['short_entry'] == 1:
                position = -1

    return res


def capital_backtest(prices,
                     initial_capital=100_000,
                     trade_size=20_000,
                     transaction_cost=0.001,
                     roll=30):
    """
    Capital-aware backtest for a spread trading strategy.

    prices: DataFrame with a 'spread' column.
    Returns: result DataFrame with capital and profit tracking, plus a trade log.
    """

    # Copy the input to avoid modifying original
    df = prices.copy()

    # --------------------------------------------------------
    # 1. Compute rolling z-score with no lookahead bias
    # --------------------------------------------------------
    roll_mean = df['spread'].rolling(roll, min_periods=roll).mean().shift(1)
    roll_std = df['spread'].rolling(roll, min_periods=roll).std().shift(1)
    df['zscore'] = (df['spread'] - roll_mean) / roll_std

    # --------------------------------------------------------
    # 2. Generate entry/exit signals based on z-score
    # --------------------------------------------------------
    entry_threshold = 1.5
    exit_threshold = 0.5

    df['long_entry'] = (df['zscore'] < -entry_threshold).astype(int)
    df['short_entry'] = (df['zscore'] > entry_threshold).astype(int)
    df['long_exit'] = (df['zscore'] > -exit_threshold).astype(int)
    df['short_exit'] = (df['zscore'] < exit_threshold).astype(int)

    # --------------------------------------------------------
    # 3. Initialize trading state and tracking variables
    # --------------------------------------------------------
    capital = initial_capital  # total cash value
    position = 0  # +1 = long, -1 = short, 0 = flat
    entry_price = 0.0  # price when trade was entered
    trade_log = []  # to store executed trades

    # Add columns to track results
    df['returns'] = 0.0  # daily raw return from spread movement
    df['capital'] = capital  # daily capital (updated if in position)
    df['profit'] = 0.0  # capital - initial capital

    # --------------------------------------------------------
    # 4. Main backtest loop
    # --------------------------------------------------------
    for i in range(1, len(df)):
        today = df.index[i]
        spread = df['spread'].iloc[i]
        prev_spread = df['spread'].iloc[i - 1]

        # ----------------------------------------------
        # If in a position → compute daily return
        # ----------------------------------------------
        if position != 0:
            # Calculate PnL in spread units
            pnl = spread - prev_spread if position == 1 else prev_spread - spread

            # Update capital based on PnL (scaled by trade size)
            capital += pnl * (trade_size / entry_price)

            # Record return and capital
            df.iloc[i, df.columns.get_loc('returns')] = pnl
            df.iloc[i, df.columns.get_loc('capital')] = capital
            df.iloc[i, df.columns.get_loc('profit')] = capital - initial_capital

            # Check if it's time to exit
            should_exit = (
                    (position == 1 and df['long_exit'].iloc[i] == 1) or
                    (position == -1 and df['short_exit'].iloc[i] == 1)
            )

            if should_exit:
                # Apply transaction cost on exit
                capital -= trade_size * transaction_cost
                df.iloc[i, df.columns.get_loc('capital')] = capital
                df.iloc[i, df.columns.get_loc('profit')] = capital - initial_capital

                # Log the exit
                trade_log.append({
                    'exit_date': today,
                    'exit_spread': spread,
                    'direction': 'long' if position == 1 else 'short',
                    'capital': capital
                })

                # Reset position
                position = 0
                entry_price = 0.0

        # ----------------------------------------------
        # If flat → check for entry signal
        # ----------------------------------------------
        if position == 0:
            if df['long_entry'].iloc[i] == 1:
                # Enter long position
                position = 1
                entry_price = spread
                capital -= trade_size * transaction_cost
                trade_log.append({
                    'entry_date': today,
                    'entry_spread': spread,
                    'direction': 'long'
                })

            elif df['short_entry'].iloc[i] == 1:
                # Enter short position
                position = -1
                entry_price = spread
                capital -= trade_size * transaction_cost
                trade_log.append({
                    'entry_date': today,
                    'entry_spread': spread,
                    'direction': 'short'
                })

            # Update flat-day capital and profit
            df.iloc[i, df.columns.get_loc('capital')] = capital
            df.iloc[i, df.columns.get_loc('profit')] = capital - initial_capital

    # --------------------------------------------------------
    # 5. Return enriched results and trade log
    # --------------------------------------------------------
    trades = pd.DataFrame(trade_log)
    return df, trades
def capital_backtest_staggered(
        prices,                       # DataFrame with a 'spread' column
        initial_capital=100_000,      # starting money
        trade_size=20_000,            # full position size (split into 3 chunks)
        transaction_cost=0.001,       # 0.1 % fee/slippage per entry/exit chunk
        roll=30,                      # look-back window for z-score
        exit_threshold=0.3,           # mean-reversion exit trigger (|z| < 0.3)
        max_holding_days=10,          # force close after this many days
        stop_loss_pct=0.02):          # hard stop-loss: 2 % of trade_size
    """
    Three-tier staggered entry (z = 1.0 / 1.5 / 2.0) + three independent exits:
        • Mean-reversion:   |z| < exit_threshold
        • Hard stop-loss:   unrealised loss ≥ stop_loss_pct * trade_size
        • Max holding days: position open ≥ max_holding_days
    Returns:
        df      – daily history with capital & PnL
        trades  – list of entry/exit events
    """
    # ------------------------------------------------------------------
    # 0) Copy input so we never mutate caller’s DataFrame
    # ------------------------------------------------------------------
    df = prices.copy()

    # ------------------------------------------------------------------
    # 1) Look-ahead-safe rolling z-score
    # ------------------------------------------------------------------
    roll_mean = df['spread'].rolling(roll, min_periods=roll).mean().shift(1)
    roll_std  = df['spread'].rolling(roll, min_periods=roll).std().shift(1)
    df['zscore'] = (df['spread'] - roll_mean) / roll_std

    # ------------------------------------------------------------------
    # 2) Book-keeping columns
    # ------------------------------------------------------------------
    df['returns']           = 0.0          # raw spread move per unit
    df['capital']           = initial_capital
    df['profit']            = 0.0          # capital – initial_capital
    df['position_fraction'] = 0            # 0 (no trade) … 3 (full size)
    df['direction']         = 0            # +1 long, –1 short, 0 flat
    df['days_in_trade']     = 0            # ageing counter

    # ------------------------------------------------------------------
    # 3) Strategy parameters
    # ------------------------------------------------------------------
    entry_levels   = [1.0, 1.5, 2.0]       # z-score tiers for 1⁄3, 2⁄3, 3⁄3
    chunk_size     = trade_size / 3        # value of a single tier
    stop_loss_cash = trade_size * stop_loss_pct

    # ------------------------------------------------------------------
    # 4) State variables
    # ------------------------------------------------------------------
    capital        = initial_capital
    position       = 0                     # +1 long, –1 short, 0 flat
    entry_price    = 0.0                   # weighted avg spread on entry
    size_entered   = 0.0                   # $ currently deployed
    entry_date     = None                  # date when first chunk was opened
    trade_log      = []                    # list(dict) for analysis

    # ------------------------------------------------------------------
    # 5) Main loop – iterate day by day
    # ------------------------------------------------------------------
    for i in range(1, len(df)):
        today        = df.index[i]
        spread       = df['spread'].iloc[i]
        prev_spread  = df['spread'].iloc[i - 1]
        z            = df['zscore'].iloc[i]

        # --------------------------------------------------------------
        # A) If we have an open position → update PnL + age
        # --------------------------------------------------------------
        if position != 0:
            # 1) daily PnL from spread move (per $ deployed)
            spread_pnl  = (spread - prev_spread) if position == 1 else (prev_spread - spread)
            capital    += spread_pnl * (size_entered / entry_price)

            # 2) ageing & unrealised PnL for risk exits
            days_held   = (today - entry_date).days
            unreal_pnl  = (spread - entry_price) if position == 1 else (entry_price - spread)
            unreal_cash = unreal_pnl * (size_entered / entry_price)

            # 3) write daily book-keeping
            df.iloc[i, df.columns.get_loc('returns')]           = spread_pnl
            df.iloc[i, df.columns.get_loc('capital')]           = capital
            df.iloc[i, df.columns.get_loc('profit')]            = capital - initial_capital
            df.iloc[i, df.columns.get_loc('days_in_trade')]     = days_held
            df.iloc[i, df.columns.get_loc('position_fraction')] = int(size_entered // chunk_size)
            df.iloc[i, df.columns.get_loc('direction')]         = position

            # 4) check *any* exit condition
            exit_by_mean   = abs(z) < exit_threshold
            exit_by_timing = days_held >= max_holding_days
            exit_by_stop   = unreal_cash <= -stop_loss_cash

            if exit_by_mean or exit_by_timing or exit_by_stop:
                # pay exit cost
                capital -= transaction_cost * size_entered
                # record to DataFrame *after* fees
                df.iloc[i, df.columns.get_loc('capital')] = capital
                df.iloc[i, df.columns.get_loc('profit')]  = capital - initial_capital

                # log exit event
                trade_log.append({
                    'exit_date'  : today,
                    'exit_spread': spread,
                    'reason'     : ('mean'   if exit_by_mean   else
                                     'max_day' if exit_by_timing else
                                     'stop_loss'),
                    'direction'  : 'long' if position == 1 else 'short',
                    'size'       : size_entered,
                    'capital'    : capital
                })

                # reset position state
                position       = 0
                size_entered   = 0.0
                entry_price    = 0.0
                entry_date     = None
                df.iloc[i, df.columns.get_loc('position_fraction')] = 0
                df.iloc[i, df.columns.get_loc('direction')]         = 0
                df.iloc[i, df.columns.get_loc('days_in_trade')]     = 0

        # --------------------------------------------------------------
        # B) Flat or scaling in – evaluate entry tiers
        # --------------------------------------------------------------
        entry_signal = (
            (z < -entry_levels[0] and position in [0, 1]) or
            (z >  entry_levels[0] and position in [0, -1])
        )

        if entry_signal:
            dir_sign = 1 if z < 0 else -1  # desired direction (+1 long, –1 short)

            # ── First chunk ────────────────────────────────────────────
            if position == 0:
                position       = dir_sign
                entry_price    = spread
                size_entered   = chunk_size
                entry_date     = today
                capital       -= chunk_size * transaction_cost

                trade_log.append({
                    'entry_date' : today,
                    'entry_spread': spread,
                    'direction'  : 'long' if dir_sign == 1 else 'short',
                    'chunk'      : 1
                })

            # ── Additional chunks (scale-in) ──────────────────────────
            elif position == dir_sign and size_entered < trade_size:
                current_level = int(size_entered // chunk_size)  # 1 or 2
                # add next chunk only if z exceeded next threshold
                if current_level < 3 and abs(z) > entry_levels[current_level]:
                    # weighted-avg entry price update
                    entry_price  = ((entry_price * size_entered) + spread * chunk_size) / (size_entered + chunk_size)
                    size_entered += chunk_size
                    capital      -= chunk_size * transaction_cost

                    trade_log.append({
                        'entry_date' : today,
                        'entry_spread': spread,
                        'direction'  : 'long' if dir_sign == 1 else 'short',
                        'chunk'      : current_level + 1
                    })

            # record common DF fields when flat or adding
            df.iloc[i, df.columns.get_loc('capital')]           = capital
            df.iloc[i, df.columns.get_loc('profit')]            = capital - initial_capital
            df.iloc[i, df.columns.get_loc('position_fraction')] = int(size_entered // chunk_size)
            df.iloc[i, df.columns.get_loc('direction')]         = position
            df.iloc[i, df.columns.get_loc('days_in_trade')]     = 0 if entry_date is None else (today - entry_date).days

    # ------------------------------------------------------------------
    # 6) Wrap up and return
    # ------------------------------------------------------------------
    trades = pd.DataFrame(trade_log)
    return df, trades



# ------------------------------------------------------------------
# run back-test
# ------------------------------------------------------------------
results = backtest(prices)     # closing_prices_clean already lower-case
results_capital, trades =capital_backtest(prices,20000, 2000,0.001)
results_capital_staggered, trades_staggered =capital_backtest_staggered(prices,20000, 2000,0.001)

print(f"total returns: {results['returns'].sum()*100:.2f}%")
results.to_csv('results.csv')

print(f"total returns capital: {results_capital['returns'].sum()*100:.2f}%")
results_capital.to_csv('results_capital.csv')

trades_staggered.to_csv('trades_staggered.csv')

# ------------------------------------------------------------------
# plots
# ------------------------------------------------------------------
sns.set(style='whitegrid')
fig, axs = plt.subplots(4, figsize=(12, 9), sharex=True)

# spread & entry markers
axs[0].plot(results.index, results['spread'], label='spread')
axs[0].plot(results[results['long_entry']==1].index,
            results[results['long_entry']==1]['spread'], 'g^', label='long entry')
axs[0].plot(results[results['short_entry']==1].index,
            results[results['short_entry']==1]['spread'], 'rv', label='short entry')
axs[0].set(title='spread and entry points', ylabel='spread ($)')
axs[0].legend()

# cumulative profit
axs[1].plot(results.index, results['profit'], color='purple', label='profit')
axs[1].set(title='profit over time', xlabel='date', ylabel='p/l ($)')
axs[1].legend()

# cumulative profit capital
axs[2].plot(results.index, results_capital['profit'], color='purple', label='profit')
axs[2].set(title='profit over time', xlabel='date', ylabel='p/l ($)')
axs[2].legend()

# cumulative profit capital staggered
axs[3].plot(results.index, results_capital_staggered['profit'], color='purple', label='profit')
axs[3].set(title='profit over time', xlabel='date', ylabel='p/l ($)')
axs[3].legend()

plt.tight_layout()
plt.show()

results_capital_staggered['daily_return'] = results_capital_staggered['capital'].pct_change()

# 2. Drop NaNs for first row
returns_capital_staggered = results_capital_staggered['daily_return'].dropna()

# 3. Compute annualized Sharpe ratio
mean_return = returns_capital_staggered.mean()
std_return = returns_capital_staggered.std()
sharpe_ratio = (mean_return / std_return) * (252 ** 0.5)

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

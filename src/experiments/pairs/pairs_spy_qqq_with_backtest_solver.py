import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from math import isfinite

plt.rcParams['figure.figsize'] = (18, 9)
plt.style.use('fivethirtyeight')

# --- Load data --------------------------------------------------------------
spy  = pd.read_csv("spy.csv", parse_dates=["datetime"], index_col="datetime")
qqq  = pd.read_csv("qqq.csv", parse_dates=["datetime"], index_col="datetime")
nvda  = pd.read_csv("nvda.csv", parse_dates=["datetime"], index_col="datetime")
ibm  = pd.read_csv("ibm.csv", parse_dates=["datetime"], index_col="datetime")

# Keep only the close columns and align on the DatetimeIndex
prices = pd.DataFrame({
    'IBM': ibm['close'],
    'NVDA': nvda['close']
}).dropna()

"""""
prices2 = pd.DataFrame({
    'SPY': spy['close'],
    'QQQ': qqq['close']
}).dropna()                        # drop rows where either price is missing
"""


# --- Compute spread & correlation ------------------------------------------
# prices['spread'] = prices['SPY'] - prices['QQQ']     # SPY minus QQQ
# corr = prices['SPY'].corr(prices['QQQ'])

prices['spread'] = prices['IBM'] - prices['NVDA']     # SPY minus QQQ
corr = prices['IBM'].corr(prices['NVDA'])

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

def objective_sharpe(params, prices):
    """
    Optimiser-friendly wrapper.

    params  – tuple = (exit_threshold, stop_loss_pct, max_holding_days)
    prices  – DataFrame with a 'spread' column (and index as dates)

    Returns
    -------
    float
        Negative Sharpe ratio  (optimisers *minimise*, we want to *maximise* Sharpe)
    """
    exit_thr, stop_pct, hold_days = params

    # --- run the back-test with supplied parameters -----------------
    results, _ = capital_backtest_staggered(
        prices,
        exit_threshold   = exit_thr,
        stop_loss_pct    = stop_pct,
        max_holding_days = int(hold_days),
    )

    # --- compute annualised Sharpe: mean / std * √252 --------------
    daily_rets = results['capital'].pct_change().dropna()
    if len(daily_rets) < 2 or daily_rets.std() == 0:
        return 1e6                            # infeasible → huge penalty

    sharpe = daily_rets.mean() / daily_rets.std() * (252 ** 0.5)
    return -sharpe                           # negative for minimiser

# ---------------------------------------------------------------
# 2) Simple grid search over parameter ranges
#    -----------------------------------------
#    Define discrete grids for each parameter, cartesian-product
# ---------------------------------------------------------------

# ---- define search ranges ---------------------------------------
param_grid = {
    'exit_thr' : [0.2, 0.25, 0.3, 0.35, 0.4],   # tighter ↔ looser exits
    'stop_pct' : [0.01, 0.015, 0.02, 0.03],     # 1–3 % stop-loss
    'hold_days': [5, 7, 10, 14, 20, 30],        # max holding period
}

# cartesian product of all combinations
grid_params = list(product(
    param_grid['exit_thr'],
    param_grid['stop_pct'],
    param_grid['hold_days']
))

best_sharpe  = -np.inf
best_params  = None

print(f"⏳ evaluating {len(grid_params)} grid points…")
for idx, p in enumerate(grid_params, 1):
    score  = objective_sharpe(p, prices)      # pass df explicitly
    if not isfinite(score):
        continue
    sharpe = -score                              # convert back to +ve
    if sharpe > best_sharpe:
        best_sharpe, best_params = sharpe, p

    if idx % 40 == 0:
        print(f"  {idx}/{len(grid_params)} done — best Sharpe so far: {best_sharpe:.2f}")

print("\n✅ GRID-SEARCH RESULT")
print(f"best Sharpe : {best_sharpe:.2f}")
print(f"best params : exit_thr={best_params[0]}, "
      f"stop_pct={best_params[1]}, hold_days={int(best_params[2])}")


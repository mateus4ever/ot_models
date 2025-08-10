import pandas as pd
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
ROLL_WINDOW   = 24
ATR_PERIOD    = 24                         # 24 hours  = 1 day of bars
SL_ATR        = 2                          # stop-loss distance  (×ATR)
TP_ATR        = 3                          # take-profit distance (×ATR)
RISK_PCT      = 0.02                       # 2 % of equity per trade
SPREAD        = 0.00010                    # ~1 pip in EUR/USD
START_CAPITAL = 10_000                     # starting account equity


def load_all_csv(data_dir: Path, resample_to_hour: bool = True) -> pd.DataFrame:
    column_names = ["datetime", "open", "high", "low", "close", "volume"]
    frames = []

    for csv_path in sorted(data_dir.glob("*.csv")):
        print(f"Loading {csv_path.name}")

        # Use ; separator and assign column names
        df = pd.read_csv(csv_path, sep=";", header=None, names=column_names, dtype=str)

        # Parse datetime
        df['timestamp'] = pd.to_datetime(df['datetime'], format="%Y%m%d %H%M%S", errors='coerce')

        # Convert OHLC to float
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        frames.append(df)

    if not frames:
        raise FileNotFoundError("No CSV files found.")

    df_all = pd.concat(frames).drop_duplicates(subset='timestamp')
    df_all = df_all.sort_values('timestamp')
    bad_rows = df_all[df_all['timestamp'].isna()]
    print("❗ Found bad timestamp rows:", len(bad_rows))
    df_all = df_all.dropna(subset=['timestamp'])

    if resample_to_hour:
        df_all = (
            df_all.set_index('timestamp')
            .resample('1H')
            .agg({'open': 'first',
                  'high': 'max',
                  'low': 'min',
                  'close': 'last'})
            .dropna()
            .reset_index()
        )

    print("✅ Loaded total rows:", len(df_all))
    return df_all[['timestamp', 'open', 'high', 'low', 'close']]

def walk_forward_backtest_chf_structure(df: pd.DataFrame) -> pd.DataFrame:
    equity_chf = START_CAPITAL
    equity_curve = []
    position = 0
    entry_price = np.nan
    qty = 0
    sl = tp = np.nan
    t0 = time.time()

    #for debugging
    signal_count = 0
    cooldown_pass = 0

    # 🆕 Precompute indicators
    df['ema_fast'] = df['close'].rolling(12).mean()
    df['ema_slow'] = df['close'].rolling(48).mean()
    atr_series = atr(df, ATR_PERIOD)
    atr_mean   = atr_series.rolling(100).mean()

    # 🆕 Cooldown tracking
    last_trade_time = None
    COOLDOWN_HOURS = 12
    break_margin = 0.0006  # ≈ 3 pips buffer

    trade_count = 0

    for i in range(ROLL_WINDOW, len(df)):
        signal = 0
        price = df['close'].iloc[i]
        usdchf_now = df['usdchf'].iloc[i]
        timestamp = df.index[i]

        # 🆕 Get precomputed ATR and its rolling mean
        atr_now = atr_series.iloc[i]
        atr_avg = atr_mean.iloc[i]

        # 🆕 Volatility filter: only trade in high-volatility regime
        if pd.notna(atr_now) and pd.notna(atr_avg) and atr_now > 1.5 * atr_avg:

            # Breakout range
            recent_high = df['high'].iloc[i - ROLL_WINDOW:i].max()
            recent_low  = df['low'].iloc[i - ROLL_WINDOW:i].min()

            # Trend filter: EMA crossover
            ema_fast = df['ema_fast'].iloc[i]
            ema_slow = df['ema_slow'].iloc[i]

            if pd.notna(ema_fast) and pd.notna(ema_slow):
                if price > recent_high + break_margin and ema_fast > ema_slow:
                    signal = 1
                elif price < recent_low - break_margin and ema_fast < ema_slow:
                    signal = -1

        if signal != 0:
            signal_count += 1
            if last_trade_time is None or timestamp > last_trade_time + pd.Timedelta(hours=COOLDOWN_HOURS):
                cooldown_pass += 1

        # 🆕 Check if cooldown has expired
        cooldown_ok = (
            last_trade_time is None or
            timestamp > last_trade_time + pd.Timedelta(hours=COOLDOWN_HOURS)
        )

        # Entry
        if position == 0 and signal != 0 and cooldown_ok:
            entry_price = price + np.sign(signal) * SPREAD / 2
            position = signal
            stop_price = entry_price - position * SL_ATR * atr_now

            qty = compute_position_size_chf(
                equity_chf = equity_chf,
                usdchf_rate = usdchf_now,
                entry_price = entry_price,
                stop_price = stop_price,
                risk_pct = RISK_PCT
            )

            sl = entry_price - position * SL_ATR * atr_now
            tp = entry_price + position * TP_ATR * atr_now

            last_trade_time = timestamp  # 🆕 set cooldown

        # Exit
        elif position != 0:
            hit_sl = (position == 1 and df['low'].iloc[i]  <= sl) or \
                     (position == -1 and df['high'].iloc[i] >= sl)
            hit_tp = (position == 1 and df['high'].iloc[i] >= tp) or \
                     (position == -1 and df['low'].iloc[i]  <= tp)

            if hit_sl or hit_tp:
                trade_count += 1
                exit_price = sl if hit_sl else tp
                pnl_usd = position * (exit_price - entry_price) * qty
                cost_usd = SPREAD * qty
                pnl_chf = (pnl_usd - cost_usd) * usdchf_now

                equity_chf += pnl_chf
                position = 0
                last_trade_time = timestamp  # ✅ Cooldown starts AFTER a trade ends

        equity_curve.append(equity_chf)

        if (i - ROLL_WINDOW) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"Progress {i}/{len(df)}  |  elapsed {elapsed:5.1f}s")

    print(f"✅ Done. Trades executed: {trade_count}")
    print(f"📊 Signals generated: {signal_count}, Passed cooldown: {cooldown_pass}")

    equity_df = pd.DataFrame({'equity_chf': equity_curve}, index=df.index[ROLL_WINDOW:])
    return equity_df, trade_count

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = np.maximum.reduce([
        df['high'] - df['low'],
        np.abs(df['high'] - df['close'].shift()),
        np.abs(df['low']  - df['close'].shift())
    ])
    return pd.Series(tr).rolling(period).mean()

def compute_position_size_chf(
    equity_chf: float,
    usdchf_rate: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.02
) -> float:
    risk_chf = equity_chf * risk_pct
    risk_usd = risk_chf / usdchf_rate
    stop_distance = abs(entry_price - stop_price)
    if stop_distance == 0:
        return 0
    return risk_usd / stop_distance

def analyze_equity_curve(equity: pd.Series, trade_count: int = None):
    returns = equity.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(24 * 252)  # hourly data
    final_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    drawdown = (equity / equity.cummax()) - 1
    max_dd = drawdown.min()

    print(f"Final return : {final_return * 100:.2f}%")
    print(f"Max drawdown : {abs(max_dd) * 100:.2f}%")
    print(f"Sharpe ratio : {sharpe:.2f}")

    if trade_count is not None:
        print(f"Trade count  : {trade_count}")
    else:
        print(f"Trade count  : ~{len(returns)} (approximate)")

# Load your main pair (e.g. EURUSD)
eurusd_df = load_all_csv(Path("data/eurusd"))

# Load USDCHF conversion rate data
usdchf_df = load_all_csv(Path("data/usdchf"))

# Merge USDCHF into main
eurusd_df = eurusd_df.set_index('timestamp')
usdchf_df = usdchf_df.set_index('timestamp')
eurusd_df['usdchf'] = usdchf_df['close'].reindex(eurusd_df.index, method='ffill')
eurusd_df = eurusd_df.dropna().copy()
eurusd_df['log_ret'] = np.log(eurusd_df['close'] / eurusd_df['close'].shift(1))


print(eurusd_df[['open', 'high', 'low', 'close']].head(20))
# Run the backtest
df_equity, trade_count = walk_forward_backtest_chf_structure(eurusd_df)
analyze_equity_curve(df_equity['equity_chf'], trade_count=trade_count)

# plt.plot(eq_chf.index, eq_chf['equity_chf'])
# plt.title("Equity Curve (CHF)")
# plt.xlabel("Time")
# plt.ylabel("Equity in CHF")
# plt.grid(True)
# plt.show()
"""
Hidden-Markov-Model Regime-Trading Back-test
===========================================

Pair:           EUR/USD (or any FX pair with the same column layout)
Granularity:    Hourly candles
Data window:    2023-07-07 .. 2025-07-07  (≈ 17 500 rows)

Strategy idea
-------------
1)  Use a *2-state Gaussian HMM* fitted on the last 6 months of hourly **log-returns**.
    • State 0 ≈ low volatility  → price tends to mean-revert
    • State 1 ≈ high volatility → price tends to trend

2)  Each new bar:
    • Infer the probability that we are in each state.
    • If the model is **≥ 60 % confident** we are in the trending state,
      and that state’s mean return is positive → go long (or short if mean < 0).
    • If the model is ≥ 60 % confident we are in the mean-reverting state,
      but price is ≥ 1 ATR away from a 24-hour EMA → fade the move back to EMA.

3)  Position sizing = 2 % of current equity risked per trade.
    Stop-loss  = 2 ATR,  Take-profit = 3 ATR,  spread cost = 1 pip.

The code keeps everything in one file for clarity.  Feel free to modularise later.
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM          # pip install hmmlearn
from pathlib import Path
import time

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
DATA_CSV      = Path("eurusd_hourly.csv")  # <-- rename to your own file
ROLL_WINDOW   = 24 * 24 * 6                # 6 months ≈ 4320 hours
HMM_STATES    = 2                          # two-state model
CONF_TH       = 0.60                       # model-confidence threshold
ATR_PERIOD    = 24                         # 24 hours  = 1 day of bars
SL_ATR        = 2                          # stop-loss distance  (×ATR)
TP_ATR        = 3                          # take-profit distance (×ATR)
RISK_PCT      = 0.02                       # 2 % of equity per trade
SPREAD        = 0.00010                    # ~1 pip in EUR/USD
START_CAPITAL = 10_000                     # starting account equity

# # ----------------------------------------------------------------------
# # Helper functions
# # ----------------------------------------------------------------------
# def load_data(path: Path) -> pd.DataFrame:
#     """
#     Read the CSV, sort chronologically, and compute hourly log-returns.
#     The file must contain: timestamp,open,high,low,close  (volume ignored).
#     """
#     df = pd.read_csv(path, parse_dates=['timestamp'])
#     df = df.sort_values('timestamp').set_index('timestamp')
#     df['log_ret'] = np.log(df['close']).diff()          # stationarised series
#     return df.dropna()

# ----------------------------------------------------------------------
# CONFIG – point this to the folder that holds ALL your CSVs
# ----------------------------------------------------------------------
DATA_DIR   = Path("data")       # ← folder with many .csv files
RESAMPLE_TO_HOUR = True         # set False if your files are already hourly

# ----------------------------------------------------------------------
# Utility: load every CSV in DATA_DIR and return one clean DataFrame
# ----------------------------------------------------------------------
from pathlib import Path
import pandas as pd

def load_all_csv(data_dir: Path, resample_to_hour: bool = True) -> pd.DataFrame:
    column_names = ["date", "time", "open", "high", "low", "close", "volume"]
    frames = []

    for csv_path in sorted(data_dir.glob("*.csv")):
        print(f"Loading {csv_path.name}")
        df = pd.read_csv(csv_path, header=None, names=column_names)

        # Combine date and time, then parse into datetime
        df['timestamp'] = pd.to_datetime(df['date'] + " " + df['time'], format="%Y.%m.%d %H:%M")

        frames.append(df)

    if not frames:
        raise FileNotFoundError("No CSV files found.")

    df_all = pd.concat(frames).drop_duplicates(subset='timestamp')
    df_all = df_all.sort_values('timestamp')

    if resample_to_hour:
        df_all = (
            df_all.set_index('timestamp')
                  .resample('1H')
                  .agg({'open': 'first',
                        'high': 'max',
                        'low' : 'min',
                        'close':'last'})
                  .dropna()
                  .reset_index()
        )

    print("✅ Loaded total rows:", len(df_all))
    return df_all[['timestamp', 'open', 'high', 'low', 'close']]


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder). Returns a pandas Series.
    """
    high_low   = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close  = (df['low']  - df['close'].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ----------------------------------------------------------------------
# Core back-test
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Walk-forward back-test with convergence checks
# ----------------------------------------------------------------------
def walk_forward_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walks forward from ROLL_WINDOW to end of df.
    • Fits 2-state Gaussian HMM on past window (n_iter=50, fast).
    • Skips bar if model didn't converge (robustness).
    • Tracks equity, position, SL/TP as before.
    """
    equity        = START_CAPITAL
    equity_curve  = []
    position      = 0
    entry_price   = np.nan
    qty           = 0
    sl = tp       = np.nan
    failures      = 0
    t0 = time.time()

    for i in range(ROLL_WINDOW, len(df)):
        # ------------------------------------------------------------------
        # 1. Fit HMM on past window
        # ------------------------------------------------------------------
        window = df[['log_ret']].iloc[i-ROLL_WINDOW : i]

        model = GaussianHMM(
            n_components = 2,
            covariance_type = "full",
            n_iter = 50,        # ✔️ faster, usually converges
            verbose = False,
            random_state = 42,
        )

        try:
            model.fit(window)
        except ValueError as e:
            print(f"[{df.index[i]}] ⚠️ HMM fit error: {e} — skipping bar")
            failures += 1
            equity_curve.append(equity)
            continue

        if not model.monitor_.converged:
            # Not reliable, skip this bar
            failures += 1
            equity_curve.append(equity)
            continue

        # ------------------------------------------------------------------
        # 2. Infer state probabilities
        # ------------------------------------------------------------------
        _, state_prob = model.score_samples(df[['log_ret']].iloc[i-1 : i])
        means = model.means_.flatten()
        trend_state = int(np.argmax(np.abs(means)))
        mean_trend  = means[trend_state]

        p_trend   = state_prob[0, trend_state]
        p_meanrev = state_prob[0, 1 - trend_state]

        # ------------------------------------------------------------------
        # 3. Compute ATR, EMA, price
        # ------------------------------------------------------------------
        df_slice    = df.iloc[: i + 1]
        price       = df_slice['close'].iloc[-1]
        current_atr = atr(df_slice, ATR_PERIOD).iloc[-1]
        ema         = df_slice['close'].ewm(span=ATR_PERIOD,
                                            adjust=False).mean().iloc[-1]

        # ------------------------------------------------------------------
        # 4. Generate trading signal
        # ------------------------------------------------------------------
        signal = 0
        if p_trend >= CONF_TH:
            signal =  1 if mean_trend > 0 else -1
        elif p_meanrev >= CONF_TH:
            if price - ema >  current_atr: signal = -1
            if price - ema < -current_atr: signal =  1

        # ------------------------------------------------------------------
        # 5. Position management
        # ------------------------------------------------------------------
        if position == 0 and signal != 0:
            entry_price = price + np.sign(signal) * SPREAD / 2
            position    = signal
            risk_usd    = equity * RISK_PCT
            qty         = risk_usd / (SL_ATR * current_atr)
            sl          = entry_price - position * SL_ATR * current_atr
            tp          = entry_price + position * TP_ATR * current_atr

        elif position != 0:
            hit_sl = (position == 1 and df['low'].iloc[i]  <= sl) or \
                     (position == -1 and df['high'].iloc[i] >= sl)
            hit_tp = (position == 1 and df['high'].iloc[i] >= tp) or \
                     (position == -1 and df['low'].iloc[i]  <= tp)

            if hit_sl or hit_tp:
                exit_price = sl if hit_sl else tp
                pnl        = position * (exit_price - entry_price) * qty
                equity    += pnl - SPREAD * qty
                position   = 0

        equity_curve.append(equity)

        # ------------------------------------------------------------------
        # 6. Progress print (every 1 000 bars)
        # ------------------------------------------------------------------
        if (i - ROLL_WINDOW) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"Progress {i}/{len(df)}  |  elapsed {elapsed:5.1f}s")

    print(f"✅ Back-test done. HMM non-convergence skipped: {failures} windows")
    return pd.DataFrame({'equity': equity_curve},
                        index=df.index[ROLL_WINDOW:])


# ----------------------------------------------------------------------
# Sharpe Ratio helper
# ----------------------------------------------------------------------
def calculate_sharpe_ratio(equity_series: pd.Series) -> float:
    """
    Compute hourly returns from equity, then compute Sharpe ratio.
    Assumes 24 hourly periods per day and 0% risk-free rate.
    """
    returns = equity_series.pct_change().dropna()
    if returns.std() == 0:
        return 0.0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(24)
    return sharpe


# ----------------------------------------------------------------------
# Main script entry-point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load data and sanity-check
    data = load_all_csv(DATA_DIR, resample_to_hour=RESAMPLE_TO_HOUR)
    data['log_ret'] = np.log(data['close']).diff()
    data = data.dropna()

    print("Loaded rows:", len(data))
    print("First timestamp:", data['timestamp'].iloc[0])
    print("Last timestamp :", data['timestamp'].iloc[-1])

    # data = load_data("data/DAT_MT_EURCHF_M1_2024.csv")
    # assert set(['open', 'high', 'low', 'close']).issubset(data.columns), \
    #     "CSV must contain open, high, low, close columns"

    # 2. Run walk-forward back-test
    eq_curve = walk_forward_backtest(data)

    # 3. Simple performance summary
    final_ret = eq_curve['equity'].iloc[-1] / START_CAPITAL - 1
    max_dd    = (eq_curve['equity'].cummax() - eq_curve['equity']).max() \
                / eq_curve['equity'].cummax().max()

    sharpe = calculate_sharpe_ratio(eq_curve['equity'])

    print(f"Final return  : {final_ret:6.2%}")
    print(f"Max drawdown  : {max_dd :6.2%}")
    print(f"Sharpe ratio  : {sharpe:6.2f}")
    print(f"Trade count   : (rough) {eq_curve['equity'].diff().ne(0).sum()}")
    # 4. Quick equity-curve plot (optional)
    #    Comment out the next 4 lines if you run headless / no display.
    import matplotlib.pyplot as plt
    eq_curve['equity'].plot(title='HMM Strategy Equity Curve')
    plt.ylabel("Equity (USD)")
    plt.grid(True)
    plt.show()

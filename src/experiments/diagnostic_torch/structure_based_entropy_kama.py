import pandas as pd
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
CONFIG = {
    'ROLL_WINDOW': 24,
    'ATR_PERIOD': 24,
    'RISK_PCT': 0.02,
    'SPREAD': 0.00010,
    'START_CAPITAL': 10_000,
    'COOLDOWN_HOURS': 0,
    'ENTROPY_WINDOW': 30,
    'VOL_CLUSTER_WINDOW': 5,
    'VOL_CLUSTER_THRESHOLD': 0.002,
    'SLOPE_WINDOW': 24,
    'KAMA_ER_WINDOW': 10,
    'KAMA_FAST': 2,
    'KAMA_SLOW': 30,
    'SL_ATR': 2.0,
    'TP_ATR': 3.0,
    'ATR_BREAKOUT_MULTIPLIER': 1.2
}

# ----------------------------------------------------------------------
# DATA LOADER
# ----------------------------------------------------------------------
def load_all_csv(data_dir: Path, resample_to_hour: bool = True) -> pd.DataFrame:
    column_names = ["datetime", "open", "high", "low", "close", "volume"]
    frames = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, sep=";", header=None, names=column_names, dtype=str)
        df['timestamp'] = pd.to_datetime(df['datetime'], format="%Y%m%d %H%M%S", errors='coerce')
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        frames.append(df)
    df_all = pd.concat(frames).drop_duplicates(subset='timestamp')
    df_all = df_all.sort_values('timestamp').dropna(subset=['timestamp'])
    if resample_to_hour:
        df_all = (
            df_all.set_index('timestamp')
            .resample('1H')
            .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
            .dropna()
            .reset_index()
        )
    return df_all[['timestamp', 'open', 'high', 'low', 'close']]

# ----------------------------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------------------------
def compute_features(df, config):
    df = df.dropna(subset=['high', 'low', 'close'])
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['entropy'] = df['log_ret'].rolling(config['ENTROPY_WINDOW']).apply(
        lambda x: -np.sum((p := np.histogram(x, bins=10)[0] / np.sum(np.histogram(x, bins=10)[0])) * np.log2(p + 1e-12)),
        raw=False
    )
    df['vol_cluster'] = (df['high'] - df['low']).rolling(config['VOL_CLUSTER_WINDOW']).mean() < config['VOL_CLUSTER_THRESHOLD']
    df['slope'] = df['close'].rolling(config['SLOPE_WINDOW']).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if x.isna().sum() == 0 else np.nan,
        raw=False
    )
    df['kama'] = compute_kama(df['close'], config['KAMA_ER_WINDOW'], config['KAMA_FAST'], config['KAMA_SLOW'])
    df['atr'] = atr(df, config['ATR_PERIOD'])
    print(f"[DEBUG] ATR NaN count: {df['atr'].isna().sum()} out of {len(df)} rows")
    return df

# ----------------------------------------------------------------------
# KAUFMAN'S ADAPTIVE MOVING AVERAGE (KAMA)
# ----------------------------------------------------------------------
def compute_kama(series, er_window, fast, slow):
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = pd.Series(index=series.index, dtype=float)
    er = series.diff(er_window).abs() / series.diff().abs().rolling(er_window).sum()
    er = er.replace([np.inf, -np.inf], 0).fillna(0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = pd.Series(index=series.index, dtype=float)
    kama.iloc[er_window] = series.iloc[er_window]
    for i in range(er_window + 1, len(series)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i - 1])
    return kama

# ----------------------------------------------------------------------
# ATR
# ----------------------------------------------------------------------
def atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = np.maximum.reduce([
        df['high'] - df['low'],
        np.abs(df['high'] - df['close'].shift()),
        np.abs(df['low'] - df['close'].shift())
    ])
    return pd.Series(tr, index=df.index).rolling(period).mean()

# ----------------------------------------------------------------------
# ENTRY SIGNAL LOGIC (MODERN INDICATORS)
# ----------------------------------------------------------------------
# Global counters for debugging (add at the top of your script or inside main for testing)
debug_counters = {
    "total_checks": 0,
    "vol_cluster_fail": 0,
    "entropy_fail": 0,
    "slope_nan": 0,
    "kama_nan": 0,
    "no_signal": 0,
    "buy": 0,
    "sell": 0
}

def generate_trade_signal_modern(df, i, config):
    debug_counters["total_checks"] += 1

    if not df['vol_cluster'].iloc[i]:
        debug_counters["vol_cluster_fail"] += 1
        return 0
    if df['entropy'].iloc[i] > config['ENTROPY_THRESHOLD']:
        debug_counters["entropy_fail"] += 1
        return 0
    if not pd.notna(df['slope'].iloc[i]):
        debug_counters["slope_nan"] += 1
        return 0
    if not pd.notna(df['kama'].iloc[i]):
        debug_counters["kama_nan"] += 1
        return 0

    # NEW: ATR breakout filter
    atr = df['atr'].iloc[i]
    range_now = df['high'].iloc[i] - df['low'].iloc[i]
    if range_now < config['ATR_BREAKOUT_MULTIPLIER'] * atr:
        return 0  # skip if no strong breakout

    trend = df['slope'].iloc[i]
    price = df['close'].iloc[i]
    kama = df['kama'].iloc[i]

    if trend > 0 and price > kama:
        debug_counters["buy"] += 1
        return 1
    elif trend < 0 and price < kama:
        debug_counters["sell"] += 1
        return -1

    debug_counters["no_signal"] += 1
    return 0

# ----------------------------------------------------------------------
# TRADE ENTRY/EXIT
# ----------------------------------------------------------------------
def enter_trade(signal, price, atr_now, equity_chf, config):
    if pd.isna(atr_now) or atr_now == 0:
        return signal, price, np.nan, np.nan, np.nan
    entry_price = price + np.sign(signal) * config['SPREAD'] / 2
    position = signal
    stop_price = entry_price - position * config['SL_ATR'] * atr_now
    qty = compute_position_size_chf(equity_chf, entry_price, stop_price, config['RISK_PCT'])
    sl = entry_price - position * config['SL_ATR'] * atr_now
    tp = entry_price + position * config['TP_ATR'] * atr_now
    return position, entry_price, qty, sl, tp

def exit_trade(df, i, position, entry_price, qty, sl, tp, equity_chf, config):
    price_hit = None
    if position == 1:
        if df['low'].iloc[i] <= sl:
            price_hit = sl
        elif df['high'].iloc[i] >= tp:
            price_hit = tp
    elif position == -1:
        if df['high'].iloc[i] >= sl:
            price_hit = sl
        elif df['low'].iloc[i] <= tp:
            price_hit = tp

    if price_hit is not None:
        pnl = position * (price_hit - entry_price) * qty
        cost = config['SPREAD'] * qty
        pnl_net = pnl - cost
        equity_chf += pnl_net
        return 0, equity_chf, True, pnl_net
    return position, equity_chf, False, 0

def compute_position_size_chf(equity_chf, entry_price, stop_price, risk_pct):
    risk_chf = equity_chf * risk_pct
    stop_distance = abs(entry_price - stop_price)
    return 0 if stop_distance == 0 else risk_chf / stop_distance

# ----------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------
def compute_metrics(equity, trades, wins, losses):
    returns = equity.pct_change().dropna()
    std = returns.std()
    if pd.isna(std) or std == 0:
        sharpe = 0.0
    else:
        sharpe = returns.mean() / std * np.sqrt(24 * 252)
    final_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    drawdown = (equity / equity.cummax()) - 1
    max_dd = drawdown.min()
    winrate = wins / trades if trades > 0 else 0
    profit_factor = np.nan if losses == 0 else wins / losses
    expectancy = (equity.iloc[-1] - equity.iloc[0]) / trades if trades > 0 else 0
    return {
        'Sharpe': sharpe,
        'FinalReturn': final_return,
        'MaxDrawdown': abs(max_dd),
        'Trades': trades,
        'Wins': wins,
        'Losses': losses,
        'WinRate': winrate,
        'ProfitFactor': profit_factor,
        'Expectancy': expectancy
    }

def compute_entropy_threshold(df, config):
    # Compute the 25th percentile of non-NaN entropy
    return df['entropy'].dropna().quantile(0.5)

# ----------------------------------------------------------------------
# BACKTEST LOOP (WITH REALISTIC ENTRY/EXIT)
# ----------------------------------------------------------------------
def run_backtest(df, config):
    equity = []
    capital = config['START_CAPITAL']
    position = 0
    wins = 0
    losses = 0
    trades = 0
    entry_price = np.nan
    qty = 0
    sl = tp = np.nan

    atr_skips = 0  # ← local counter for debugging

    start_index = max(config['ROLL_WINDOW'], config['ATR_PERIOD'])
    for i in range(start_index, len(df)):
        atr_now = df['atr'].iloc[i]
        if pd.isna(atr_now) or atr_now == 0:
            atr_skips += 1
            equity.append(capital)
            continue

        signal = generate_trade_signal_modern(df, i, config)
        price = df['close'].iloc[i]

        if position == 0 and signal != 0:
            position, entry_price, qty, sl, tp = enter_trade(signal, price, atr_now, capital, config)
            if not pd.isna(qty):
                trades += 1

        elif position != 0:
            position, capital, closed, pnl = exit_trade(df, i, position, entry_price, qty, sl, tp, capital, config)
            if closed:
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

        equity.append(capital)

    equity_series = pd.Series(equity, index=df.index[start_index:])
    metrics = compute_metrics(equity_series, trades, wins, losses)
    metrics['ATR_Skips'] = atr_skips  # ← attach to metrics
    return equity_series, metrics


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    df = load_all_csv(Path("data/eurusd"))
    df = df.set_index('timestamp').dropna()
    df = compute_features(df, CONFIG)
    CONFIG['ENTROPY_THRESHOLD'] = compute_entropy_threshold(df, CONFIG)

    equity_series, metrics = run_backtest(df, CONFIG)
    print(metrics)
    print("\n🔍 Signal Debug Summary:")
    for k, v in debug_counters.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

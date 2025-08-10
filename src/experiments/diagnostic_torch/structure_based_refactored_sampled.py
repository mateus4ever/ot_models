import pandas as pd
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
    'ATR_MULTIPLIER': 1.2,
    'SL_ATR': 2.0,
    'TP_ATR': 3.0,
    'BREAK_MARGIN': 0.0006
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
# BACKTEST MODULES
# ----------------------------------------------------------------------
def atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = np.maximum.reduce([
        df['high'] - df['low'],
        np.abs(df['high'] - df['close'].shift()),
        np.abs(df['low'] - df['close'].shift())
    ])
    return pd.Series(tr).rolling(period).mean()

def compute_position_size_chf(equity_chf, usdchf_rate, entry_price, stop_price, risk_pct):
    risk_chf = equity_chf * risk_pct
    risk_usd = risk_chf / usdchf_rate
    stop_distance = abs(entry_price - stop_price)
    return 0 if stop_distance == 0 else risk_usd / stop_distance

def generate_trade_signal(df, i, atr_now, atr_avg, config):
    if not pd.notna(atr_now) or not pd.notna(atr_avg): return 0
    if atr_now <= config['ATR_MULTIPLIER'] * atr_avg: return 0

    price = df['close'].iloc[i]
    ema_fast = df['ema_fast'].iloc[i]
    ema_slow = df['ema_slow'].iloc[i]

    if not pd.notna(ema_fast) or not pd.notna(ema_slow): return 0

    recent_high = df['high'].iloc[i - config['ROLL_WINDOW']:i].max()
    recent_low  = df['low'].iloc[i - config['ROLL_WINDOW']:i].min()

    if price > recent_high + config['BREAK_MARGIN'] and ema_fast > ema_slow:
        return 1
    elif price < recent_low - config['BREAK_MARGIN'] and ema_fast < ema_slow:
        return -1
    return 0

def enter_trade(signal, price, atr_now, usdchf_rate, equity_chf, config):
    entry_price = price + np.sign(signal) * config['SPREAD'] / 2
    position = signal
    stop_price = entry_price - position * config['SL_ATR'] * atr_now

    qty = compute_position_size_chf(
        equity_chf, usdchf_rate, entry_price, stop_price, config['RISK_PCT']
    )

    sl = entry_price - position * config['SL_ATR'] * atr_now
    tp = entry_price + position * config['TP_ATR'] * atr_now

    return position, entry_price, qty, sl, tp

def exit_trade(df, i, position, entry_price, qty, sl, tp, usdchf_rate, equity_chf, config):
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
        pnl_usd = position * (price_hit - entry_price) * qty
        cost_usd = config['SPREAD'] * qty
        pnl_chf = (pnl_usd - cost_usd) * usdchf_rate
        equity_chf += pnl_chf
        return 0, equity_chf, True, pnl_chf

    return position, equity_chf, False, 0

def walk_forward_backtest_chf_structure(df, config):
    equity_chf = config['START_CAPITAL']
    equity_curve = []
    position = 0
    entry_price = np.nan
    qty = 0
    sl = tp = np.nan
    trade_count = 0
    wins = 0
    losses = 0
    profit_sum = 0
    loss_sum = 0

    df['ema_fast'] = df['close'].rolling(12).mean()
    df['ema_slow'] = df['close'].rolling(48).mean()
    atr_series = atr(df, config['ATR_PERIOD'])
    atr_mean = atr_series.rolling(100).mean()

    last_trade_time = None

    for i in range(config['ROLL_WINDOW'], len(df)):
        row = df.iloc[i]
        price = row['close']
        usdchf_now = row['usdchf']
        timestamp = df.index[i]
        atr_now = atr_series.iloc[i]
        atr_avg = atr_mean.iloc[i]

        signal = generate_trade_signal(df, i, atr_now, atr_avg, config)
        cooldown_ok = last_trade_time is None or timestamp > last_trade_time + pd.Timedelta(hours=config['COOLDOWN_HOURS'])

        if position == 0 and signal != 0 and cooldown_ok:
            position, entry_price, qty, sl, tp = enter_trade(signal, price, atr_now, usdchf_now, equity_chf, config)
            last_trade_time = timestamp

        elif position != 0:
            position, equity_chf, trade_executed, pnl_chf = exit_trade(df, i, position, entry_price, qty, sl, tp, usdchf_now, equity_chf, config)
            if trade_executed:
                last_trade_time = timestamp
                trade_count += 1
                if pnl_chf > 0:
                    wins += 1
                    profit_sum += pnl_chf
                else:
                    losses += 1
                    loss_sum += abs(pnl_chf)

        equity_curve.append(equity_chf)

    equity_df = pd.DataFrame({'equity_chf': equity_curve}, index=df.index[config['ROLL_WINDOW']:])
    stats = {
        'Trades': trade_count,
        'Wins': wins,
        'Losses': losses,
        'WinRate': wins / trade_count if trade_count else 0,
        'ProfitFactor': profit_sum / loss_sum if loss_sum else np.nan,
        'Expectancy': (profit_sum - loss_sum) / trade_count if trade_count else 0
    }
    return equity_df, stats

def analyze_equity_curve(equity: pd.Series):
    returns = equity.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(24 * 252)
    final_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    drawdown = (equity / equity.cummax()) - 1
    max_dd = drawdown.min()
    return final_return, sharpe, abs(max_dd)

# ----------------------------------------------------------------------
# WALK-FORWARD SPLIT HELPER
# ----------------------------------------------------------------------
def split_walkforward(df, train_years=1, test_months=6):
    splits = []
    start = df.index.min()
    while start + pd.DateOffset(years=train_years + 1) < df.index.max():
        train_end = start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)
        splits.append((start, train_end, test_end))
        start += pd.DateOffset(months=test_months)
    return splits

# ----------------------------------------------------------------------
# CROSS-VALIDATION + WALK-FORWARD + OUT-OF-SAMPLE
# ----------------------------------------------------------------------
def run_walkforward_optimization(df, config):
    print("\n[WALK-FORWARD OPTIMIZATION]")
    splits = split_walkforward(df)
    all_results = []

    for idx, (train_start, train_end, test_end) in enumerate(splits):
        train_df = df[(df.index >= train_start) & (df.index < train_end)]
        test_df = df[(df.index >= train_end) & (df.index < test_end)]

        best_cfg = None
        best_sharpe = -np.inf

        for atr_mult, sl_atr, tp_atr, bm in itertools.product([1.0, 1.2, 1.5], [1.5, 2.0], [2.5, 3.0], [0.0006, 0.0010]):
            cfg = config.copy()
            cfg['ATR_MULTIPLIER'] = atr_mult
            cfg['SL_ATR'] = sl_atr
            cfg['TP_ATR'] = tp_atr
            cfg['BREAK_MARGIN'] = bm

            eq_train, _ = walk_forward_backtest_chf_structure(train_df.copy(), cfg)
            final_ret, sharpe, _ = analyze_equity_curve(eq_train['equity_chf'])
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_cfg = cfg

        eq_test, stats = walk_forward_backtest_chf_structure(test_df.copy(), best_cfg)
        final_ret, sharpe, max_dd = analyze_equity_curve(eq_test['equity_chf'])
        stats.update({
            'Period': f"{train_end.date()} → {test_end.date()}",
            'Sharpe': sharpe,
            'FinalReturn': final_ret,
            'MaxDrawdown': max_dd,
            'ATR_MULTIPLIER': best_cfg['ATR_MULTIPLIER'],
            'SL_ATR': best_cfg['SL_ATR'],
            'TP_ATR': best_cfg['TP_ATR'],
            'BREAK_MARGIN': best_cfg['BREAK_MARGIN']
        })
        all_results.append(stats)

    df_walk = pd.DataFrame(all_results)
    print(df_walk)
    return df_walk

# ----------------------------------------------------------------------
# OUT-OF-SAMPLE TEST (last year)
# ----------------------------------------------------------------------
def run_out_of_sample(df, config):
    cutoff = df.index.max() - pd.DateOffset(years=1)
    train_df = df[df.index < cutoff]
    test_df = df[df.index >= cutoff]

    best_cfg = None
    best_sharpe = -np.inf

    for atr_mult, sl_atr, tp_atr, bm in itertools.product([1.0, 1.2, 1.5], [1.5, 2.0], [2.5, 3.0], [0.0006, 0.0010]):
        cfg = config.copy()
        cfg['ATR_MULTIPLIER'] = atr_mult
        cfg['SL_ATR'] = sl_atr
        cfg['TP_ATR'] = tp_atr
        cfg['BREAK_MARGIN'] = bm

        eq_train, _ = walk_forward_backtest_chf_structure(train_df.copy(), cfg)
        final_ret, sharpe, _ = analyze_equity_curve(eq_train['equity_chf'])
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_cfg = cfg

    eq_test, stats = walk_forward_backtest_chf_structure(test_df.copy(), best_cfg)
    final_ret, sharpe, max_dd = analyze_equity_curve(eq_test['equity_chf'])
    stats.update({
        'Sharpe': sharpe,
        'FinalReturn': final_ret,
        'MaxDrawdown': max_dd,
        'ATR_MULTIPLIER': best_cfg['ATR_MULTIPLIER'],
        'SL_ATR': best_cfg['SL_ATR'],
        'TP_ATR': best_cfg['TP_ATR'],
        'BREAK_MARGIN': best_cfg['BREAK_MARGIN']
    })
    print("\n[OUT-OF-SAMPLE TEST]")
    print(stats)
    return stats

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    eurusd_df = load_all_csv(Path("data/eurusd"))
    usdchf_df = load_all_csv(Path("data/usdchf"))

    eurusd_df = eurusd_df.set_index('timestamp')
    usdchf_df = usdchf_df.set_index('timestamp')
    eurusd_df['usdchf'] = usdchf_df['close'].reindex(eurusd_df.index, method='ffill')
    eurusd_df = eurusd_df.dropna().copy()
    eurusd_df['log_ret'] = np.log(eurusd_df['close'] / eurusd_df['close'].shift(1))

    walk_results = run_walkforward_optimization(eurusd_df.copy(), CONFIG)
    oos_results = run_out_of_sample(eurusd_df.copy(), CONFIG)

     # Persist results to CSV
    walk_results.to_csv("walk_forward_results.csv", index=False)
    pd.DataFrame([oos_results]).to_csv("out_of_sample_results.csv", index=False)


if __name__ == "__main__":
    main()

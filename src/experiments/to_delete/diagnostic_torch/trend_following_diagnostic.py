import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
CONFIG = {
    'ATR_PERIOD': 24,
    'KAMA_ER_WINDOW': 10,
    'KAMA_FAST': 2,a
    'KAMA_SLOW': 30,
    'SLOPE_WINDOW': 24,
    'KALMAN_R': 0.001
}

# ----------------------------------------------------------------------
# DATA LOADING (EUR/USD ONLY)
# ----------------------------------------------------------------------
def load_all_csv(data_dir: Path) -> pd.DataFrame:
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
    df_all = (
        df_all.set_index('timestamp')
        .resample('1H')
        .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        .dropna()
    )
    return df_all[['open', 'high', 'low', 'close']]

# ----------------------------------------------------------------------
# INDICATOR FUNCTIONS
# ----------------------------------------------------------------------
def compute_kama(series, er_window, fast, slow):
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    er = series.diff(er_window).abs() / series.diff().abs().rolling(er_window).sum()
    er = er.replace([np.inf, -np.inf], 0).fillna(0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = pd.Series(index=series.index, dtype=float)
    kama.iloc[er_window] = series.iloc[er_window]
    for i in range(er_window + 1, len(series)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i - 1])
    return kama

def compute_slope(series, window):
    return series.rolling(window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if x.isna().sum() == 0 else np.nan,
        raw=False
    )

def kalman_filter(series, R=0.001, Q=1e-5):
    n = len(series)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = series.iloc[0]
    P[0] = 1.0
    for k in range(1, n):
        xhat_minus = xhat[k - 1]
        P_minus = P[k - 1] + Q
        K = P_minus / (P_minus + R)
        xhat[k] = xhat_minus + K * (series.iloc[k] - xhat_minus)
        P[k] = (1 - K) * P_minus
    return pd.Series(xhat, index=series.index)

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = np.maximum.reduce([
        df['high'] - df['low'],
        np.abs(df['high'] - df['close'].shift()),
        np.abs(df['low'] - df['close'].shift())
    ])
    return pd.Series(tr, index=df.index).rolling(period).mean()

# ----------------------------------------------------------------------
# METRIC EVALUATION
# ----------------------------------------------------------------------
def evaluate_trend(name, trend, price, forward_return_horizon=6):
    mae = np.mean(np.abs(trend - price))
    rmse = np.sqrt(np.mean((trend - price) ** 2))
    forward_return = price.pct_change(periods=forward_return_horizon).shift(-forward_return_horizon)
    corr = trend.corr(forward_return)
    print(f"\n{name} Trend Estimator")
    print(f"MAE vs Price: {mae:.6f}")
    print(f"RMSE vs Price: {rmse:.6f}")
    print(f"Correlation with Forward Return: {corr:.4f}")
    return mae, rmse, corr

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    df = load_all_csv(Path("data/eurusd"))
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    df['kama'] = compute_kama(df['close'], CONFIG['KAMA_ER_WINDOW'], CONFIG['KAMA_FAST'], CONFIG['KAMA_SLOW'])
    df['slope'] = compute_slope(df['close'], CONFIG['SLOPE_WINDOW'])
    df['kalman'] = kalman_filter(df['close'], R=CONFIG['KALMAN_R'])
    df['atr'] = atr(df, CONFIG['ATR_PERIOD'])

    df = df.dropna(subset=['kama', 'slope', 'kalman', 'atr'])

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.plot(df['close'], label='Price', alpha=0.5)
    plt.plot(df['kama'], label='KAMA')
    plt.plot(df['kalman'], label='Kalman Filter')
    plt.plot(df['close'] + df['slope'], label='Slope (offset)', linestyle='--')
    plt.title("Trend Estimators vs Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Evaluate trends
    evaluate_trend("KAMA", df['kama'], df['close'])
    evaluate_trend("Slope", df['close'] + df['slope'], df['close'])  # slope is derivative
    evaluate_trend("Kalman", df['kalman'], df['close'])
    evaluate_trend("ATR", df['atr'], df['close'])

if __name__ == "__main__":
    main()

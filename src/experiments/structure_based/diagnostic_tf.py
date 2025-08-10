from pathlib import Path

# Define the output paths
base_path = Path("/mnt/data")
tf_script_path = base_path / "diagnostic_tf.py"
torch_script_path = base_path / "diagnostic_torch.py"

# TensorFlow script content
# diagnostic_tf.py
# TensorFlow-based diagnostic script for EUR/USD ML signal comparison
# Includes: LSTM, Transformer + KAMA, Kalman, Slope, ATR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# ---------------------- CONFIG ----------------------
CONFIG = {
    "LOOKBACK": 12,
    "TRAIN_RATIO": 0.3,
    "ATR_PERIOD": 24,
    "KAMA_ER_WINDOW": 10,
    "KAMA_FAST": 2,
    "KAMA_SLOW": 30,
}

# ---------------------- LOAD CSV ----------------------
def load_all_csv(data_dir: Path, resample_to_hour: bool = True) -> pd.DataFrame:
    column_names = ["datetime", "open", "high", "low", "close", "volume"]
    frames = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, sep=";", header=None, names=column_names, dtype=str)
        df["timestamp"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S", errors="coerce")
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    df_all = pd.concat(frames).drop_duplicates(subset="timestamp").sort_values("timestamp").dropna()
    if resample_to_hour:
        df_all = (
            df_all.set_index("timestamp")
            .resample("1H")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
            .reset_index()
        )
    return df_all[["timestamp", "open", "high", "low", "close"]].set_index("timestamp")

# ---------------------- INDICATORS ----------------------
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
    return series.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)

def compute_kalman_filter(series):
    n = len(series)
    estimate = np.zeros(n)
    error = np.zeros(n)
    estimate[0] = series.iloc[0]
    error[0] = 1.0
    q = 1e-5
    r = 0.01
    for t in range(1, n):
        estimate[t] = estimate[t - 1]
        error[t] = error[t - 1] + q
        k = error[t] / (error[t] + r)
        estimate[t] = estimate[t] + k * (series.iloc[t] - estimate[t])
        error[t] = (1 - k) * error[t]
    return pd.Series(estimate, index=series.index)

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = np.maximum.reduce([
        df["high"] - df["low"],
        np.abs(df["high"] - df["close"].shift()),
        np.abs(df["low"] - df["close"].shift())
    ])
    return pd.Series(tr, index=df.index).rolling(period).mean()

# ---------------------- ML MODEL UTILS ----------------------
def create_lstm_model(input_shape):
    model = models.Sequential()
    model.add(layers.LSTM(64, input_shape=input_shape))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ---------------------- MAIN ----------------------
def main():
    df = load_all_csv(Path("data/eurusd"))
    df = df.dropna()

    # Compute forward return as binary classification label
    df["forward_ret"] = df["close"].shift(-CONFIG["LOOKBACK"]) - df["close"]
    df["label"] = (df["forward_ret"] > 0).astype(int)

    # Indicators
    df["kama"] = compute_kama(df["close"], CONFIG["KAMA_ER_WINDOW"], CONFIG["KAMA_FAST"], CONFIG["KAMA_SLOW"])
    df["slope"] = compute_slope(df["close"], CONFIG["LOOKBACK"])
    df["kalman"] = compute_kalman_filter(df["close"])
    df["atr"] = atr(df, CONFIG["ATR_PERIOD"])

    df = df.dropna()

    # ML Input Setup
    features = []
    for i in range(CONFIG["LOOKBACK"]):
        features.append(df["close"].shift(i))
    X = np.stack([df.shift(i)["close"].values for i in range(CONFIG["LOOKBACK"])], axis=1)[CONFIG["LOOKBACK"]:]
    y = df["label"].values[CONFIG["LOOKBACK"]:]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape((-1, CONFIG["LOOKBACK"], 1))

    split = int(len(X_scaled) * CONFIG["TRAIN_RATIO"])
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM
    model = create_lstm_model((CONFIG["LOOKBACK"], 1))
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    preds = model.predict(X_test).flatten()
    pred_labels = (preds > 0.5).astype(int)

    # Metrics
    result = {
        "Accuracy": accuracy_score(y_test, pred_labels),
        "Precision": precision_score(y_test, pred_labels),
        "Recall": recall_score(y_test, pred_labels),
        "F1": f1_score(y_test, pred_labels),
        "AUC": roc_auc_score(y_test, preds),
        "Correlation": np.corrcoef(preds, df["forward_ret"].values[-len(preds):])[0, 1],
        "MAE": mean_absolute_error(df["close"].values[-len(preds):], preds),
        "RMSE": np.sqrt(mean_squared_error(df["close"].values[-len(preds):], preds))
    }

    print("LSTM Diagnostic Results:")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
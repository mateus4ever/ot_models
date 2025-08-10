import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------
LOOKBACK = 12
TRAIN_RATIO = 0.3
ATR_PERIOD = 24
KAMA_ER_WINDOW = 10
KAMA_FAST = 2
KAMA_SLOW = 30

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
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

# -----------------------------------------
# INDICATORS
# -----------------------------------------
def atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = np.maximum.reduce([
        df['high'] - df['low'],
        np.abs(df['high'] - df['close'].shift()),
        np.abs(df['low'] - df['close'].shift())
    ])
    return pd.Series(tr, index=df.index).rolling(period).mean()

def compute_kama(series, er_window, fast, slow):
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    er = series.diff(er_window).abs() / series.diff().abs().rolling(er_window).sum()
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = pd.Series(index=series.index, dtype=float)
    kama.iloc[er_window] = series.iloc[er_window]
    for i in range(er_window + 1, len(series)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i - 1])
    return kama

def compute_slope(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)

def kalman_filter(price: pd.Series) -> pd.Series:
    n = len(price)
    xhat = np.zeros(n)
    P = np.zeros(n)
    Q = 1e-5
    R = 1e-2
    xhat[0] = price.iloc[0]
    P[0] = 1.0
    for k in range(1, n):
        xhat_minus = xhat[k-1]
        P_minus = P[k-1] + Q
        K = P_minus / (P_minus + R)
        xhat[k] = xhat_minus + K * (price.iloc[k] - xhat_minus)
        P[k] = (1 - K) * P_minus
    return pd.Series(xhat, index=price.index)

# -----------------------------------------
# LABELS AND FEATURES
# -----------------------------------------
def generate_labels(df: pd.DataFrame) -> pd.Series:
    return (df['close'].shift(-1) > df['close']).astype(int).shift(-1)

def create_features(df: pd.DataFrame, lookback: int) -> (np.ndarray, np.ndarray):
    X, y = [], []
    for i in range(lookback, len(df) - 1):
        X.append(df[['close']].iloc[i - lookback:i].values.flatten())
        y.append(df['label'].iloc[i])
    return np.array(X), np.array(y)

# -----------------------------------------
# EVALUATION
# -----------------------------------------
def evaluate(y_true, y_pred, y_prob, fwd_returns):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'CorrWithFwdReturn': np.corrcoef(y_pred, fwd_returns)[0, 1]
    }

# -----------------------------------------
# MAIN
# -----------------------------------------
def main():
    df = load_all_csv(Path("data/eurusd")).dropna()
    df['label'] = generate_labels(df)
    df['kama'] = compute_kama(df['close'], KAMA_ER_WINDOW, KAMA_FAST, KAMA_SLOW)
    df['slope'] = compute_slope(df['close'], LOOKBACK)
    df['kalman'] = kalman_filter(df['close'])
    df['atr'] = atr(df, ATR_PERIOD)
    df['fwd_return'] = df['close'].shift(-1) - df['close']

    df = df.dropna()

    X, y = create_features(df, LOOKBACK)
    cutoff = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]
    fwd_ret = df['fwd_return'].iloc[LOOKBACK+cutoff:LOOKBACK+len(X)]

    results = {}

    # --- Logistic Regression ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logreg = LogisticRegression()
    logreg.fit(X_train_scaled, y_train)
    pred_lr = logreg.predict(X_test_scaled)
    prob_lr = logreg.predict_proba(X_test_scaled)[:, 1]
    results['LogReg'] = evaluate(y_test, pred_lr, prob_lr, fwd_ret)

    # --- SVM ---
    svm = SVC(probability=True)
    svm.fit(X_train_scaled, y_train)
    pred_svm = svm.predict(X_test_scaled)
    prob_svm = svm.predict_proba(X_test_scaled)[:, 1]
    results['SVM'] = evaluate(y_test, pred_svm, prob_svm, fwd_ret)

    # --- LSTM (Keras) ---
    X_train_lstm = X_train.reshape((-1, LOOKBACK, 1))
    X_test_lstm = X_test.reshape((-1, LOOKBACK, 1))
    model_lstm = Sequential()
    model_lstm.add(LSTM(32, input_shape=(LOOKBACK, 1)))
    model_lstm.add(Dense(1, activation='sigmoid'))
    model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_lstm.fit(X_train_lstm, y_train, epochs=5, batch_size=32, verbose=0)
    prob_lstm = model_lstm.predict(X_test_lstm).flatten()
    pred_lstm = (prob_lstm > 0.5).astype(int)
    results['LSTM'] = evaluate(y_test, pred_lstm, prob_lstm, fwd_ret)

    # --- Transformer (PyTorch) ---
    class TransformerClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.encoder = nn.Linear(input_dim, 64)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128),
                num_layers=2
            )
            self.classifier = nn.Linear(64, 1)

        def forward(self, x):
            x = self.encoder(x)
            x = self.transformer(x.unsqueeze(1)).squeeze(1)
            return torch.sigmoid(self.classifier(x)).squeeze()

    torch.manual_seed(0)
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    model = TransformerClassifier(input_dim=X_train.shape[1])
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = loss_fn(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        prob_tf = model(X_test_tensor).numpy()
    pred_tf = (prob_tf > 0.5).astype(int)
    results['Transformer'] = evaluate(y_test, pred_tf, prob_tf, fwd_ret)

    # --- Classic indicators ---
    for name in ['kama', 'kalman', 'slope', 'atr']:
        signal = df[name].diff().iloc[LOOKBACK+cutoff:LOOKBACK+len(X)]
        signal_bin = (signal > 0).astype(int)
        results[name.upper()] = evaluate(y_test, signal_bin, signal_bin, fwd_ret)

    df_result = pd.DataFrame(results).T.round(4)
    print("\n📊 Comparison Table:")
    print(df_result)

if __name__ == "__main__":
    main()

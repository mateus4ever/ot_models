# Restructured diagnostic_torch.py with better organization and LSTM progress tracking
# Key improvements:
# - Added progress tracking for LSTM/Transformer training
# - Better error handling and validation
# - Cleaner code organization with classes
# - More robust data processing
# - Enhanced logging and debugging

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ------------------------
# CONFIG
# ------------------------
CONFIG = {
    "ATR_PERIOD": 24,
    "KAMA_ER_WINDOW": 10,
    "KAMA_FAST": 2,
    "KAMA_SLOW": 30,
    "SLOPE_WINDOW": 24,
    "LOOKBACK": 24,  # Increased from 12 for more context
    "TRAIN_SPLIT": 0.3,
    "LSTM_EPOCHS": 25,  # Reduced for large datasets
    "TRANSFORMER_EPOCHS": 25,
    "LEARNING_RATE": 0.0005,  # Reduced learning rate
    "BATCH_SIZE": 512,  # Increased for better GPU utilization
    "USE_GPU": True,  # Enable GPU acceleration
    "SAMPLE_SIZE": 1000000,  # Compromise: more data but manageable time
    "FEATURE_SELECTION": True,  # Enable feature selection
    "EARLY_STOP_PATIENCE": 10,  # Increased patience
    "PREDICTION_HORIZON": 1,  # How many steps ahead to predict
    "MIN_RETURN_THRESHOLD": 0.0001  # Minimum return to consider as "up" move
}


# ------------------------
# DATA LOADER CLASS
# ------------------------
class DataLoader:
    @staticmethod
    def load_all_csv(data_dir: Path) -> pd.DataFrame:
        """Load and combine all CSV files with better error handling"""
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not found")

        column_names = ["datetime", "open", "high", "low", "close", "volume"]
        frames = []
        csv_files = list(data_dir.glob("*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")

        print(f"Loading {len(csv_files)} CSV files...")

        for csv_path in tqdm(sorted(csv_files), desc="Loading CSV files"):
            try:
                df = pd.read_csv(csv_path, sep=";", header=None, names=column_names, dtype=str)
                df["timestamp"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S", errors="coerce")

                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Basic data validation
                if df.empty or df["timestamp"].isna().all():
                    print(f"Warning: Skipping {csv_path.name} - invalid data")
                    continue

                frames.append(df)
            except Exception as e:
                print(f"Error loading {csv_path.name}: {e}")
                continue

        if not frames:
            raise ValueError("No valid data files could be loaded")

        df_all = pd.concat(frames, ignore_index=True).drop_duplicates(subset="timestamp")
        df_all = df_all.sort_values("timestamp").dropna(subset=["timestamp"])
        df_all = df_all.set_index("timestamp").dropna()

        print(f"Loaded {len(df_all)} records from {df_all.index[0]} to {df_all.index[-1]}")
        return df_all[["open", "high", "low", "close"]]


# ------------------------
# TECHNICAL INDICATORS CLASS
# ------------------------
class TechnicalIndicators:
    @staticmethod
    def compute_kama(series, er_window, fast, slow):
        """Kaufman's Adaptive Moving Average with improved stability"""
        if len(series) < er_window + 1:
            return pd.Series(index=series.index, dtype=float)

        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)

        # Calculate Efficiency Ratio with better handling of edge cases
        change = series.diff(er_window).abs()
        volatility = series.diff().abs().rolling(er_window).sum()

        # Avoid division by zero
        er = change / volatility.replace(0, np.nan)
        er = er.replace([np.inf, -np.inf], 0).fillna(0)

        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama = pd.Series(index=series.index, dtype=float)
        kama.iloc[er_window] = series.iloc[er_window]

        for i in range(er_window + 1, len(series)):
            if pd.notna(sc.iloc[i]) and pd.notna(kama.iloc[i - 1]):
                kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i - 1])

        return kama

    @staticmethod
    def slope(series: pd.Series, window: int) -> pd.Series:
        """Calculate slope with better error handling"""

        def calc_slope(x):
            if len(x) < 2 or x.isna().sum() > 0:
                return np.nan
            try:
                return np.polyfit(range(len(x)), x, 1)[0]
            except:
                return np.nan

        return series.rolling(window).apply(calc_slope, raw=False)

    @staticmethod
    def kalman_filter(series: pd.Series, Q=1e-5, R=0.001) -> pd.Series:
        """Kalman filter with configurable noise parameters"""
        n = len(series)
        if n == 0:
            return pd.Series(dtype=float)

        xhat = np.zeros(n)
        P = np.zeros(n)

        xhat[0] = series.iloc[0] if pd.notna(series.iloc[0]) else 0
        P[0] = 1.0

        for k in range(1, n):
            if pd.isna(series.iloc[k]):
                xhat[k] = xhat[k - 1]
                P[k] = P[k - 1] + Q
                continue

            # Prediction
            xhatminus = xhat[k - 1]
            Pminus = P[k - 1] + Q

            # Update
            K = Pminus / (Pminus + R)
            xhat[k] = xhatminus + K * (series.iloc[k] - xhatminus)
            P[k] = (1 - K) * Pminus

        return pd.Series(xhat, index=series.index)

    @staticmethod
    def atr(df: pd.DataFrame, period: int) -> pd.Series:
        """Average True Range with improved calculation"""
        if len(df) < period:
            return pd.Series(index=df.index, dtype=float)

        high_low = df["high"] - df["low"]
        high_close_prev = np.abs(df["high"] - df["close"].shift())
        low_close_prev = np.abs(df["low"] - df["close"].shift())

        tr = np.maximum.reduce([high_low, high_close_prev, low_close_prev])
        return pd.Series(tr, index=df.index).rolling(period).mean()


# ------------------------
# PYTORCH MODELS
# ------------------------
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,  # Now properly using the actual input size
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, features)
        # Reshape for LSTM: (batch, seq_len=1, features)
        x = x.unsqueeze(1)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Use the last (and only) output
        out = self.dropout(lstm_out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return torch.sigmoid(out)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)

        # Since we're treating this as a single time step, we'll use a simple approach
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Project to d_model dimensions
        x = self.input_projection(x)  # Shape: (batch, d_model)

        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # Shape: (batch, 1, d_model)

        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Feed forward with residual connection
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        # Remove sequence dimension and classify
        x = x.squeeze(1)  # Shape: (batch, d_model)
        out = self.classifier(x)

        return torch.sigmoid(out)


# ------------------------
# TRAINING CLASS WITH PROGRESS TRACKING
# ------------------------
class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train_pytorch_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """Train PyTorch model with progress tracking and validation"""
        print(f"\nTraining {model_name}...")

        # GPU setup
        device = torch.device('cuda' if torch.cuda.is_available() and self.config["USE_GPU"] else 'cpu')
        print(f"Using device: {device}")
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=self.config["LEARNING_RATE"])
        criterion = nn.BCELoss()

        # Convert to tensors and move to device
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)

        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop with progress bar
        epochs = self.config[f"{model_name.upper()}_EPOCHS"]
        progress_bar = tqdm(range(epochs), desc=f"Training {model_name}")

        # Create data loader for batch processing
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=True
        )

        for epoch in progress_bar:
            # Training
            model.train()
            epoch_train_loss = 0
            num_batches = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                train_output = model(batch_X)
                train_loss = criterion(train_output, batch_y)
                train_loss.backward()
                optimizer.step()

                epoch_train_loss += train_loss.item()
                num_batches += 1

            avg_train_loss = epoch_train_loss / num_batches

            # Validation (full batch for simplicity)
            model.eval()
            with torch.no_grad():
                val_output = model(X_test_tensor)
                val_loss = criterion(val_output, y_test_tensor)

            train_losses.append(avg_train_loss)
            val_losses.append(val_loss.item())

            # Update progress bar
            progress_bar.set_postfix({
                'Train Loss': f'{avg_train_loss:.4f}',
                'Val Loss': f'{val_loss.item():.4f}',
                'Best Val': f'{min(val_losses):.4f}'
            })

            # Early stopping with patience
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config["EARLY_STOP_PATIENCE"]:
                print(f"\nEarly stopping at epoch {epoch + 1} (patience: {self.config['EARLY_STOP_PATIENCE']})")
                break

        # Final evaluation
        model.eval()
        with torch.no_grad():
            y_prob = model(X_test_tensor).cpu().detach().numpy().flatten()
            y_pred = (y_prob > 0.5).astype(int)

        return y_pred, y_prob, train_losses, val_losses


# ------------------------
# EVALUATION
# ------------------------
def evaluate_model(y_true, y_pred, y_prob, forward_returns=None):
    """Enhanced model evaluation with regression metrics"""
    try:
        # Classification metrics
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
        }

        # AUC calculation with error handling
        try:
            metrics["AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["AUC"] = 0.5  # Random classifier baseline

        # Regression-style metrics using probabilities as continuous predictions
        if forward_returns is not None:
            # MAE: Mean Absolute Error between probabilities and actual returns
            # We normalize returns to [0,1] range for comparison with probabilities
            normalized_returns = (forward_returns - forward_returns.min()) / (
                        forward_returns.max() - forward_returns.min())
            metrics["MAE"] = np.mean(np.abs(y_prob - normalized_returns))

            # RMSE: Root Mean Square Error
            metrics["RMSE"] = np.sqrt(np.mean((y_prob - normalized_returns) ** 2))

            # Correlation between probabilities and actual forward returns
            metrics["Correlation"] = np.corrcoef(y_prob, forward_returns)[0, 1] if len(np.unique(y_prob)) > 1 else 0.0
        else:
            # Fallback: use binary targets for regression metrics
            metrics["MAE"] = np.mean(np.abs(y_prob - y_true))
            metrics["RMSE"] = np.sqrt(np.mean((y_prob - y_true) ** 2))
            metrics["Correlation"] = np.corrcoef(y_prob, y_true)[0, 1] if len(np.unique(y_prob)) > 1 else 0.0

        return metrics
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {
            "Accuracy": 0, "Precision": 0, "Recall": 0, "F1": 0, "AUC": 0.5,
            "MAE": 0, "RMSE": 0, "Correlation": 0
        }


def make_ml_dataset(df: pd.DataFrame, lookback: int):
    """Create ML dataset with enhanced feature engineering"""
    if len(df) < lookback + 10:
        raise ValueError(f"Insufficient data: need at least {lookback + 10} rows, got {len(df)}")

    df = df.copy()

    # Use prediction horizon from config
    horizon = CONFIG["PREDICTION_HORIZON"]
    df["forward_return"] = (df["close"].shift(-horizon) - df["close"]) / df["close"]

    # More nuanced target: filter out very small moves
    min_threshold = CONFIG["MIN_RETURN_THRESHOLD"]
    significant_moves = np.abs(df["forward_return"]) > min_threshold
    df["target"] = ((df["forward_return"] > min_threshold) & significant_moves).astype(int)

    features = []
    forward_returns = []
    targets = []

    for i in range(lookback, len(df) - horizon):
        # Get price window and technical indicators
        price_window = df["close"].values[i - lookback:i]
        if np.any(np.isnan(price_window)):
            continue

        # Skip if target is not significant move
        if not significant_moves.iloc[i]:
            continue

        # Enhanced feature engineering
        returns = np.diff(price_window) / price_window[:-1]  # Returns
        log_returns = np.diff(np.log(price_window))  # Log returns

        # Technical features
        current_price = price_window[-1]
        sma_5 = np.mean(price_window[-5:]) if len(price_window) >= 5 else current_price
        sma_10 = np.mean(price_window[-10:]) if len(price_window) >= 10 else current_price
        sma_full = np.mean(price_window)

        # Volatility measures
        vol_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0
        vol_10 = np.std(returns[-10:]) if len(returns) >= 10 else 0
        vol_full = np.std(returns)

        # Momentum features
        momentum_5 = (current_price - price_window[-6]) / price_window[-6] if len(price_window) >= 6 else 0
        momentum_10 = (current_price - price_window[-11]) / price_window[-11] if len(price_window) >= 11 else 0

        # Range features
        high_low_ratio = (np.max(price_window) - np.min(price_window)) / current_price
        recent_high = current_price / np.max(price_window[-10:]) if len(price_window) >= 10 else 1
        recent_low = np.min(price_window[-10:]) / current_price if len(price_window) >= 10 else 1

        # Trend features
        trend_short = np.polyfit(range(5), price_window[-5:], 1)[0] if len(price_window) >= 5 else 0
        trend_long = np.polyfit(range(lookback), price_window, 1)[0]

        # Get technical indicators from dataframe
        kama_val = df["kama"].iloc[i] if pd.notna(df["kama"].iloc[i]) else current_price
        slope_val = df["slope"].iloc[i] if pd.notna(df["slope"].iloc[i]) else 0
        kalman_val = df["kalman"].iloc[i] if pd.notna(df["kalman"].iloc[i]) else current_price
        atr_val = df["atr"].iloc[i] if pd.notna(df["atr"].iloc[i]) else 0

        # Combine all features (total: ~33 features)
        feature_vector = np.concatenate([
            returns[-10:],  # Last 10 returns (10 features)
            log_returns[-10:],  # Last 10 log returns (10 features)
            [current_price / sma_5 - 1],  # Price vs short SMA (1)
            [current_price / sma_10 - 1],  # Price vs medium SMA (1)
            [current_price / sma_full - 1],  # Price vs long SMA (1)
            [vol_5, vol_10, vol_full],  # Volatility measures (3)
            [momentum_5, momentum_10],  # Momentum indicators (2)
            [high_low_ratio, recent_high, recent_low],  # Range features (3)
            [trend_short, trend_long],  # Trend slopes (2)
            [current_price / kama_val - 1],  # KAMA ratio (1)
            [slope_val],  # Slope indicator (1)
            [current_price / kalman_val - 1],  # Kalman ratio (1)
            [atr_val / current_price if current_price > 0 else 0],  # Normalized ATR (1)
            # Additional technical features
            [np.mean(returns[-5:]) if len(returns) >= 5 else 0],  # Recent avg return (1)
            [np.max(returns[-5:]) if len(returns) >= 5 else 0],  # Recent max return (1)
            [np.min(returns[-5:]) if len(returns) >= 5 else 0],  # Recent min return (1)
            [len([r for r in returns[-5:] if r > 0]) / 5 if len(returns) >= 5 else 0.5],  # Win rate (1)
            # Price position features
            [(current_price - np.min(price_window)) / (np.max(price_window) - np.min(price_window)) if np.max(
                price_window) != np.min(price_window) else 0.5],  # Price position in range (1)
            [np.mean(price_window[-3:]) / np.mean(price_window[-10:]) - 1 if len(price_window) >= 10 else 0],
            # Short vs medium term SMA (1)
            # Volatility ratios
            [vol_5 / vol_10 if vol_10 > 0 else 1],  # Short vs medium vol ratio (1)
            [vol_10 / vol_full if vol_full > 0 else 1]  # Medium vs long vol ratio (1)
        ])

        features.append(feature_vector)
        forward_returns.append(df["forward_return"].iloc[i])
        targets.append(df["target"].iloc[i])

    if not features:
        raise ValueError("No valid feature windows could be created")

    X = np.array(features)
    forward_returns = np.array(forward_returns)
    y = np.array(targets)

    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)} (class 0: down, class 1: up)")
    print(f"Target balance: {np.mean(y):.3f} (0.5 = balanced)")
    print(f"Average absolute return: {np.mean(np.abs(forward_returns)):.6f}")
    print(f"Filtered out {len(df) - len(features)} insignificant moves")

    return X, y, forward_returns


# ------------------------
# MAIN EXECUTION
# ------------------------
def main():
    """Main execution with comprehensive error handling"""
    try:
        print("=== Financial ML Pipeline Starting ===\n")

        # Load data
        data_path = Path("data/eurusd")
        df = DataLoader.load_all_csv(data_path)
        print(f"Dataset shape: {df.shape}")

        # Optional: Use subset for faster testing
        if CONFIG["SAMPLE_SIZE"] and len(df) > CONFIG["SAMPLE_SIZE"]:
            print(f"Using sample of {CONFIG['SAMPLE_SIZE']} records for faster processing")
            df = df.tail(CONFIG["SAMPLE_SIZE"])  # Use most recent data

        # Calculate technical indicators
        print("\nCalculating technical indicators...")
        indicators = TechnicalIndicators()

        with tqdm(total=4, desc="Computing indicators") as pbar:
            df["kama"] = indicators.compute_kama(df["close"], CONFIG["KAMA_ER_WINDOW"],
                                                 CONFIG["KAMA_FAST"], CONFIG["KAMA_SLOW"])
            pbar.update(1)

            df["slope"] = indicators.slope(df["close"], CONFIG["SLOPE_WINDOW"])
            pbar.update(1)

            df["kalman"] = indicators.kalman_filter(df["close"])
            pbar.update(1)

            df["atr"] = indicators.atr(df, CONFIG["ATR_PERIOD"])
            pbar.update(1)

        # Clean data
        df = df.dropna().copy()
        print(f"Data shape after cleaning: {df.shape}")

        if len(df) < 100:
            raise ValueError("Insufficient data after cleaning")

        df["forward_return"] = df["close"].shift(-1) - df["close"]

        # Evaluate indicators
        print("\n=== Technical Indicator Evaluation ===")
        indicator_names = ["kama", "slope", "kalman", "atr"]
        for ind in indicator_names:
            mae = np.mean(np.abs(df[ind] - df["close"]))
            rmse = np.sqrt(np.mean((df[ind] - df["close"]) ** 2))
            corr = df[ind].corr(df["forward_return"])
            print(f"{ind.upper()}: MAE={mae:.6f}, RMSE={rmse:.6f}, Corr={corr:.4f}")

        # Prepare ML dataset
        print(f"\nPreparing ML dataset with lookback={CONFIG['LOOKBACK']}...")
        X, y, forward_returns = make_ml_dataset(df, CONFIG["LOOKBACK"])
        print(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=CONFIG["TRAIN_SPLIT"], shuffle=False
        )

        # Also split forward returns for evaluation
        _, forward_returns_test = train_test_split(
            forward_returns, train_size=CONFIG["TRAIN_SPLIT"], shuffle=False
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
        print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

        # Train models
        results = {}
        trainer = ModelTrainer(CONFIG)

        # Logistic Regression
        print("\n=== Training Logistic Regression ===")
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        y_pred = lr_model.predict(X_test_scaled)
        y_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
        results["LogisticRegression"] = evaluate_model(y_test, y_pred, y_prob, forward_returns_test)

        # SVM
        print("\n=== Training SVM ===")
        svm_model = SVC(probability=True, random_state=42)
        svm_model.fit(X_train_scaled, y_train)
        y_pred = svm_model.predict(X_test_scaled)
        y_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
        results["SVM"] = evaluate_model(y_test, y_pred, y_prob, forward_returns_test)

        # LSTM
        print("\n=== Training LSTM ===")
        lstm = SimpleLSTM(X_train_scaled.shape[1])  # Use actual feature dimension
        y_pred, y_prob, train_losses, val_losses = trainer.train_pytorch_model(
            lstm, X_train_scaled, y_train, X_test_scaled, y_test, "LSTM"
        )
        results["LSTM"] = evaluate_model(y_test, y_pred, y_prob, forward_returns_test)

        # Transformer
        print("\n=== Training Transformer ===")
        transformer = TransformerModel(X_train_scaled.shape[1])  # Use actual feature dimension
        y_pred, y_prob, train_losses_t, val_losses_t = trainer.train_pytorch_model(
            transformer, X_train_scaled, y_train, X_test_scaled, y_test, "TRANSFORMER"
        )
        results["Transformer"] = evaluate_model(y_test, y_pred, y_prob, forward_returns_test)

        # Results summary
        print("\n=== Model Comparison Results ===")
        df_results = pd.DataFrame(results).T
        df_results = df_results.round(4)
        print(df_results)

        # Find best model
        best_model = df_results['F1'].idxmax()
        print(f"\nBest performing model (by F1): {best_model}")
        print(f"F1 Score: {df_results.loc[best_model, 'F1']:.4f}")

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
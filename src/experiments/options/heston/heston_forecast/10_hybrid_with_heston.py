import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def calculate_log_returns(data):
    return np.log(data['Close'] / data['Close'].shift(1)).dropna()

def calculate_variance_proxy(log_returns, window=3):
    var_proxy = (log_returns ** 2) * 252
    return var_proxy.rolling(window=window).mean().dropna()

def calculate_changes_in_variance(var_proxy):
    vt = var_proxy[:-1].values
    vt_plus_1 = var_proxy[1:].values
    delta_t = 1 / 252
    y = (vt_plus_1 - vt) / delta_t
    return vt, y

def estimate_heston_params(log_returns, var_proxy):
    vt, y = calculate_changes_in_variance(var_proxy)

    if len(vt) < 2 or len(y) < 2:
        raise ValueError("Not enough data points for regression. Increase data length or reduce rolling window size.")

    vt = vt.reshape(-1, 1)
    model = LinearRegression()
    model.fit(vt, y)

    kappa = -model.coef_[0]
    theta = model.intercept_ / kappa if kappa != 0 else np.mean(vt)

    # Rough sigma estimate from changes in volatility proxy
    daily_vol = np.sqrt(log_returns ** 2)
    delta_vol = daily_vol.diff().dropna()
    sigma = np.std(delta_vol) * np.sqrt(252)

    # Rough rho estimate: correlation between returns and changes in volatility proxy
    volatility_proxy = np.abs(log_returns)
    delta_vol_proxy = volatility_proxy.diff().dropna()
    aligned_returns = log_returns[delta_vol_proxy.index]
    rho = np.corrcoef(aligned_returns, delta_vol_proxy)[0, 1]

    print(f"Estimated parameters:\n kappa={kappa:.4f}, theta={theta:.4f}, sigma={sigma:.4f}, rho={rho:.4f}")
    return kappa, theta, sigma, rho

def simulate_heston_paths(S0, v0, kappa, theta, sigma, rho, T, N, M=1000, seed=42):
    np.random.seed(seed)
    dt = T / N
    S = np.zeros((M, N + 1))
    v = np.zeros((M, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    sqrt_dt = np.sqrt(dt)

    for t in range(1, N + 1):
        Z1 = np.random.normal(size=M)
        Z2 = np.random.normal(size=M)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

        v_prev = v[:, t - 1]
        v_next = v_prev + kappa * (theta - v_prev) * dt + sigma * np.sqrt(np.maximum(v_prev, 0)) * sqrt_dt * W2
        v_next = np.maximum(v_next, 0)  # variance must be non-negative
        v[:, t] = v_next

        S_prev = S[:, t - 1]
        S_next = S_prev * np.exp(-0.5 * v_prev * dt + np.sqrt(np.maximum(v_prev, 0)) * sqrt_dt * W1)
        S[:, t] = S_next

    mean_path = S.mean(axis=0)
    return mean_path

def backtest(data, days_ahead=50, window=3):
    log_returns = calculate_log_returns(data)
    var_proxy = calculate_variance_proxy(log_returns, window=window)

    kappa, theta, sigma, rho = estimate_heston_params(log_returns, var_proxy)

    predictions = []
    real_values = []

    # For each day where we have enough future data
    for i in range(len(data) - days_ahead):
        S0 = data['Close'].iloc[i]
        v0 = var_proxy.iloc[i] if i < len(var_proxy) else var_proxy.iloc[-1]

        mean_path = simulate_heston_paths(S0, v0, kappa, theta, sigma, rho, T=days_ahead/252, N=days_ahead, M=500)

        predictions.append(mean_path[-1])  # predict price at day i + days_ahead
        real_values.append(data['Close'].iloc[i + days_ahead])

    predictions = np.array(predictions)
    real_values = np.array(real_values)

    differences = predictions - real_values
    cum_diff = np.cumsum(differences)

    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(predictions)), real_values, label='Real Price')
    plt.plot(range(len(predictions)), predictions, label='Predicted Price (Heston Mean)')
    plt.legend()
    plt.title('Predicted vs Real Price')
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(range(len(differences)), differences, label='Difference (Predicted - Real)')
    plt.legend()
    plt.title('Prediction Error per Day')
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(range(len(cum_diff)), cum_diff, label='Cumulative Difference')
    plt.legend()
    plt.title('Cumulative Prediction Error over Time')
    plt.show()

    # Summary metrics
    mse = np.mean(differences ** 2)
    mae = np.mean(np.abs(differences))
    print(f"Summary:\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nFinal cumulative difference: {cum_diff[-1]:.4f}")

if __name__ == "__main__":
    data = pd.read_csv("tesla_eod.csv", parse_dates=['Datetime'], index_col='Datetime')
    data = data[['Close']].dropna()
    backtest(data, days_ahead=50, window=3)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# --- Data loading and processing ---

def get_clean_financial_data(file_path: str, price_column: str = "Price") -> pd.DataFrame:
    data = pd.read_csv(file_path, parse_dates=True, index_col=0)
    if price_column not in data.columns:
        raise ValueError(f"CSV must contain '{price_column}' column.")
    data = data[[price_column]].ffill()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data.rename(columns={price_column: 'Price'})


def calculate_variance_proxies(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    log_returns = np.log(data['Price'] / data['Price'].shift(1)).dropna()
    variance_proxy = (log_returns ** 2) * 252  # annualized variance proxy
    variance_proxy = variance_proxy.rolling(window=5).mean().dropna()
    return variance_proxy, log_returns


def calculate_changes_in_variance(variance_proxy: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    vt = variance_proxy[:-1].values
    vt_plus_1 = variance_proxy[1:].values
    delta_t = 1 / 252
    y = (vt_plus_1 - vt) / delta_t
    return vt, y


def estimate_heston_parameters(log_returns: pd.Series, variance_proxy: pd.Series) -> tuple[float, float, float, float]:
    vt, y = calculate_changes_in_variance(variance_proxy)
    vt_reshaped = vt.reshape(-1, 1)

    model = LinearRegression()
    model.fit(vt_reshaped, y)
    kappa = -model.coef_[0]
    theta = model.intercept_ / kappa if kappa != 0 else np.mean(vt)

    daily_volatility = np.sqrt(log_returns ** 2)
    delta_volatility = daily_volatility.diff().dropna()
    sigma = np.std(delta_volatility) * np.sqrt(252)

    volatility_proxy = np.abs(log_returns)
    delta_volatility = volatility_proxy.diff().dropna()
    aligned_returns = log_returns[delta_volatility.index]
    rho = np.corrcoef(aligned_returns, delta_volatility)[0, 1]

    return kappa, theta, sigma, rho


def estimate_drift(log_returns: pd.Series) -> float:
    return log_returns.mean() * 252


# --- Heston simulation ---

def heston_simulate(S0, v0, kappa, theta, sigma, rho, mu, T, N, n_paths=100):
    dt = T / N
    S_paths = np.zeros((n_paths, N + 1))
    v_paths = np.zeros((n_paths, N + 1))
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0

    for t in range(1, N + 1):
        z1 = np.random.normal(size=n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=n_paths)

        v_prev = v_paths[:, t - 1]
        v_paths[:, t] = np.abs(v_prev + kappa * (theta - v_prev) * dt + sigma * np.sqrt(v_prev * dt) * z2)

        S_prev = S_paths[:, t - 1]
        S_paths[:, t] = S_prev * np.exp((mu - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * z1)

    return S_paths


# --- Backtesting ---

def backtest_strategy(data, kappa, theta, sigma, rho, mu, threshold_prob=0.6, threshold_move=0.01):
    prices = data['Price'].values
    n = len(prices)
    delta_t = 1 / 252
    capital = 10000
    position = 0  # 0 = no position, 1 = long, -1 = short
    capital_over_time = [capital]

    v0 = np.var(np.diff(np.log(prices))) * 252  # initial variance proxy

    for i in range(n - 11):  # simulate next 10 days from day i
        S0 = prices[i]
        sim_paths = heston_simulate(S0, v0, kappa, theta, sigma, rho, mu, T=10 * delta_t, N=10, n_paths=1000)
        mean_pred = sim_paths[:, -1].mean()
        predicted_return = (mean_pred - S0) / S0

        real_return = (prices[i + 10] - S0) / S0

        # Decide position
        if position == 0:
            if predicted_return > threshold_move and predicted_return > 0:
                position = 1
            elif predicted_return < -threshold_move and predicted_return < 0:
                position = -1
        else:
            # Close position if prediction reverses
            if position == 1 and predicted_return < 0:
                position = 0
            elif position == -1 and predicted_return > 0:
                position = 0

        # Update capital based on real return and position
        capital = capital * (1 + position * real_return)
        capital_over_time.append(capital)

    return capital_over_time


# --- Main ---

def main():
    data = get_clean_financial_data("tesla_eod.csv")
    variance_proxy, log_returns = calculate_variance_proxies(data)
    kappa, theta, sigma, rho = estimate_heston_parameters(log_returns, variance_proxy)
    mu = estimate_drift(log_returns)

    print(f"Estimated Heston parameters:\n"
          f"  kappa: {kappa:.6f}\n"
          f"  theta: {theta:.8f}\n"
          f"  sigma: {sigma:.6f}\n"
          f"  rho: {rho:.6f}\n"
          f"  mu: {mu:.6f}")

    capital_over_time = backtest_strategy(data, kappa, theta, sigma, rho, mu)

    plt.figure(figsize=(10, 6))
    plt.plot(capital_over_time, label='Strategy Capital')
    plt.title('Backtest Strategy Capital Over Time')
    plt.xlabel('Time (days)')
    plt.ylabel('Capital')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

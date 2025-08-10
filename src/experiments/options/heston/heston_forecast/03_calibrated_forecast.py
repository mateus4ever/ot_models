import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def get_clean_financial_data(file_path: str, start_date: str, end_date: str, price_column: str) -> pd.DataFrame:
    data = pd.read_csv(file_path, parse_dates=True, index_col=0)
    if price_column not in data.columns:
        raise ValueError(f"The CSV file must contain a '{price_column}' column.")
    data = data.ffill()
    data.index = pd.to_datetime(data.index)
    data = data.loc[start_date:end_date]
    data = data[[price_column]].rename(columns={price_column: 'Price'})
    return data

def calculate_variance_proxies(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    log_returns = np.log(data['Price'] / data['Price'].shift(1)).dropna()
    variance_proxy = (log_returns**2) * 252  # Annualized variance proxy
    variance_proxy = variance_proxy.rolling(window=5).mean().dropna()  # 5-period rolling mean
    return variance_proxy, log_returns

def calculate_changes_in_variance(variance_proxy: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    vt = variance_proxy[:-1].values
    vt_plus_1 = variance_proxy[1:].values
    delta_t = 1 / 252
    y = (vt_plus_1 - vt) / delta_t
    return vt, y

def estimate_heston_parameters(log_returns: pd.Series, variance_proxy: pd.Series) -> tuple[float, float, float, float, float]:
    vt, y = calculate_changes_in_variance(variance_proxy)
    vt_reshaped = vt.reshape(-1, 1)

    model = LinearRegression()
    model.fit(vt_reshaped, y)
    kappa = -model.coef_[0]
    theta = model.intercept_ / kappa if kappa != 0 else np.mean(vt)

    daily_volatility = np.sqrt(log_returns**2)
    delta_volatility = daily_volatility.diff().dropna()
    sigma = np.std(delta_volatility) * np.sqrt(252)

    volatility_proxy = np.abs(log_returns)
    delta_volatility = volatility_proxy.diff().dropna()
    aligned_returns = log_returns[delta_volatility.index]
    rho = np.corrcoef(aligned_returns, delta_volatility)[0, 1]

    return vt[-1], theta, kappa, sigma, rho

def estimate_drift(log_returns: pd.Series) -> float:
    return log_returns.mean() * 252

def simulate_heston(S0: float, v0: float, theta: float, kappa: float, sigma: float, rho: float,
                    mu: float, T: float = 1, N: int = 252, M: int = 10) -> np.ndarray:
    dt = T / N
    S = np.zeros((N + 1, M))
    v = np.zeros((N + 1, M))
    S[0] = S0
    v[0] = v0

    for t in range(1, N + 1):
        Z1 = np.random.normal(size=M)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(size=M)
        v[t] = np.abs(v[t-1] + kappa * (theta - v[t-1]) * dt + sigma * np.sqrt(v[t-1] * dt) * Z2)
        S[t] = S[t-1] * np.exp((mu - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1] * dt) * Z1)

    return S

def main_rolling_forecast():
    file_path = "2023_2025_usdchf_hourly.csv"
    price_column = "Price"
    start_date = "2023-01-01"
    end_date = "2025-07-04"

    data = get_clean_financial_data(file_path, start_date, end_date, price_column)

    window_size = 30  # rolling window size in days
    forecast_horizon = 10  # forecast horizon in days
    num_simulations = 1000

    all_diffs = []

    for start_idx in range(len(data) - window_size - forecast_horizon):
        calib_data = data.iloc[start_idx : start_idx + window_size]

        variance_proxy, log_returns = calculate_variance_proxies(calib_data)
        if len(variance_proxy) < 5 or len(log_returns) < 5:
            continue

        v0, theta, kappa, sigma, rho = estimate_heston_parameters(log_returns, variance_proxy)
        mu = estimate_drift(log_returns)

        S0 = calib_data['Price'].iloc[-1]

        simulated_paths = simulate_heston(
            S0, v0, theta, kappa, sigma, rho, mu,
            T=forecast_horizon/252, N=forecast_horizon, M=num_simulations
        )

        mean_prediction = simulated_paths[-1].mean()
        real_price = data['Price'].iloc[start_idx + window_size + forecast_horizon - 1]

        diff = mean_prediction - real_price
        all_diffs.append(diff)

    all_diffs = np.array(all_diffs)
    cum_sum_diffs = np.cumsum(all_diffs)

    plt.figure(figsize=(12, 6))
    plt.plot(all_diffs, label="Daily Prediction Difference")
    plt.plot(cum_sum_diffs, label="Cumulative Sum of Differences", linewidth=2)
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Prediction Differences and Their Cumulative Sum Over Time")
    plt.xlabel("Rolling Window Index")
    plt.ylabel("Price Difference")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Mean difference: {np.mean(all_diffs):.6f}")
    print(f"Std deviation difference: {np.std(all_diffs):.6f}")

if __name__ == "__main__":
    main_rolling_forecast()

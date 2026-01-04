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
                    mu: float, T: float = 1, N: int = 252, M: int = 1000) -> np.ndarray:
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

def compare_simulations_vs_real(data: pd.DataFrame, v0: float, theta: float, kappa: float, sigma: float,
                                rho: float, mu: float, days_ahead: int = 50, monte_carlo_paths: int = 500):

    dates = data.index
    prices = data['Price'].values
    n = days_ahead

    predicted_means = []
    real_prices_n_days = []
    valid_dates = []

    for i in range(len(prices) - n):
        S0 = prices[i]
        sim = simulate_heston(S0, v0, theta, kappa, sigma, rho, mu, T=n/252, N=n, M=monte_carlo_paths)
        mean_pred = sim[-1].mean()
        predicted_means.append(mean_pred)
        real_prices_n_days.append(prices[i + n])
        valid_dates.append(dates[i])

    predicted_means = np.array(predicted_means)
    real_prices_n_days = np.array(real_prices_n_days)

    # Plot real vs predicted at horizon
    plt.figure(figsize=(12, 6))
    plt.plot(valid_dates, real_prices_n_days, label='Real Price (n days later)')
    plt.plot(valid_dates, predicted_means, label='Mean Predicted Price (horizon n)', linestyle='--')
    plt.title(f'Real vs Predicted Prices at {n}-Day Horizon')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Difference plot
    diff = predicted_means - real_prices_n_days
    plt.figure(figsize=(12, 4))
    plt.plot(valid_dates, diff, color='red')
    plt.title(f'Difference (Predicted - Real) at {n}-Day Horizon')
    plt.xlabel('Date')
    plt.ylabel('Price Difference')
    plt.grid(True)
    plt.show()

    # Histogram of differences
    plt.figure(figsize=(10, 5))
    plt.hist(diff, bins=30, color='orange', edgecolor='black')
    plt.title(f'Histogram of Differences at {n}-Day Horizon')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Summary statistics
    mean_diff = diff.mean()
    median_diff = np.median(diff)
    std_diff = diff.std()

    print(f"Summary Statistics of Differences (Predicted - Real) at {n}-Day Horizon:")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Median difference: {median_diff:.6f}")
    print(f"  Std deviation: {std_diff:.6f}")

    # Cumulative sum of differences plot (summary)
    cumulative_diff = np.cumsum(diff)
    plt.figure(figsize=(12, 5))
    plt.plot(valid_dates, cumulative_diff, color='purple')
    plt.title(f'Cumulative Sum of Differences (Predicted - Real) over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Difference')
    plt.grid(True)
    plt.show()


def main_forex_with_daily_simulations():
    file_path = "2023_2025_usdchf_hourly.csv"
    price_column = "Price"  # Adjust to your CSV column name
    start_date = "2023-01-01"
    end_date = "2025-07-04"

    data = get_clean_financial_data(file_path, start_date, end_date, price_column)
    variance_proxy, log_returns = calculate_variance_proxies(data)
    v0, theta, kappa, sigma, rho = estimate_heston_parameters(log_returns, variance_proxy)
    mu = estimate_drift(log_returns)

    print(f"Estimated parameters from CSV data:")
    print(f"  kappa: {kappa:.4f}")
    print(f"  theta: {theta:.6f}")
    print(f"  sigma: {sigma:.6f}")
    print(f"  rho: {rho:.4f}")
    print(f"  mu: {mu:.4f} ({mu*100:.2f}%)")

    days_ahead = 50
    monte_carlo_paths = 1000

    compare_simulations_vs_real(data, v0, theta, kappa, sigma, rho, mu, days_ahead, monte_carlo_paths)

if __name__ == "__main__":
    main_forex_with_daily_simulations()

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

def plot_final_price_distribution(simulated_paths: np.ndarray, S0: float, target_return: float = 0.10) -> None:
    final_prices = simulated_paths[-1]
    pct_change = (final_prices - S0) / S0 * 100
    mean_pct = np.mean(pct_change)

    gain_prob = np.mean(final_prices > S0)
    loss_prob = 1 - gain_prob
    target_prob = np.mean(pct_change >= target_return * 100)

    p5 = np.percentile(pct_change, 5)
    p50 = np.percentile(pct_change, 50)
    p95 = np.percentile(pct_change, 95)

    plt.figure(figsize=(10, 6))
    plt.hist(pct_change, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='Break-even (0%)')
    plt.axvline(x=mean_pct, color='green', linestyle='--', label=f'Mean: {mean_pct:.2f}%')
    plt.axvline(p5, color='orange', linestyle='--', label=f'5th %ile: {p5:.2f}%')
    plt.axvline(p50, color='black', linestyle='--', label=f'50th %ile (Median): {p50:.2f}%')
    plt.axvline(p95, color='purple', linestyle='--', label=f'95th %ile: {p95:.2f}%')

    plt.title("Simulated Final Price Distribution (% Change)")
    plt.xlabel("Percentage Change from Initial Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\n📈 Probability of Gain (> 0%): {gain_prob:.2%}")
    print(f"📉 Probability of Loss (<= 0%): {loss_prob:.2%}")
    print(f"🎯 Probability of Return ≥ {target_return*100:.1f}%: {target_prob:.2%}")
    print(f"\n📊 Return Percentile Estimates:")
    print(f"  5th Percentile: {p5:.2f}%")
    print(f"  50th Percentile (Median): {p50:.2f}%")
    print(f"  95th Percentile: {p95:.2f}%")

def main_forex_with_rolling_simulation():
    file_path = "2023_2025_usdchf_hourly.csv"
    price_column = "Price"
    start_date = "2023-01-01"
    end_date = "2025-07-04"
    forecast_days = 50
    N = 50  # steps in simulation for forecast_days
    M = 1000  # simulation paths
    T = forecast_days / 252  # time horizon in years

    data = get_clean_financial_data(file_path, start_date, end_date, price_column)
    variance_proxy, log_returns = calculate_variance_proxies(data)
    v0, theta, kappa, sigma, rho = estimate_heston_parameters(log_returns, variance_proxy)
    mu = estimate_drift(log_returns)

    print(f"Estimated parameters:")
    print(f"  kappa: {kappa:.4f}")
    print(f"  theta: {theta:.6f}")
    print(f"  sigma: {sigma:.6f}")
    print(f"  rho: {rho:.4f}")
    print(f"  mu: {mu:.4f} ({mu*100:.2f}%)")

    # Prepare daily closing prices for comparison (last price of each day)
    daily_close = data['Price'].resample('D').last().dropna()

    predicted_means = []
    real_prices = []

    # For each day where we can forecast 50 days ahead
    for current_date in daily_close.index[:-forecast_days]:
        S0 = daily_close.loc[current_date]
        simulated_paths = simulate_heston(S0, v0, theta, kappa, sigma, rho, mu, T, N, M)
        mean_prediction = np.mean(simulated_paths[-1])

        target_date = current_date + pd.Timedelta(days=forecast_days)
        try:
            real_price = daily_close.loc[target_date]
        except KeyError:
            # fallback: get the next available date after target_date
            next_dates = daily_close.index[daily_close.index > target_date]
            if len(next_dates) > 0:
                real_price = daily_close.loc[next_dates[0]]
            else:
                print(f"No real price data available after {target_date}. Stopping simulation loop.")
                break

        predicted_means.append(mean_prediction)
        real_prices.append(real_price)

    predicted_means = np.array(predicted_means)
    real_prices = np.array(real_prices)

    # Plot predicted vs real prices
    plt.figure(figsize=(12,6))
    plt.plot(daily_close.index[:len(predicted_means)], predicted_means, label='Predicted Mean Price')
    plt.plot(daily_close.index[:len(real_prices)], real_prices, label='Real Price')
    plt.title('Predicted Mean Price vs Real Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot difference
    diff = predicted_means - real_prices
    plt.figure(figsize=(12,6))
    plt.plot(daily_close.index[:len(diff)], diff, label='Difference (Predicted - Real)')
    plt.title('Difference Between Predicted Mean and Real Price')
    plt.xlabel('Date')
    plt.ylabel('Price Difference')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot cumulative difference
    cum_diff = np.cumsum(diff)
    plt.figure(figsize=(12,6))
    plt.plot(daily_close.index[:len(cum_diff)], cum_diff, label='Cumulative Difference')
    plt.title('Cumulative Sum of Differences Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Price Difference')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main_forex_with_rolling_simulation()

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

def main_forex_daily_recalibration():
    file_path = "2023_2025_usdchf_hourly.csv"
    price_column = "Price"
    start_date = "2023-01-01"
    end_date = "2025-07-04"

    forecast_days = 20
    M = 1000  # simulations per forecast
    rolling_window_days = 30

    data = get_clean_financial_data(file_path, start_date, end_date, price_column)

    # Because data is hourly, resample to daily closing price
    daily_data = data['Price'].resample('B').last().dropna()  # Business day close

    errors = []
    dates = []

    # Loop over each day after rolling window + forecast_days to have enough data
    for current_day_idx in range(rolling_window_days, len(daily_data) - forecast_days):
        # Define calibration window
        calib_start = current_day_idx - rolling_window_days
        calib_end = current_day_idx - 1

        calib_prices = daily_data.iloc[calib_start:calib_end + 1]
        variance_proxy, log_returns = calculate_variance_proxies(calib_prices.to_frame())

        # estimate parameters for this window
        v0, theta, kappa, sigma, rho = estimate_heston_parameters(log_returns, variance_proxy)
        mu = estimate_drift(log_returns)

        # Simulate forecast_days forward from current day closing price
        S0 = daily_data.iloc[current_day_idx]
        simulated_paths = simulate_heston(S0, v0, theta, kappa, sigma, rho, mu,
                                          T=forecast_days / 252, N=forecast_days, M=M)

        mean_forecast = np.mean(simulated_paths[-1, :])

        # Actual price forecast_days later
        real_price = daily_data.iloc[current_day_idx + forecast_days]

        # Calculate prediction error
        error = mean_forecast - real_price

        errors.append(error)
        dates.append(daily_data.index[current_day_idx])

    errors = np.array(errors)
    dates = pd.to_datetime(dates)

    # Plot errors over time
    plt.figure(figsize=(12, 6))
    plt.plot(dates, errors, label="Prediction Error (Mean Forecast - Real Price)")
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Heston Model Forecast Errors over Time (Forecast Horizon = {forecast_days} Days)")
    plt.xlabel("Date")
    plt.ylabel("Price Error")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot cumulative error
    plt.figure(figsize=(12, 6))
    plt.plot(dates, np.cumsum(errors), label="Cumulative Prediction Error")
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Cumulative Prediction Error over Time")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Error")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main_forex_daily_recalibration()

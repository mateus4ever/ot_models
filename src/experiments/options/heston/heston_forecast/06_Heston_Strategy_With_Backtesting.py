import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_tesla_eod(filepath="tesla_eod.csv"):
    data = pd.read_csv(filepath, parse_dates=['Datetime'])
    data.set_index('Datetime', inplace=True)
    data.sort_index(inplace=True)
    data['Price'] = data['Close']  # Use Close as Price
    return data

def calculate_variance_proxies(data):
    log_returns = np.log(data['Price'] / data['Price'].shift(1)).dropna()
    variance_proxy = (log_returns ** 2) * 252  # Annualized proxy for variance
    variance_proxy = variance_proxy.rolling(window=5).mean().dropna()  # smooth
    return variance_proxy, log_returns

def calculate_changes_in_variance(variance_proxy):
    vt = variance_proxy[:-1].values
    vt_plus_1 = variance_proxy[1:].values
    delta_t = 1 / 252
    y = (vt_plus_1 - vt) / delta_t
    return vt, y

def estimate_heston_parameters(log_returns, variance_proxy):
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
    delta_volatility2 = volatility_proxy.diff().dropna()
    aligned_returns = log_returns[delta_volatility2.index]
    rho = np.corrcoef(aligned_returns, delta_volatility2)[0, 1]

    return vt[-1], theta, kappa, sigma, rho

def estimate_drift(log_returns):
    return log_returns.mean() * 252

def simulate_heston_variance(v0, theta, kappa, sigma, rho, T=10/252, N=50, M=1000):
    dt = T / N
    v = np.zeros((N + 1, M))
    v[0] = v0

    for t in range(1, N + 1):
        Z1 = np.random.normal(size=M)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(size=M)
        v[t] = np.abs(v[t-1] + kappa * (theta - v[t-1]) * dt + sigma * np.sqrt(v[t-1] * dt) * Z2)

    return v

def moving_average(series, window):
    return series.rolling(window=window).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def backtest_strategy(data, variance_proxy, kappa, theta, sigma, rho, mu, N_days=50, M_paths=1000):
    data = data.copy()
    data['VarianceProxy'] = variance_proxy.reindex(data.index).ffill()
    data['MA20'] = moving_average(data['Price'], 20)
    data['RSI14'] = RSI(data['Price'], 14)

    data['Signal'] = 0
    data['PositionSize'] = 0.0
    data['StrategyReturn'] = 0.0

    dates = data.index

    for i in range(len(data) - N_days):
        current_date = dates[i]
        v0 = data['VarianceProxy'].iloc[i]
        if np.isnan(v0):
            continue

        simulated_variances = simulate_heston_variance(v0, theta, kappa, sigma, rho, T=N_days/252, N=N_days, M=M_paths)
        forecast_volatility = np.sqrt(np.mean(simulated_variances, axis=1))
        vol_forecast = forecast_volatility[-1]

        price = data['Price'].iloc[i]
        ma20 = data['MA20'].iloc[i]
        rsi = data['RSI14'].iloc[i]

        vol_threshold = 0.04
        signal = 1 if (price > ma20) and (rsi < 30) and (vol_forecast < vol_threshold) else 0
        data.at[current_date, 'Signal'] = signal

        position_size = 1 / (vol_forecast + 1e-6)
        position_size = min(position_size, 10)
        data.at[current_date, 'PositionSize'] = position_size if signal == 1 else 0.0

        if i + 1 < len(data):
            ret = np.log(data['Price'].iloc[i + 1] / data['Price'].iloc[i])
            strategy_ret = signal * position_size * ret
            data.at[dates[i + 1], 'StrategyReturn'] = strategy_ret

    data['StrategyCumulative'] = np.exp(data['StrategyReturn'].cumsum()) - 1
    data['BuyHoldReturn'] = np.exp(np.log(data['Price'] / data['Price'].iloc[0]).cumsum()) - 1

    return data

def main():
    data = load_tesla_eod("tesla_eod.csv")

    variance_proxy, log_returns = calculate_variance_proxies(data)

    v0, theta, kappa, sigma, rho = estimate_heston_parameters(log_returns, variance_proxy)
    mu = estimate_drift(log_returns)

    print(f"Estimated Heston parameters:")
    print(f"  kappa: {kappa:.4f}")
    print(f"  theta: {theta:.6f}")
    print(f"  sigma: {sigma:.6f}")
    print(f"  rho: {rho:.4f}")
    print(f"  mu: {mu:.4f} ({mu*100:.2f}%)")

    backtest_results = backtest_strategy(data, variance_proxy, kappa, theta, sigma, rho, mu)

    plt.figure(figsize=(14, 7))
    plt.plot(backtest_results.index, backtest_results['StrategyCumulative'], label='Strategy Cumulative Return')
    plt.plot(backtest_results.index, backtest_results['BuyHoldReturn'], label='Buy and Hold Return')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Tesla EOD Data: Strategy vs Buy and Hold')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

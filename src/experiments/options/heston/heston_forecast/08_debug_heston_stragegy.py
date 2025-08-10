import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("tesla_eod.csv", parse_dates=["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)
df["Close"] = df["Close"].astype(float)

# Parameters
initial_capital = 10000
position_size = 0.02  # 2%
forecast_horizon = 10  # days
prediction_threshold = 0.01  # 1%
confidence_threshold = 0.6  # 60%

# Dummy Heston forecast generator
def simulate_heston_forecast(current_price, horizon=10, simulations=1000):
    # Random walk as placeholder
    returns = np.random.normal(loc=0, scale=0.02, size=(simulations, horizon))
    paths = current_price * np.exp(np.cumsum(returns, axis=1))
    return paths

# Backtest variables
capital = initial_capital
position = 0  # number of shares
entry_price = 0
portfolio_values = []
log = []

# Backtest loop
for i in range(len(df) - forecast_horizon):
    date = df.loc[i, "Datetime"]
    price = df.loc[i, "Close"]

    # Forecast future prices
    forecast_paths = simulate_heston_forecast(price, forecast_horizon)
    final_prices = forecast_paths[:, -1]
    mean_forecast = final_prices.mean()
    prob_up = np.mean(final_prices >= price * (1 + prediction_threshold))
    prob_down = np.mean(final_prices <= price * (1 - prediction_threshold))

    # Entry logic
    action = "HOLD"
    if position == 0 and prob_up >= confidence_threshold:
        shares_to_buy = (capital * position_size) // price
        capital -= shares_to_buy * price
        position = shares_to_buy
        entry_price = price
        action = "BUY"

    # Exit logic
    elif position > 0 and prob_down >= confidence_threshold:
        capital += position * price
        position = 0
        action = "SELL"

    # Record portfolio value
    portfolio_value = capital + position * price
    portfolio_values.append(portfolio_value)

    # Log for debugging
    log.append({
        "Date": date,
        "Price": price,
        "Action": action,
        "Position": position,
        "Cash": capital,
        "Portfolio": portfolio_value,
        "MeanForecast": mean_forecast,
        "ProbUp": prob_up,
        "ProbDown": prob_down,
    })

# Convert log to DataFrame
log_df = pd.DataFrame(log)

# Plot portfolio value
plt.figure(figsize=(12, 6))
plt.plot(log_df["Date"], log_df["Portfolio"], label="Portfolio Value")
plt.title("Backtest Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("USD")
plt.legend()
plt.grid()
plt.show()

# Show sample logs for inspection
print(log_df[["Date", "Action", "Price", "ProbUp", "ProbDown", "Position", "Cash", "Portfolio"]].tail(20))

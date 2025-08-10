import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model

# Step 1: Fetch Apple stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2025-01-05')

# Step 2: Calculate daily returns
data['Return'] = data['Adj Close'].pct_change() * 100  # Convert to percentage
returns = data['Return'].dropna()

# Step 3: Fit an EGARCH model
model = arch_model(returns, vol='EGARCH', p=1, q=1, mean='constant', dist='normal')
results = model.fit(disp='off')

# Step 4: Forecast variance using EGARCH
split_date = '2024-08-01'
forecast = results.forecast(horizon=1, start=split_date)

# Extract forecasted variance and align with the index
forecast_variance = forecast.variance.loc[split_date:]
predicted_volatility = np.sqrt(forecast_variance.values.flatten())

# Step 5: Annualize Volatilities
trading_days = 252  # Approximate number of trading days in a year
realized_annualized_volatility = returns.rolling(30).std() * np.sqrt(trading_days)
predicted_annualized_volatility = predicted_volatility * np.sqrt(trading_days)

# Step 6: Plot annualized volatilities
sns.set()
plt.figure(figsize=(20, 10))

# Plot realized annualized volatility
plt.plot(realized_annualized_volatility.index, realized_annualized_volatility,
         label='Realized Annualized Volatility (30-day Rolling)', color='blue')

# Plot EGARCH predicted annualized volatility
plt.plot(forecast_variance.index, predicted_annualized_volatility,
         label='EGARCH Predicted Annualized Volatility', color='red', linestyle='--')

# Add labels and title
plt.title("Realized vs EGARCH Predicted Annualized Volatility", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Annualized Volatility (%)", fontsize=12)
plt.legend(loc='best', fontsize=12)
plt.grid(True)

plt.show()
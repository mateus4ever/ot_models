# momentum_strategy.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("tesla_eod.csv", parse_dates=['Datetime'])
df.set_index('Datetime', inplace=True)
df['Return'] = df['Close'].pct_change()

# Momentum indicators
df['Momentum_5'] = df['Close'].pct_change(5)
df['Momentum_20'] = df['Close'].pct_change(20)
df['Volatility'] = df['Close'].rolling(10).std()

# Define trading signals
momentum_thresh = 0.02
vol_thresh = df['Volatility'].median()
df['Signal'] = 0
df.loc[(df['Momentum_5'] > momentum_thresh) & (df['Volatility'] < vol_thresh), 'Signal'] = 1
df['Position'] = df['Signal'].shift(1)

# Backtest
df['Strategy_Return'] = df['Position'] * df['Return']
df[['Return', 'Strategy_Return']].cumsum().plot(title="Momentum Strategy vs Buy and Hold")
plt.grid(True)
plt.show()
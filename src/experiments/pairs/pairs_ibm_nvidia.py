import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from termcolor import colored as cl
from math import floor
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator


plt.rcParams['figure.figsize'] = (20,10)
plt.style.use('fivethirtyeight')

# EXTRACTING STOCK DATA

""""
def get_historical_data(symbol, start_date, end_date):
    api_key = 'cd5c3e9701464658a921f29685fe2757'
    api_url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api_key}'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df['values']).iloc[::-1].set_index('datetime').astype(float)
    df = df[df.index >= start_date]
    df = df[df.index <= end_date]
    df.index = pd.to_datetime(df.index)
    return df

fb = get_historical_data('META', '2020-01-01', '2025-05-20')
amzn = get_historical_data('AMZN', '2020-01-01', '2025-05-20')
aapl = get_historical_data('AAPL', '2020-01-01', '2025-05-20')
nflx = get_historical_data('NFLX', '2020-01-01', '2025-05-20')
googl = get_historical_data('GOOGL', '2020-01-01', '2025-05-20')
"""

ibm = pd.read_csv("ibm.csv", parse_dates=["datetime"], index_col="datetime")
ibm.index = pd.to_datetime(ibm.index)

nvda = pd.read_csv("nvda.csv", parse_dates=["datetime"], index_col="datetime")
nvda.index = pd.to_datetime(nvda.index)

print(nvda.head())
print(nvda.index)

df = pd.DataFrame(columns = ['nvda','ibm'])
df.nvda = nvda.close
df.ibm = ibm.close
df.index = nvda.index
print(df.tail(10))

beta = np.polyfit(df.ibm, df.nvda, 1)[0]
spread = df.nvda - beta * df.ibm

df['spread'] = spread
df['adx'] = ADXIndicator(high=spread, low=spread, close=spread, window=14).adx()
df['rsi'] = RSIIndicator(close=spread, window=14).rsi()

df = df.dropna()
print(df.tail())


def implement_pairs_trading_strategy(df, investment):
    in_position = False
    equity = investment
    nvda_shares = 0
    ibm_shares = 0

    for i in range(1, len(df)):
        if df['rsi'][i] < 30 and 20 < df['adx'][i] < 25 and not in_position:
            nvda_allocation = equity * 0.5
            ibm_allocation = equity * 0.5

            nvda_shares = floor(nvda_allocation / df['nvda'][i])
            equity -= nvda_shares * df['nvda'][i]

            ibm_shares = floor(ibm_allocation / df['ibm'][i])
            equity += ibm_shares * df['ibm'][i]  # short IBM

            in_position = True
            print(cl('ENTER MARKET:', color='green', attrs=['bold']),
                  f'Bought {nvda_shares} NVDA shares at ${df["nvda"][i]}, '
                  f'Sold {ibm_shares} IBM shares at ${df["ibm"][i]} on {df.index[i]}')

        elif df['rsi'][i] > 70 and 20 < df['adx'][i] < 25 and in_position:
            equity += nvda_shares * df['nvda'][i]
            nvda_shares = 0

            equity -= ibm_shares * df['ibm'][i]
            ibm_shares = 0

            in_position = False
            print(cl('EXIT MARKET:', color='red', attrs=['bold']),
                  f'Sold NVDA and Bought IBM on {df.index[i]} at NVDA=${df["nvda"][i]}, IBM=${df["ibm"][i]}')

    # Closing any remaining positions at the end
    if in_position:
        equity += nvda_shares * df['nvda'].iloc[-1]
        equity -= ibm_shares * df['ibm'].iloc[-1]
        print(cl(f'\nClosing positions at NVDA=${df["nvda"].iloc[-1]}, '
                 f'IBM=${df["ibm"].iloc[-1]} on {df.index[-1]}', attrs=['bold']))

    # Calculating earnings and ROI
    earning = round(equity - investment, 2)
    roi = round((earning / investment) * 100, 2)

    print('')
    print(cl('PAIRS TRADING BACKTESTING RESULTS:', attrs=['bold']))
    print(cl(f'EARNING: ${earning} ; ROI: {roi}%', attrs=['bold']))


investment = 100000
implement_pairs_trading_strategy(df, investment)


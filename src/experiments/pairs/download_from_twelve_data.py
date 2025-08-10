import requests
import pandas as pd
from datetime import datetime, timedelta

tickers = ["UPRO"]
api_key = "KcBuukngrqHUFDpAz5Rb2zSS1djvpmpI"
end_date = datetime.today()
start_date = end_date - timedelta(days=730)

def get_historical_data(symbol, start_date, end_date):
    api_key = 'cd5c3e9701464658a921f29685fe2757'
    api_url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api_key}'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df['values']).iloc[::-1].set_index('datetime').astype(float)
    df = df[df.index >= start_date]
    df = df[df.index <= end_date]
    df.index = pd.to_datetime(df.index)  # Good to ensure it's datetime
    df.to_csv(f"{symbol.lower()}.csv", index=True)
    return df

fb = get_historical_data('META', '2020-01-01', '2025-05-20')
amzn = get_historical_data('AMZN', '2020-01-01', '2025-05-20')
aapl = get_historical_data('AAPL', '2020-01-01', '2025-05-20')
nflx = get_historical_data('NFLX', '2020-01-01', '2025-05-20')
googl = get_historical_data('GOOGL', '2020-01-01', '2025-05-20')

fb_rets, fb_rets.name = fb['close'] / fb['close'].iloc[0], 'fb'
amzn_rets, amzn_rets.name = amzn['close'] / amzn['close'].iloc[0], 'amzn'
aapl_rets, aapl_rets.name = aapl['close'] / aapl['close'].iloc[0], 'aapl'
nflx_rets, nflx_rets.name = nflx['close'] / nflx['close'].iloc[0], 'nflx'
googl_rets, googl_rets.name = googl['close'] / googl['close'].iloc[0], 'googl'


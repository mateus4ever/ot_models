import requests
import pandas as pd
from datetime import datetime, timedelta

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

nvda = get_historical_data('SPY', '2020-01-01', '2025-05-20')
ibm = get_historical_data('QQQ', '2020-01-01', '2025-05-20')



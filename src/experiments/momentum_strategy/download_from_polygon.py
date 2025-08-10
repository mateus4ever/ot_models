import requests
import pandas as pd
from datetime import datetime, timedelta

tickers = ["UPRO"]
api_key = "KcBuukngrqHUFDpAz5Rb2zSS1djvpmpI"
end_date = datetime.today()
start_date = end_date - timedelta(days=730)

def download_stock(ticker):
    print(f"Downloading {ticker}...")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date:%Y-%m-%d}/{end_date:%Y-%m-%d}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key
    }
    r = requests.get(url, params=params)
    d = r.json()
    if "results" not in d:
        print(f"❌ Error downloading {ticker}: {d.get('message', 'Unknown error')}")
        return
    df = pd.DataFrame(d["results"])
    df["Datetime"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={"c": "Close", "o": "Open", "h": "High", "l": "Low", "v": "Volume"})
    df = df[["Datetime", "Close", "Open", "High", "Low", "Volume"]]
    df.to_csv(f"{ticker.lower()}_eod.csv", index=False)
    print(f"✅ Saved {ticker.lower()}_eod.csv")

for ticker in tickers:
    download_stock(ticker)

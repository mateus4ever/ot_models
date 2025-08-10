import requests
import pandas as pd
from datetime import datetime
from time import sleep

# Replace with your own API key
API_KEY = "YOUR_API_KEY_HERE"
SYMBOL = "TSLA"

# Define yearly date ranges (5 years)
YEARS = [(2020, 2020), (2021, 2021), (2022, 2022), (2023, 2023), (2024, 2024)]

def fetch_year(start_year, end_year):
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    url = f"https://api.polygon.io/v2/aggs/ticker/{SYMBOL}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": 	"KcBuukngrqHUFDpAz5Rb2zSS1djvpmpI"
    }
    print(f"Fetching {start_year}...")
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error {response.status_code} for {start_year}: {response.text}")
        return pd.DataFrame()
    data = response.json()
    if "results" not in data:
        print(f"No data for {start_year}")
        return pd.DataFrame()
    df = pd.DataFrame(data["results"])
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    return df

# Download and merge all years
all_data = []
for start, end in YEARS:
    df = fetch_year(start, end)
    all_data.append(df)
    sleep(1.1)  # Avoid rate limits

full_df = pd.concat(all_data).reset_index(drop=True)

# Save to file
csv_filename = "tesla_5y_eod.csv"
full_df.to_csv(csv_filename, index=False)
print(f"\n✅ Saved full 5 years of TSLA data to {csv_filename}")

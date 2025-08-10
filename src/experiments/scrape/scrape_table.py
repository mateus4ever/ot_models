import requests
import pandas as pd
from bs4 import BeautifulSoup

# URL of the page
url = "https://www.interactivebrokers.co.uk/portal/?loginType=2&action=ACCT_MGMT_MAIN&clt=0&RL=1#/quote/265598/fundamentals/ratings?u=false&wb=0&exchange=NASDAQ&SESSIONID=67a98be5.0000000b&impact_settings=true&supportsLeaf=true&widgets=profile,ratings,financials,key_ratios,analyst_forecast,ownership,dividends,competitors,events,news,esg,social_sentiment,securities_lending,equity_mstar,sv,short_sale,ukuser"  # Replace with actual URL

# Fetch page content
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive"
}


response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Find the container div
container = soup.find("div", class_="analyst-coverage-list-wrapper")

# Locate the table inside
table = container.find("table") if container else None

if table:
    # Extract data
    data = []
    for row in table.find_all("tr"):
        cols = row.find_all(["td", "th"])  # Include headers
        cols = [col.text.strip() for col in cols]
        data.append(cols)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(df)

    # Convert to a Python array
    array_data = df.values.tolist()
    print(array_data)
else:
    print("Table not found.")

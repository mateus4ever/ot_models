from playwright.sync_api import sync_playwright
import pandas as pd
from bs4 import BeautifulSoup

url = "https://www.interactivebrokers.co.uk/portal/?loginType=2&action=ACCT_MGMT_MAIN&clt=0&RL=1#/quote/265598/fundamentals/ratings?u=false&wb=0&exchange=NASDAQ&SESSIONID=67a98be5.0000000b&impact_settings=true&supportsLeaf=true&widgets=profile,ratings,financials,key_ratios,analyst_forecast,ownership,dividends,competitors,events,news,esg,social_sentiment,securities_lending,equity_mstar,sv,short_sale,ukuser"  # Replace with actual URL

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # Use False to see actions, True for speed
    page = browser.new_page()
    page.goto(url, timeout=60000)  # Load page with timeout

    # Wait for iframe to load
    page.wait_for_selector("iframe")

    # Locate the iframe by its 'src' attribute
    iframe_element = page.frame_locator("iframe[src*='tipranks.com']")

    # Wait for content inside iframe
    iframe_element.wait_for_selector("table", timeout=15000)  # Wait for table

    # Extract table HTML from iframe
    table_html = iframe_element.inner_html("table")

    browser.close()

# Parse the extracted HTML with BeautifulSoup
soup = BeautifulSoup(table_html, "html.parser")
data = [[cell.text.strip() for cell in row.find_all(["td", "th"])] for row in soup.find_all("tr")]

# Convert to DataFrame
df = pd.DataFrame(data)
print(df)

# Convert to a list if needed
array_data = df.values.tolist()
print(array_data)
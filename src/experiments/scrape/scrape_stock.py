import requests
from bs4 import BeautifulSoup
import time
import random


def scrape_tipranks_data(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive"
    }

    max_retries = 5  # Retry up to 5 times
    retry_delay = 5  # Delay between retries (seconds)

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} to scrape the page...")

            response = requests.get(url, headers=headers)

            # If request is blocked (403, 503), retry
            if response.status_code in [403, 503]:
                print(f"Received status code {response.status_code}. Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay + random.uniform(1, 3))  # Random delay to avoid detection
                continue

            response.raise_for_status()  # Raise an error for other bad responses

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the success rate from the <text> element
            success_rate_element = soup.find('text', {
                'class': 'override fontSize6small colorgreen fontWeightsemibold'
            })
            success_rate = success_rate_element.text.strip() if success_rate_element else 'Not Found'

            # Find the average return from its specific class
            average_return_element = soup.find('div', class_='analyst-card-average-return')
            average_return = average_return_element.get_text(strip=True) if average_return_element else 'Not Found'

            return {
                'Success Rate': success_rate,
                'Average Return': average_return
            }

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            time.sleep(retry_delay)  # Wait before retrying

    print("Failed to retrieve the page after multiple attempts.")
    return None


# URL of the page to scrape
url = "https://www.tipranks.com/experts/analysts/brian-white?ref=ib"

# Scrape the data
data = scrape_tipranks_data(url)

if data:
    print("Scraped Data:")
    print(f"Success Rate: {data['Success Rate']}")
    print(f"Average Return: {data['Average Return']}")
else:
    print("Failed to scrape the data.")
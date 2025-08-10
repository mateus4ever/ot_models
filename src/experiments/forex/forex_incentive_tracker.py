"""
forex_incentive_tracker.py
--------------------------------------------------
Semi-automated helper for the Incentive-Based Forex
Strategy with forward 10-day return analytics.

Usage
-----
• Live check   : `python forex_incentive_tracker.py --once`
• Back-test    : `python forex_incentive_tracker.py --backtest`
• Scheduler    : `python forex_incentive_tracker.py` (weekdays 18:00 UTC)

Dependencies
------------
pip install pandas requests pyyaml beautifulsoup4 \
            apscheduler python-dateutil python-dotenv

Create .env:
FRED_API_KEY=your_key_here
"""

import csv
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Generator, Optional

import requests
import yaml
import pandas as pd
from bs4 import BeautifulSoup
from apscheduler.schedulers.blocking import BlockingScheduler
from dateutil import parser as dtparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "forex_strategy_tracker.csv"
CONFIG_FILE = BASE_DIR / "config.yaml"

FIELDNAMES = [
    "Date Logged",
    "Currency Pair",
    "Observed Distortion",
    "Incentive Conflict",
    "Institution Involved",
    "Bias (Long/Short)",
    "Entry Planned?",
    "Entry Date",
    "Exit Criteria",
    "Actual Outcome",
    "Fwd 10d Return",
    "Notes",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        logging.error("config.yaml missing – create it first.")
        sys.exit(1)
    with CONFIG_FILE.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def fred_get_series(series_id: str, start: str = "2000-01-01") -> pd.Series:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY missing in environment variables")
    url = (
        "https://api.stlouisfed.org/fred/series/observations?"
        f"series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start}"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json().get("observations", [])
    dates = [dtparse.parse(obs["date"]).date() for obs in data]
    vals = [float(obs["value"]) if obs["value"] != "." else None for obs in data]
    return pd.Series(vals, index=pd.to_datetime(dates), name=series_id).dropna()

def fetch_rss_items(url: str, max_items: int = 3) -> Generator[tuple[str, Optional[datetime]], None, None]:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except requests.RequestException as exc:
        logging.warning("RSS fetch failed for %s – %s", url, exc)
        return
    soup = BeautifulSoup(r.content, "xml")
    for item in soup.find_all("item")[:max_items]:
        title = item.title.text.strip() if item.title else "(no title)"
        pub_date = dtparse.parse(item.pubDate.text.strip()) if item.pubDate else None
        yield title, pub_date

def safe_forward_return(price_series: Optional[pd.Series], date: pd.Timestamp, days: int = 10):
    if price_series is None or date not in price_series.index:
        return ""
    p0 = price_series.loc[date]
    future_idx = price_series.index[price_series.index >= date + timedelta(days=days)]
    if future_idx.empty:
        return ""
    p1 = price_series.loc[future_idx[0]]
    return round((p1 - p0) / p0 * 100, 2)

def append_trade_log(row: Dict[str, Any]):
    for k in FIELDNAMES:
        row.setdefault(k, "")
    new_file = not DATA_FILE.exists()
    with DATA_FILE.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    logging.info("Logged %s (%s) – %s", row["Currency Pair"], row["Entry Date"], row["Observed Distortion"][:60])
def distortion_triggered(infl, rate, infl_th: float, rate_th: float):
    return (infl > infl_th) & (rate < rate_th)

# ---------------------------------------------------------------------------
# Live evaluation
# ---------------------------------------------------------------------------

def evaluate_pairs():
    cfg = load_config()
    for pair, pcfg in cfg["pairs"].items():
        try:
            infl = fred_get_series(pcfg["inflation_series"]).iloc[-1]
            rate = fred_get_series(pcfg["rate_series"]).iloc[-1]
            if not distortion_triggered(infl, rate, pcfg.get("infl_threshold", 3), pcfg.get("rate_threshold", 1)):
                logging.info("No distortion for %s", pair)
                continue
            headline, h_date = next(fetch_rss_items(pcfg["rss"], 1), ("n/a", None))
            row = {
                "Date Logged": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                "Currency Pair": pair,
                "Observed Distortion": (
                    f"Infl {infl:.2f}>{pcfg['infl_threshold']} & Rate {rate:.2f}<{pcfg['rate_threshold']}"
                ),
                "Incentive Conflict": pcfg.get("incentive_summary", "See notes"),
                "Institution Involved": pcfg.get("institution", "Unknown"),
                "Bias (Long/Short)": pcfg.get("bias", "Review"),
                "Entry Planned?": "No",
                "Entry Date": datetime.utcnow().strftime("%Y-%m-%d"),
                "Exit Criteria": "Infl/Rates converge or thesis invalid",
                "Notes": f"Latest CB headline: {headline} ({h_date.date() if h_date else 'n/a'})",
            }
            append_trade_log(row)
        except Exception as exc:
            logging.error("Live check failed for %s: %s", pair, exc, exc_info=True)

# ---------------------------------------------------------------------------
# Historical back-test
# ---------------------------------------------------------------------------

def backtest(years: int = 2):
    cfg = load_config()
    start_date = (datetime.utcnow().date().replace(day=1) - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    for pair, pcfg in cfg["pairs"].items():
        try:
            infl_s = fred_get_series(pcfg["inflation_series"], start=start_date)
            rate_s = fred_get_series(pcfg["rate_series"], start=start_date)
            df = pd.DataFrame({"infl": infl_s, "rate": rate_s}).dropna()
            mask = distortion_triggered(df["infl"], df["rate"], pcfg.get("infl_threshold", 3), pcfg.get("rate_threshold", 1))
            triggers = df[mask]
            logging.info("Back-test: %d triggers for %s", len(triggers), pair)
            price_s = fred_get_series(pcfg["price_series"], start=start_date) if "price_series" in pcfg else None
            for dt, vals in triggers.iterrows():
                fwd_ret = safe_forward_return(price_s, dt) if price_s is not None else ""
                row = {
                    "Date Logged": dt.strftime("%Y-%m-%d %H:%M"),
                    "Currency Pair": pair,
                    "Observed Distortion": (
                        f"[Hist] Infl {vals['infl']:.2f}>{pcfg['infl_threshold']} & Rate {vals['rate']:.2f}<{pcfg['rate_threshold']}"
                    ),
                    "Incentive Conflict": pcfg.get("incentive_summary", "See notes"),
                    "Institution Involved": pcfg.get("institution", "Unknown"),
                    "Bias (Long/Short)": pcfg.get("bias", "Review"),
                    "Entry Planned?": "n/a",
                    "Entry Date": dt.strftime("%Y-%m-%d"),
                    "Exit Criteria": "n/a (historical)",
                    "Actual Outcome": "",
                    "Fwd 10d Return": fwd_ret,
                    "Notes": "Back-test trigger",
                }
                append_trade_log(row)
        except Exception as exc:
            logging.error("Back-test failed for %s: %s", pair, exc, exc_info=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--backtest" in sys.argv:
        backtest()
        sys.exit(0)

    if "--once" in sys.argv:
        evaluate_pairs()
        sys.exit(0)

    sched = BlockingScheduler()
    sched.add_job(evaluate_pairs, "cron", day_of_week="mon-fri", hour=18, minute=0)
    logging.info("Scheduler started – weekdays 18:00 UTC")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        pass

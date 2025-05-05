import os
import requests
from datetime import datetime

# Directory to save data
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# List of (url, filename) pairs to download
DATA_SOURCES = [
    ("https://d2v1fiwobg9w6.cloudfront.net/daily_notional_liquidated_by_coin", "daily_notional_liquidated_by_coin.json"),
    ("https://d2v1fiwobg9w6.cloudfront.net/daily_inflow", "daily_inflow.json"),
    ("https://d2v1fiwobg9w6.cloudfront.net/liquidity_by_coin", "liquidity_by_coin.json"),
    ("https://d2v1fiwobg9w6.cloudfront.net/asset_ctxs", "asset_ctxs.json"),
    ("https://d2v1fiwobg9w6.cloudfront.net/hlp_positions", "hlp_positions.json"),
]

def fetch_and_save(url, filename):
    print(f"Fetching {url} ...")
    resp = requests.get(url)
    resp.raise_for_status()
    out_path = os.path.join(DATA_DIR, filename)
    with open(out_path, "w") as f:
        f.write(resp.text)
    print(f"Saved to {out_path}")

def main():
    print(f"--- HLP Data Fetcher: {datetime.now()} ---")
    for url, filename in DATA_SOURCES:
        try:
            fetch_and_save(url, filename)
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")

if __name__ == "__main__":
    main() 
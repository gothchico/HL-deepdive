from datetime import datetime
from collections import defaultdict
import json

# Load chart_data from data/daily_usd_volume_by_coin.json
with open("data/daily_usd_volume_by_coin.json", "r") as file:
    data = json.load(file)
    
    # Extract chart_data from the JSON structure
    chart_data = data.get("chart_data", [])

def get_earliest_times(chart_data):
    earliest_dates = {}
    for record in chart_data:
        coin = record["coin"]
        time = datetime.strptime(record["time"], "%Y-%m-%dT%H:%M:%S")
        if coin not in earliest_dates or time < earliest_dates[coin]:
            earliest_dates[coin] = time
    # Convert datetime objects back to strings
    return {coin: time.strftime("%Y-%m-%d") for coin, time in earliest_dates.items()}

if __name__ == "__main__":
    result = get_earliest_times(chart_data)
    
    # Save the result to a file
    with open("data/coin_listing_dates.json", "w") as outfile:
        json.dump(result, outfile, indent=4)
    
    # Print confirmation
    print("Coin listing dates have been saved to data/coin_listing_dates.json")

import requests
import pandas as pd

# Endpoint and headers
url = "https://api.hyperliquid.xyz/info"
headers = {
    "Content-Type": "application/json"
}

# Properly structured request payload
payload = {
    "type": "metaAndAssetCtxs"
}

# Send request
response = requests.post(url, headers=headers, json=payload)
if response.status_code != 200:
    raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")

# Parse response
meta, ctxs = response.json()

# Merge meta (universe) and assetCtxs using zip
combined_data = [meta_i | ctx_i for meta_i, ctx_i in zip(meta["universe"], ctxs)]

# Convert to DataFrame
df = pd.json_normalize(combined_data)
available_coins = pd.unique(df["name"])

# Save available coins to a txt file
with open("available_coins.txt", "w") as f:
    for coin in available_coins:
        f.write(f"{coin}\n")

# Save to CSV
df.to_csv("hyperliquid_universe_data.csv", index=False)
print("Data saved to hyperliquid_universe_data.csv")

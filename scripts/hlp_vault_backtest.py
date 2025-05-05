import pandas as pd
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import requests
import os

# --- CONFIG ---
apply_rebalance = True  # Toggle this to enable/disable inflow/outflow rebalancing
slippage_factor = 0.1   # 10% slippage per full liquidity taken

# --- LOAD DATA ---
def load_json_df(path, key=None):
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Handle nested coin data structure
    if isinstance(data, dict) and not key and any(isinstance(v, list) for v in data.values()):
        # Flatten the nested coin structure
        flattened_data = []
        for coin, records in data.items():
            for record in records:
                record['coin'] = coin
                flattened_data.append(record)
        return pd.DataFrame(flattened_data)
    
    # Handle regular flat data structure
    if key:
        return pd.DataFrame(data[key])
    return pd.DataFrame(data)

positions = load_json_df("data/hlp_positions.json", key="chart_data")
positions["time"] = pd.to_datetime(positions["time"])

prices = load_json_df("data/asset_ctxs.json", key="chart_data")
prices["time"] = pd.to_datetime(prices["time"])
prices = prices.pivot(index="time", columns="coin", values="avg_oracle_px")

liq = load_json_df("data/daily_notional_liquidated_by_coin.json", key="chart_data")
liq["time"] = pd.to_datetime(liq["time"])
liq = liq.pivot(index="time", columns="coin", values="daily_notional_liquidated").fillna(0)

inflows = load_json_df("data/daily_inflow.json", key="chart_data")
inflows["time"] = pd.to_datetime(inflows["time"])
inflows = inflows.set_index("time")["inflow"]

# --- LOAD LIQUIDITY DATA FOR SLIPPAGE ---
liquidity = load_json_df("data/liquidity_by_coin.json")  # No key needed for nested structure
liquidity["time"] = pd.to_datetime(liquidity["time"])
liquidity = liquidity.pivot(index="time", columns="coin", values="median_liquidity").fillna(1e9)  # Use median_liquidity instead of liquidity

# --- FUNDING RATES FETCHER ---
def fetch_funding_rates(coins, dates):
    funding = {}
    for coin in coins:
        start = int(dates[0].timestamp() * 1000)
        end = int(dates[-1].timestamp() * 1000)
        body = {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": start,
            "endTime": end
        }
        try:
            resp = requests.post("https://api.hyperliquid.xyz/info", json=body)
            data = resp.json()
            for entry in data:
                dt = datetime.utcfromtimestamp(entry["time"] / 1000)
                funding.setdefault(dt, {})[coin] = float(entry["fundingRate"])
        except Exception as e:
            print(f"Error fetching funding for {coin}: {e}")
    df = pd.DataFrame.from_dict(funding, orient="index").sort_index()
    return df.reindex(index=dates, columns=coins).fillna(0)

# --- PREPARE ---
coins = positions["coin"].unique().tolist()
dates = pd.date_range(positions["time"].min(), positions["time"].max())

positions_wide = positions.pivot(index="time", columns="coin", values="daily_ntl").fillna(0)

funding_rates = fetch_funding_rates(coins, dates)

# --- INITIALIZE ---
open_positions = {coin: {'size': 0, 'avg_entry': 0} for coin in coins}
realized_pnl = []
funding_paid = []
vault_value = 0
vault_value_series = []

# --- MAIN LOOP ---
for i, day in enumerate(dates):
    if i == 0:
        prev_day = day
        vault_value = positions_wide.loc[day].sum()
        vault_value_series.append(vault_value)
        continue
    prev_day = dates[i-1]
    # Optionally rebalance for inflows/outflows
    if apply_rebalance and day in inflows.index:
        inflow = inflows.loc[day]
        if vault_value != 0:
            scaling = (vault_value + inflow) / vault_value
            for coin in coins:
                open_positions[coin]['size'] *= scaling
            vault_value += inflow
    daily_pnl = 0
    daily_funding = 0
    for coin in coins:
        prev_size = open_positions[coin]['size']
        prev_entry = open_positions[coin]['avg_entry']
        today_size = positions_wide.at[day, coin] if day in positions_wide.index else 0
        today_price = prices.at[day, coin] if (day in prices.index and coin in prices.columns) else None
        prev_price = prices.at[prev_day, coin] if (prev_day in prices.index and coin in prices.columns) else None
        liq_ntl = liq.at[day, coin] if (day in liq.index and coin in liq.columns) else 0
        liq_today = liquidity.at[day, coin] if (day in liquidity.index and coin in liquidity.columns) else 1e9

        # Exclude liquidation notional from market making
        delta = today_size - prev_size
        mm_delta = delta - liq_ntl  # Only market making notional change

        # Entry (increase position)
        if mm_delta > 0 and today_price is not None:
            slip = slippage_factor * (mm_delta / liq_today)
            effective_price = today_price * (1 + slip)
            new_total = prev_size + mm_delta
            open_positions[coin]['avg_entry'] = (
                (prev_entry * prev_size + effective_price * mm_delta) / new_total
            ) if new_total != 0 else 0
            open_positions[coin]['size'] = new_total

        # Exit (decrease position)
        elif mm_delta < 0 and today_price is not None:
            exit_size = -mm_delta
            slip = slippage_factor * (exit_size / liq_today)
            effective_price = today_price * (1 - slip)
            pnl = exit_size * (effective_price - prev_entry)
            realized_pnl.append({'time': day, 'coin': coin, 'pnl': pnl})
            daily_pnl += pnl
            open_positions[coin]['size'] = prev_size + mm_delta
            if open_positions[coin]['size'] == 0:
                open_positions[coin]['avg_entry'] = 0

        # Funding payment
        funding_rate = funding_rates.at[day, coin] if (day in funding_rates.index and coin in funding_rates.columns) else 0
        funding_payment = open_positions[coin]['size'] * today_price * funding_rate if today_price is not None else 0
        funding_paid.append({'time': day, 'coin': coin, 'funding': funding_payment})
        daily_funding += funding_payment

    vault_value = sum(open_positions[coin]['size'] * (prices.at[day, coin] if (day in prices.index and coin in prices.columns) else 0) for coin in coins)
    vault_value_series.append(vault_value)

# --- AGGREGATE RESULTS ---
realized_pnl_df = pd.DataFrame(realized_pnl)
daily_realized_pnl = realized_pnl_df.groupby("time")["pnl"].sum().reindex(dates, fill_value=0)
funding_paid_df = pd.DataFrame(funding_paid)
daily_funding = funding_paid_df.groupby("time")["funding"].sum().reindex(dates, fill_value=0)

total_daily_pnl = daily_realized_pnl + daily_funding
total_cum_pnl = total_daily_pnl.cumsum()

# Apply 10% protocol fee (vault keeps 90%)
total_daily_pnl_net = total_daily_pnl * 0.9
total_cum_pnl_net = total_daily_pnl_net.cumsum()

# --- PLOT ---
plt.figure(figsize=(12,6))
plt.plot(dates, total_cum_pnl, label="Gross Cumulative PnL")
plt.plot(dates, total_cum_pnl_net, label="Net Cumulative PnL (90%)")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL (USD)")
plt.title("Reconstructed HLP Vault PnL (Realized, Funding, Slippage, No Fees)")
plt.legend()
plt.show() 
import os
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime


st.set_page_config(layout="wide")
st.title("HYPE Airdrop ROI")

st.subheader("Points Program Analysis")
# -- Config --
PRO_API_KEY = "CG-7paibS1SPUH9QTufMH3t3NQC"
CG_PRO_BASE = "https://pro-api.coingecko.com/api/v3"

# -- 0) Fetch & save fees.json from Hypurrscan --
FEES_ENDPOINT = "https://abc.hypurrscan.io/fees"
os.makedirs("data", exist_ok=True)
try:
    resp = requests.get(FEES_ENDPOINT)
    resp.raise_for_status()
    with open("data/fees.json", "w") as f:
        json.dump(resp.json(), f)
    st.sidebar.success("Fetched latest fees data")
except Exception as e:
    st.sidebar.error(f"Fees fetch failed: {e}")


# -- 1) Season definitions --
SEASONS = [
    ("Closed Alpha", None,           "2023-10-31", 4_460_000),
    ("Season 1",    "2023-11-01",    "2024-05-01", 26_000_000),
    ("Season 1.5",  "2024-05-01",    "2024-05-28", 8_000_000),
    ("Season 2",    "2024-05-29",    "2024-09-29", 14_000_000),
    ("Season 2.5",  "2024-09-30",    "2024-11-29", 8_400_000),
    ("Post TGE",    "2024-11-29",    None,        310_000_000),
]
for i, (n, s, e, pts) in enumerate(SEASONS):
    start = datetime.fromisoformat(s) if s else None
    end   = datetime.fromisoformat(e) if e else None
    SEASONS[i] = (n, start, end, pts)

# -- 2) Load on-chain time series --
@st.cache_data
def load_chain_ts():
    vol = pd.DataFrame(json.load(open("data/daily_usd_volume.json"))["chart_data"])
    vol["time"] = pd.to_datetime(vol["time"])
    vol = vol.rename(columns={"daily_usd_volume": "volume_usd"})

    usr = pd.DataFrame(json.load(open("data/daily_unique_users.json"))["chart_data"])
    usr["time"] = pd.to_datetime(usr["time"])
    usr = usr.rename(columns={"daily_unique_users": "users"})

    fees = pd.DataFrame(json.load(open("data/fees.json")))
    fees["time"] = pd.to_datetime(fees["time"], unit="s").dt.normalize()
    fees["fees_usd"] = fees["total_fees"] / 1e6

    df = (
        vol
        .merge(usr, on="time", how="outer")
        .merge(fees[["time", "fees_usd"]], on="time", how="outer")
        .sort_values("time")
        .fillna(method="ffill")
        .fillna(0)
    )
    return df

df = load_chain_ts()

# -- 3) Fetch HYPE price via CoinGecko Pro --
@st.cache_data(ttl=3600)
def fetch_hype_price():
    # find id
    coins = requests.get(f"{CG_PRO_BASE}/coins/list", headers={
        "accept": "application/json", "x-cg-pro-api-key": PRO_API_KEY
    }).json()
    hype = next(c for c in coins if c["symbol"].lower() == "hype")
    cid  = hype["id"]

    # call market_chart
    url = f"{CG_PRO_BASE}/coins/{cid}/market_chart"
    params = {"vs_currency": "usd", "days": "max"}
    data = requests.get(url, params=params, headers={
        "accept": "application/json", "x-cg-pro-api-key": PRO_API_KEY
    }).json()

    prices = pd.DataFrame(data["prices"], columns=["ts", "price"])
    prices["time"] = pd.to_datetime(prices["ts"], unit="ms").dt.normalize()
    return prices[["time", "price"]]

price_df = fetch_hype_price()
df = df.merge(price_df, on="time", how="left").fillna(method="ffill")

# -- 4) Simulate daily points --
records = []
for name, start, end, pts in SEASONS:
    season = df.copy()
    if start: season = season[season["time"] >= start]
    if end:   season = season[season["time"] < end]
    totv = season["volume_usd"].sum()
    if totv <= 0: continue
    season["daily_points"] = season["volume_usd"] / totv * pts
    season["season"] = name
    records.append(season)
pts_df = pd.concat(records, ignore_index=True)

# -- 5) Season summaries --
rows = []
for name, *_ in SEASONS:
    s = pts_df[pts_df["season"] == name]
    if s.empty: continue
    pts = s["daily_points"].sum()
    vol = s["volume_usd"].sum()
    nu  = s["users"].diff().fillna(s["users"]).sum()
    fe  = s["fees_usd"].sum()
    price_avg = s["price"].mean()
    rows.append({
      "season": name,
      "points": int(pts),
      "volume_usd": vol,
      "new_users": int(nu),
      "fees_usd": fe,
      "pts_per_usd": pts / vol,
      "pts_per_user": pts / nu if nu > 0 else None,
      "fees_per_hype": fe / (pts * price_avg)
    })
sum_df = pd.DataFrame(rows).set_index("season")

st.toast("Season by Season Summary")
st.dataframe(sum_df, width=900)

st.subheader("Cost Efficiency Across Seasons")
fig1 = px.bar(
    sum_df.reset_index(),
    x="season",
    y=["pts_per_usd", "pts_per_user"],
    barmode="group",
    labels={"value": "Cost", "variable": "Metric"},
    title="Cost Efficiency Across Seasons"
)

# Add custom colors for better distinction
fig1.update_layout(
    legend_title_text="Metric",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Season",
    yaxis_title="Cost",
    barmode="group"
)

st.plotly_chart(fig1, use_container_width=True)

# -- 8) Cumulative Fees vs HYPE-Airdropped --
st.subheader("Cumulative Fees vs HYPE-Airdropped")

pts_df["cum_fees"] = pts_df["fees_usd"].cumsum()
pts_df["cum_hype"] = pts_df["daily_points"].cumsum()

ch = (
    pts_df
    .groupby("time")[["cum_fees", "cum_hype"]]
    .last()
    .reset_index()
    .melt(
        id_vars="time",
        value_vars=["cum_fees", "cum_hype"],
        var_name="metric",
        value_name="value"
    )
)

# remap for nicer labels
ch["metric"] = ch["metric"].map({
    "cum_fees": "Cumulative Fees (USD)",
    "cum_hype": "Cumulative HYPE Airdropped"
})

fig2 = px.line(
    ch,
    x="time",
    y="value",
    color="metric",
    labels={"time": "Date", "value": "Amount", "metric": "Series"},
    title="Cumulative Fees vs HYPE-Tokens Airdropped"
)
st.plotly_chart(fig2, use_container_width=True)


st.markdown("""
---
*Fees pulled via Hypurrscan; prices & volume via CoinGecko Pro.*  
""")

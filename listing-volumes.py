import json
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import os
import requests
import plotly.express as px


# ——— Load listing dates ———
with open('data/coin_listing_dates.json', 'r') as f:
    raw = json.load(f)

listing_data = {
    tok: datetime.strptime(date, "%Y-%m-%d")
    for tok, date in raw.items()
    if not tok.startswith('@')
}
listing_df = pd.DataFrame({
    "Token": list(listing_data.keys()),
    "ListingDate": list(listing_data.values())
})

# ——— Load per‑token daily volumes ———
with open('data/daily_usd_volume_by_coin.json', 'r') as f:
    raw_volumes = json.load(f)

volume_data = {}
for entry in raw_volumes["chart_data"]:
    tok = entry["coin"]
    dt = datetime.strptime(entry["time"], "%Y-%m-%dT%H:%M:%S")
    vol = entry["daily_usd_volume"]
    volume_data.setdefault(tok, pd.Series(dtype=float)).at[dt] = vol

# ——— Compute total volume across all tokens ———
all_dates = sorted({dt for s in volume_data.values() for dt in s.index})
total_volume = pd.Series(0.0, index=all_dates)
for s in volume_data.values():
    total_volume = total_volume.add(s, fill_value=0)

# ——— Streamlit UI ———
st.title("Hyperliquid Deep Dive")

all_tokens = sorted(listing_df["Token"])
selected = st.multiselect("Select tokens:", all_tokens)
if not selected:
    selected = ["ETH", "AVAX", "BNB", "BTC", "SOL"]
    st.caption(f"No tokens chosen → defaulting to: {selected}")

events = listing_df[listing_df["Token"].isin(selected)]

# ——— Prepare a wide‐format DataFrame for all selected tokens ———
vol_wide = pd.DataFrame({
    tok: volume_data.get(tok, pd.Series(dtype=float))
                .reindex(all_dates, fill_value=0)
    for tok in selected
}, index=all_dates)

# ——— Melt to long form for stacking ———
vol_long = (
    vol_wide
    .reset_index()
    .melt(id_vars="index", var_name="Token", value_name="Volume")
    .rename(columns={"index": "Date"})
)

# ——— Build one stacked‐area chart with px.area ———
fig = px.area(
    vol_long,
    x="Date",
    y="Volume",
    color="Token",
    title="Token Listing Dates and Stacked Daily Volumes"
)

# ——— Overlay total volume as a subtle line ———
fig.add_scatter(
    x=total_volume.index,
    y=total_volume.values,
    mode="lines",
    name="Total Volume",
    line=dict(color="lightgray", width=2),
    opacity=0.5
)

# ——— Add listing‐date lines & annotations ———
ymax = total_volume.max() * 1.02
for _, row in events.iterrows():
    d = row["ListingDate"]
    fig.add_vline(x=d, line=dict(color="orange", dash="dash"), opacity=0.6, layer="below")
    fig.add_annotation(
        x=d, y=ymax,
        text=row["Token"],
        showarrow=False,
        yanchor="bottom",
        textangle=90,
        font=dict(size=10, color="orange")
    )

# ——— Final layout & single render call ———
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Volume",
    yaxis_range=[0, ymax],
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    margin=dict(l=40, r=40, t=60, b=40)
)

st.plotly_chart(fig, use_container_width=True)


# --- Load open interest data from JSON file ---
with open("data/open_interest.json", "r") as f:
    data = json.load(f)

# Convert JSON data to DataFrame
df = pd.DataFrame(data["chart_data"])
df["date"] = pd.to_datetime(df["time"])
df.rename(columns={"coin": "token", "open_interest": "openInterest"}, inplace=True)

# --- Determine Blue-Chip vs Meme-Coin dynamically based on daily volumes ---
# Simulate daily volumes for each token
daily_volumes = df.groupby("token")["openInterest"].sum().to_dict()
sorted_tokens = sorted(daily_volumes.items(), key=lambda x: x[1], reverse=True)
top_10_tokens = [token for token, _ in sorted_tokens[:10]]

# Tag each token as "Blue" or "Meme"
df["category"] = df["token"].isin(top_10_tokens).map({True: "Blue Chips", False: "Meme Coins"})
print(f"Top 10 tokens: {top_10_tokens}")
# --- Aggregate by date & category ---
agg = (
    df
    .groupby(["date", "category"], as_index=False)
    .openInterest
    .sum()
)

# --- Plot: Overlaid line chart + ratio subplot ---
# Compute ratio (meme / blue)
blue_oi = agg[agg.category == "Blue Chips"].set_index("date")["openInterest"]
meme_oi = agg[agg.category == "Meme Coins"].set_index("date")["openInterest"]
ratio = (meme_oi / blue_oi).reset_index().rename(columns={"openInterest": "ratio"})

# Create a single plot with total open interest in the background
fig = go.Figure()

# Add total open interest as a background trace
total_oi = agg.groupby("date")["openInterest"].sum()
fig.add_trace(
    go.Scatter(
        x=total_oi.index,
        y=total_oi.values,
        mode="lines",
        name="Total OI",
        line=dict(color="lightgray", width=2),
        opacity=0.5
    )
)

# Add main OI traces for Blue Chip and Meme Coin
for cat, color in [("Blue Chips", "blue"), ("Meme Coins", "red")]:
    d = agg[agg.category == cat]
    fig.add_trace(
        go.Scatter(
            x=d.date,
            y=d.openInterest,
            mode="lines",
            name=cat,
            line=dict(color=color)
        )
    )

# Add ratio trace
fig.add_trace(
    go.Scatter(
        x=ratio.date,
        y=ratio["ratio"],
        mode="lines",
        name="Memecoins / Bluechips OI Ratio",
        line=dict(color="orange", dash="dot")
    )
)

# Create a new figure for the ratio line only
fig_ratio = go.Figure()

# Add the ratio trace to the new figure
fig_ratio.add_trace(
    go.Scatter(
        x=ratio.date,
        y=ratio["ratio"],
        mode="lines",
        name="Memecoins / Bluechips OI Ratio",
        line=dict(color="orange", dash="dot")
    )
)

# Update layout for the ratio-only figure
fig_ratio.update_layout(
    height=400,
    hovermode="x unified",
    title_text="Memecoins / Bluechips Open Interest Ratio",
    xaxis_title="Date",
    yaxis_title="Ratio",
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom")
)

# Display the ratio-only chart in Streamlit
st.plotly_chart(fig_ratio, use_container_width=True)

# Update layout
fig.update_layout(
    height=600,
    hovermode="x unified",
    title_text="Open Interest: Blue‑Chips vs Meme Coins",
    xaxis_title="Date",
    yaxis_title="Open Interest (USD)",
    yaxis2=dict(
        title="Ratio",
        overlaying="y",
        side="right"
    ),
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom")
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)

# st.set_page_config(layout="wide")
# 

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

import plotly.express as px

df = sum_df.reset_index().melt(id_vars="season", 
                              value_vars=["pts_per_usd","pts_per_user"],
                              var_name="Metric", value_name="Cost")

fig = px.bar(
    df, 
    x="season", 
    y="Cost", 
    color="Metric", 
    barmode="group",   # side‑by‑side
    title="Cost Efficiency Across Seasons",
    labels={"season":"Season", "Cost":"Cost", "Metric":"Metric"}
)

st.plotly_chart(fig, use_container_width=True)

# -- 8) Cumulative Fees vs HYPE-Airdropped --
# st.subheader("Cumulative Fees vs HYPE-Airdropped")
pts_df["cum_fees"] = pts_df["fees_usd"].cumsum()
pts_df["cum_hype"] = pts_df["daily_points"].cumsum()

fig2 = go.Figure()

# Add cumulative fees trace (left y-axis)
fig2.add_trace(
    go.Scatter(
        x=pts_df["time"],
        y=pts_df["cum_fees"],
        mode="lines",
        name="Cumulative Fees (USD)",
        line=dict(color="blue"),
        yaxis="y1"
    )
)

# Add cumulative HYPE trace (right y-axis)
fig2.add_trace(
    go.Scatter(
        x=pts_df["time"],
        y=pts_df["cum_hype"],
        mode="lines",
        name="Cumulative HYPE Airdropped",
        line=dict(color="orange"),
        yaxis="y2"
    )
)

# Update layout to include dual y-axes
fig2.update_layout(
    title="Cumulative Fees vs HYPE-Tokens Airdropped",
    xaxis_title="Date",
    yaxis=dict(
        title="Cumulative Fees (USD)",
        title_font=dict(color="blue"),
        tickfont=dict(color="blue")
    ),
    yaxis2=dict(
        title="Cumulative HYPE Airdropped",
        title_font=dict(color="orange"),
        tickfont=dict(color="orange"),
        overlaying="y",
        side="right"
    ),
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom")
)

st.plotly_chart(fig2, use_container_width=True)

# ── 6) Load actual Hypurrscan fee breakdown ──
fees = pd.DataFrame(json.load(open("data/fees.json")))
fees["time"] = pd.to_datetime(fees["time"], unit="s").dt.normalize()
fees["total_fees_usd"] = fees["total_fees"] / 1e6
fees["spot_fees_usd"] = fees["total_spot_fees"] / 1e6
fees["perps_fees_usd"] = fees["total_fees_usd"] - fees["spot_fees_usd"]

# ── 7) Plot actual Perps vs Spot vs Total fees ──
# st.subheader("Actual Fees from Hypurrscan")
fig2 = px.line(
  fees,
  x="time",
  y=["perps_fees_usd", "spot_fees_usd", "total_fees_usd"],
  labels={"variable": "Series", "value": "USD", "time": "Date"},
  title="Actual Fee Breakdown: Perps vs Spot vs Total"
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
---
**Notes**  
- **Simulated** splits use a 14-day rolling volume ⇒ fee tiers ⇒ apply maker/taker bps.  
- **Actual** fees are pulled directly from Hypurrscan's `total_fees` and `total_spot_fees`.  
""")


st.markdown("""
---
*Fees pulled via Hypurrscan; prices & volume via CoinGecko Pro.*  
""")

# — 8) Blue‑Chip vs Meme‑Coin Analysis —
st.subheader("Blue‑Chip vs Meme‑Coin PnL / Funding")

hlp = pd.DataFrame(json.load(open("data/hlp_positions.json"))["chart_data"])
fund = pd.DataFrame(json.load(open("data/funding_rate.json"))["chart_data"])
# slip = pd.DataFrame(json.load(open("data/slippage.json"))["chart_data"])

hlp["time"]  = pd.to_datetime(hlp["time"])
fund["time"] = pd.to_datetime(fund["time"])
# no time in slip; just static slippage by coin

BLUE_CHIPS = {
    "BTC","ETH","BNB","SOL","AVAX",
    "LINK","UNI","MATIC","ADA","DOT",
    "LTC","ATOM","AAVE","COMP","MKR"
}
all_coins = set(hlp.coin) | set(fund.coin) 
# | set(slip.coin)
MEME_COINS = all_coins - BLUE_CHIPS

def segment_of(c):
    return "Meme-Coin" if c in MEME_COINS else "Blue-Chip"

hlp["segment"]  = hlp["coin"].apply(segment_of)
fund["segment"] = fund["coin"].apply(segment_of)
# slip["segment"] = slip["coin"].apply(segment_of)

# aggregate realized PnL
pnl_df = (
    hlp
    .groupby(["time","segment"])["daily_ntl"]
    .sum()
    .unstack(fill_value=0)
    .reset_index()
)

# aggregate funding PnL
fund_df = (
    fund
    .groupby(["time","segment"])["sum_funding"]
    .sum()
    .unstack(fill_value=0)
    .reset_index()
)

# average slippage
# slip_avg = (
#     slip
#     .groupby("segment")["slippage_pct"]
#     .mean()
#     .reset_index()
#     .rename(columns={"slippage_pct":"avg_slippage_pct"})
# )

# PnL chart
fig_pnl = px.line(
    pnl_df,
    x="time",
    y=["Blue-Chip","Meme-Coin"],
    labels={"value":"Daily PnL (USD)","time":"Date"},
    title="Daily Realized PnL by Segment"
)
st.plotly_chart(fig_pnl, use_container_width=True)

# Funding chart
fig_fund = px.line(
    fund_df,
    x="time",
    y=["Blue-Chip","Meme-Coin"],
    labels={"value":"Daily Funding PnL (USD)","time":"Date"},
    title="Daily Funding PnL by Segment"
)
st.plotly_chart(fig_fund, use_container_width=True)

# # Slippage bar
# fig_slip = px.bar(
#     slip_avg,
#     x="segment",
#     y="avg_slippage_pct",
#     labels={"segment":"Segment","avg_slippage_pct":"Avg Slippage (%)"},
#     title="Average Slippage Percentage by Segment"
# )
# st.plotly_chart(fig_slip, use_container_width=True)

# Summary table
summary = {
    "Metric": ["Total Realized PnL", "Total Funding PnL"],
    "Blue-Chip": [
        pnl_df["Blue-Chip"].sum(),
        fund_df["Blue-Chip"].sum(),
        # slip_avg.loc[slip_avg.segment=="Blue-Chip","avg_slippage_pct"].iat[0]
    ],
    "Meme-Coin": [
        pnl_df["Meme-Coin"].sum(),
        fund_df["Meme-Coin"].sum(),
        # slip_avg.loc[slip_avg.segment=="Meme-Coin","avg_slippage_pct"].iat[0]
    ]
}
summary_df = pd.DataFrame(summary).set_index("Metric")
st.table(summary_df)


import json
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")

# ========== Load Listing Dates ==========
with open('data/coin_listing_dates.json', 'r') as f:
    listing_data = {
        tok: datetime.strptime(date, "%Y-%m-%d")
        for tok, date in json.load(f).items()
        if not tok.startswith('@')
    }
listing_df = pd.DataFrame({"Token": list(listing_data.keys()), "ListingDate": list(listing_data.values())})

# ========== Load Daily Volumes ==========
with open('data/daily_usd_volume_by_coin.json', 'r') as f:
    raw_volumes = json.load(f)["chart_data"]

volume_data = {}
for entry in raw_volumes:
    tok, dt, vol = entry["coin"], datetime.strptime(entry["time"], "%Y-%m-%dT%H:%M:%S"), entry["daily_usd_volume"]
    volume_data.setdefault(tok, {})[dt] = vol

# Compute total volume across all tokens
all_dates = sorted({dt for s in volume_data.values() for dt in s})
total_volume = pd.Series(0.0, index=all_dates)
for s in volume_data.values():
    total_volume = total_volume.add(pd.Series(s), fill_value=0)

# UI
# st.title("Hyperliquid Deep Dive")
all_tokens = sorted(listing_df["Token"])
selected = st.multiselect("Select tokens:", all_tokens, default=["ETH", "TRUMP", "FARTCOIN", "CHILLGUY", "BERA"])
show_seasons = st.checkbox("Show Season Shading", value=True)
use_log_y = st.checkbox("Use log-scale Y-axis", value=False)

# Color map (ETH always blue)
color_map = px.colors.qualitative.Set1
token_colors = {tok: "#1f77b4" if tok == "ETH" else color_map[i % len(color_map)] for i, tok in enumerate(selected)}

# Masked volume (post-listing only)
masked_volume = {}
for tok in selected:
    listing = listing_data[tok]
    full_series = pd.Series(0.0, index=all_dates)
    vol_series = pd.Series(volume_data.get(tok, {}))
    vol_series = vol_series[vol_series.index >= listing]
    full_series.update(vol_series)
    masked_volume[tok] = full_series

vol_wide = pd.DataFrame(masked_volume)
ymax = total_volume.max() * 1.02

# Long-form volume data
vol_long = pd.concat([
    pd.DataFrame({"Date": vol.index, "Token": tok, "Volume": vol.values})
    for tok, vol in masked_volume.items() if not vol.empty
])

# Area chart
fig = px.area(vol_long, x="Date", y="Volume", color="Token", color_discrete_map=token_colors)
fig.add_scatter(x=all_dates, y=total_volume, mode="lines", name="Total Volume",
                line=dict(color="lightgray", width=2), opacity=0.5)

# Season shading
SEASONS = [
    ("Closed Alpha", None, "2023-10-31"),
    ("Season 1", "2023-11-01", "2024-05-01"),
    ("Season 1.5", "2024-05-01", "2024-05-28"),
    ("Season 2", "2024-05-29", "2024-09-29"),
    ("Season 2.5", "2024-09-30", "2024-11-29"),
    ("Post TGE", "2024-11-29", None)
]
season_colors = px.colors.qualitative.Pastel
latest_date = all_dates[-1]
earlist_date = all_dates[0]

if show_seasons:
    for i, (name, start, end) in enumerate(SEASONS):
        x0 = pd.to_datetime(start) if start else earlist_date - pd.Timedelta(days=5)
        x1 = pd.to_datetime(end) if end else latest_date + pd.Timedelta(days=5)
        fig.add_vrect(x0=x0, x1=x1, fillcolor=season_colors[i % len(season_colors)],
                      opacity=0.2, layer="below", line_width=0)
        fig.add_annotation(
            x=x0 + (x1 - x0) / 2, y=ymax * 1.1, text=f"<b>{name}</b>",
            showarrow=False, xanchor="center", yanchor="bottom", font=dict(size=11)
        )

# Listing lines
for tok in selected:
    listing = listing_data[tok]
    fig.add_trace(go.Scatter(
        x=[listing, listing], y=[0, ymax * 0.9],
        mode="lines", line=dict(color=token_colors[tok], dash="dash"),
        hovertemplate=f"<b>{tok}</b><br>Listed on: {listing.date()}<extra></extra>",
        opacity=0.85, showlegend=False
    ))
    fig.add_annotation(
        x=listing, y=ymax * 0.45, text=f"<b>{tok}</b>",
        showarrow=False, textangle=90, font=dict(size=10, color=token_colors[tok]), xshift=10
    )

fig.update_layout(
    yaxis=dict(title="Volume", type="log" if use_log_y else "linear"),
    xaxis_title="Date",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
    margin=dict(l=40, r=40, t=80, b=80)
)
st.plotly_chart(fig, use_container_width=True)

# ========== Open Interest & PnL Section ==========

blue_chips = ["BTC", "ETH", "USDT", "BNB", "SOL", "XRP", "USDC", "DOGE", "TON", "ADA"]
st.markdown("**Blue-Chip Tokens Considered:** " + ", ".join(sorted(blue_chips)))

with open("data/open_interest.json", "r") as f:
    df = pd.DataFrame(json.load(f)["chart_data"])
df["date"] = pd.to_datetime(df["time"])
df.rename(columns={"coin": "token", "open_interest": "openInterest"}, inplace=True)
df["category"] = df["token"].apply(lambda t: "Blue Chips" if t in blue_chips else "Meme Coins")

agg = df.groupby(["date", "category"], as_index=False)["openInterest"].sum()
blue_oi = agg[agg.category == "Blue Chips"].set_index("date")["openInterest"]
meme_oi = agg[agg.category == "Meme Coins"].set_index("date")["openInterest"]
ratio = (meme_oi / blue_oi).reset_index().rename(columns={"openInterest": "ratio"})

# OI and Ratio Chart
fig_combined = go.Figure()
fig_combined.add_trace(go.Scatter(x=agg.date.unique(), y=agg.groupby("date")["openInterest"].sum(),
                                  name="Total OI", line=dict(color="lightgray"), opacity=0.5))
for cat, color in [("Blue Chips", "blue"), ("Meme Coins", "red")]:
    d = agg[agg.category == cat]
    fig_combined.add_trace(go.Scatter(x=d.date, y=d.openInterest, name=cat, mode="lines", line=dict(color=color)))
fig_combined.add_trace(go.Scatter(x=ratio.date, y=ratio["ratio"], name="Memecoins / Bluechips OI Ratio",
                                  line=dict(color="orange", dash="dot"), yaxis="y2"))
fig_combined.update_layout(
    height=600, title_text="Open Interest and Memecoins / Bluechips OI Ratio",
    xaxis_title="Date",
    yaxis=dict(title="Open Interest (USD)"),
    yaxis2=dict(title="Ratio", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom")
)
st.plotly_chart(fig_combined, use_container_width=True)

# PnL / Funding Section
st.subheader("Blue‑Chip vs Meme‑Coin PnL and Funding")
hlp = pd.DataFrame(json.load(open("data/hlp_positions.json"))["chart_data"])
fund = pd.DataFrame(json.load(open("data/funding_rate.json"))["chart_data"])
hlp["time"], fund["time"] = pd.to_datetime(hlp["time"]), pd.to_datetime(fund["time"])
hlp["segment"] = hlp["coin"].apply(lambda c: "Blue-Chip" if c in blue_chips else "Meme-Coin")
fund["segment"] = fund["coin"].apply(lambda c: "Blue-Chip" if c in blue_chips else "Meme-Coin")

pnl_df = hlp.groupby(["time", "segment"])["daily_ntl"].sum().unstack(fill_value=0).reset_index()
fund_df = fund.groupby(["time", "segment"])["sum_funding"].sum().unstack(fill_value=0).reset_index()

# PnL Chart with colored bands
fig_pnl = go.Figure()
colors = ["green" if b > m else "red" for b, m in zip(pnl_df["Blue-Chip"], pnl_df["Meme-Coin"])]
for i in range(len(pnl_df) - 1):
    fig_pnl.add_vrect(x0=pnl_df["time"][i], x1=pnl_df["time"][i+1], fillcolor=colors[i], opacity=0.12, line_width=0)
fig_pnl.add_trace(go.Scatter(x=pnl_df["time"], y=pnl_df["Blue-Chip"], name="Blue-Chip", line=dict(color="blue")))
fig_pnl.add_trace(go.Scatter(x=pnl_df["time"], y=pnl_df["Meme-Coin"], name="Meme-Coin", line=dict(color="orange")))
fig_pnl.update_layout(title="Daily Realized PnL by Segment", xaxis_title="Date", yaxis_title="Daily PnL (USD)",
                      hovermode="x unified", legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))

st.plotly_chart(fig_pnl, use_container_width=True)

st.markdown("Blue-chip PnL > Meme-coin PnL; green shows blue-chip lead over meme-coins, red shows the opposite.")
st.markdown("Funding PnL is the sum of funding rates received or paid by the protocol. Positive values indicate profit, negative values indicate loss.")

# Funding Chart
fig_fund = px.line(fund_df, x="time", y=["Blue-Chip", "Meme-Coin"],
                   title="Daily Funding PnL by Segment", labels={"value": "Funding PnL (USD)", "time": "Date"})
st.plotly_chart(fig_fund, use_container_width=True)

# Metric Summary Table
def human_fmt(val):
    return f"{val/1e9:.2f}B" if abs(val) >= 1e9 else f"{val/1e6:.2f}M" if abs(val) >= 1e6 else f"{val:,.2f}"

summary = {
    "Metric": ["Total Realized $ PnL", "Total Funding $ PnL"],
    "Blue-Chip": [human_fmt(pnl_df["Blue-Chip"].sum()), human_fmt(fund_df["Blue-Chip"].sum())],
    "Meme-Coin": [human_fmt(pnl_df["Meme-Coin"].sum()), human_fmt(fund_df["Meme-Coin"].sum())],
}
st.table(pd.DataFrame(summary).set_index("Metric"))

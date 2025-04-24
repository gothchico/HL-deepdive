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

    usr = pd.DataFrame(json.load(open("data/cumulative_new_users.json"))["chart_data"])
    usr["time"] = pd.to_datetime(usr["time"])
    usr = usr.rename(columns={"daily_unique_users": "users"})
    usr.set_index("time", inplace=True)  # Set 'time' as the index for resampling

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

# st.markdown("Debugging: Price DataFrame")
# st.dataframe(df, width=600)

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
    

# st.markdown("Debugging: Points DataFrame")
# st.dataframe(pts_df, width=1100)
# -- 5) Season summaries --
rows = []
for name, start, end, *_ in SEASONS:
    s = pts_df[pts_df["season"] == name]
    if s.empty: continue
    pts = s["daily_points"].sum()
    vol = s["volume_usd"].sum()
    nu = s["cumulative_new_users"].iloc[-1] - s["cumulative_new_users"].iloc[0]
    fe  = s["fees_usd"].sum()
    price_avg = s["price"].mean()
    def format_val(x):
        if x >= 1e9:
            return f"{x / 1e9:.3f} B"
        elif x >= 1e6:
            return f"{x / 1e6:.3f} M"
        elif x >= 1e3:
            return f"{x / 1e3:.2f} K"
        else:
            return f"{x}"

    rows.append({
        "season": name,
        "dates": (
            (s["time"].min().strftime("June 2023") if start is None else s["time"].min().strftime("%b %d %Y")) +
            " – " +
            (s["time"].max().strftime("%b %d %Y") if end is not None else "today")
        ),
        "points_distributed": format_val(pts),
        "#days": (s["time"].max() - s["time"].min()).days + 1,
        "volume_usd": format_val(vol),
        "new_users": format_val(nu),
        "fees_usd": format_val(fe),
        "pts_per_usd": (pts / (s["volume_usd"] * ((s["time"].max() - s["time"].min()).days + 1))).mean(),
        "pts_per_user": pts / nu if nu > 0 else None,
        "fees_per_point": fe / pts
    })
    
sum_df = pd.DataFrame(rows).set_index("season")

st.toast("Season by Season Summary")
st.dataframe(sum_df, width=1100)


st.markdown("Here pts_per_usd is calculated as the total points distributed divided by the daily volume traded in that season (normalised by #days in that season to get a DAILY average),"
" while pts_per_user, which is the TOTAL points distributed divided by the number of new users in that season.")
st.subheader("Cost Efficiency Across Seasons")

from plotly.subplots import make_subplots

# Add subplot with secondary y-axis
fig1 = make_subplots(
    specs=[[{"secondary_y": True}]],
    shared_xaxes=True,
    vertical_spacing=0.1
)

# Checkbox for including Closed Alpha
include_closed_alpha = st.checkbox("Include Closed Alpha", value=False)

# Filter out the Closed Alpha season if the box is unchecked
filtered_sum_df = sum_df if include_closed_alpha else sum_df.drop("Closed Alpha", errors="ignore")

# Calculate pts_per_usd as a time series for each season
# filtered_sum_df["pts_per_usd"] = pts_df["daily_points"] / pts_df["volume_usd"]

# First metric: pts_per_usd (left y-axis)
fig1.add_bar(
    x=filtered_sum_df.index,
    y=filtered_sum_df["pts_per_usd"],
    name="Points per USD",
    marker_color="lightskyblue",
    offsetgroup=0,
    yaxis="y1"
)

# Second metric: pts_per_user (right y-axis)
fig1.add_bar(
    x=filtered_sum_df.index,
    y=filtered_sum_df["pts_per_user"],
    name="Points per User",
    marker_color="springgreen",
    offsetgroup=1,
    yaxis="y2"
)

# Update layout
fig1.update_layout(
    # title="Cost Efficiency Across Seasons (Dual Axis)",
    barmode="group",
    xaxis_title="Season",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=40, r=40, t=60, b=40),
    hovermode="x unified"
)

# Add y-axis titles
fig1.update_yaxes(title_text="Points per USD", secondary_y=False)
fig1.update_yaxes(title_text="Points per User", secondary_y=True)

st.plotly_chart(fig1, use_container_width=True)


# st.plotly_chart(fig1, use_container_width=True)

# -- 8) Cumulative Fees vs HYPE-Airdropped --
st.subheader("Cumulative Fees vs HYPE-Airdropped")

import plotly.graph_objects as go

# Compute cumulative metrics
pts_df["cum_fees"] = pts_df["fees_usd"].cumsum()
pts_df["cum_hype"] = pts_df["daily_points"].cumsum()

# Group by time for the dual axis plot
grouped = pts_df.groupby("time")[["cum_fees", "cum_hype"]].last().reset_index()

# Create a subplot with a secondary y-axis
fig2 = make_subplots(specs=[[{"secondary_y": True}]])

# Add cumulative fees trace on the primary y-axis
fig2.add_trace(
    go.Scatter(
        x=grouped["time"],
        y=grouped["cum_fees"],
        mode="lines",
        name="Cumulative Fees (USD)",
        line=dict(color="firebrick")
    ),
    secondary_y=False
)

# Add cumulative HYPE airdropped trace on the secondary y-axis
fig2.add_trace(
    go.Scatter(
        x=grouped["time"],
        y=grouped["cum_hype"],
        mode="lines",
        name="Cumulative points distributed",
        line=dict(color="royalblue")
    ),
    secondary_y=True
)

# Update layout and axes labels
fig2.update_layout(
    # title="Cumulative Fees vs Points Distribution",
    xaxis_title="Date",
    margin=dict(l=40, r=40, t=60, b=40),
    hovermode="x unified"
)

fig2.update_yaxes(title_text="Cumulative Fees (USD)", secondary_y=False)
fig2.update_yaxes(title_text="Cumulative Fees vs Points Distribution", secondary_y=True)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
---
*Fees pulled via Hypurrscan -> Shifting to Defillama API soon; prices & volume via CoinGecko Pro.*  

""")

# -- 9) Cumulative New Users with Seasonal Shading --
st.subheader("Cumulative New Users")

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# st.markdown("Debugging: Points DataFrame")
# st.dataframe(pts_df, width=1100)

usr = pd.DataFrame(json.load(open("data/cumulative_new_users.json"))["chart_data"])
usr["time"] = pd.to_datetime(usr["time"])
usr = usr.rename(columns={"daily_unique_users": "users"})

# Compute daily and cumulative new users
# pts_df["daily_new_users"] = pts_df["users"]
usr["cumulative_new_users"] = usr["cumulative_new_users"]

# Create a subplot with a secondary y-axis
fig3 = make_subplots(specs=[[{"secondary_y": True}]])

# Daily new users (bars)
fig3.add_trace(
    go.Bar(
        x=usr["time"],
        y=usr["daily_new_users"],
        name="Daily New Users",
        marker_color="salmon",
        opacity=0.75,
    ),
    secondary_y=False
)

# Cumulative new users (line)
fig3.add_trace(
    go.Scatter(
        x=usr["time"],
        y=usr["cumulative_new_users"],
        mode="lines",
        name="Cumulative New Users",
        line=dict(color="purple")
    ),
    secondary_y=True
)

# Season shading
season_colors = px.colors.qualitative.Set3
latest_date = usr["time"].max()
earliest_date = usr["time"].min()

for i, (name, start, end, _) in enumerate(SEASONS):
    x0 = start if start else earliest_date - pd.Timedelta(days=5)
    x1 = end if end else latest_date + pd.Timedelta(days=5)
    fig3.add_vrect(
        x0=x0,
        x1=x1,
        fillcolor=season_colors[i % len(season_colors)],
        opacity=0.2,
        layer="below",
        line_width=0
    )
    fig3.add_annotation(
        x=x0 + (x1 - x0) / 2,
        y=max(usr["daily_new_users"])*1.02,
        text=f"<b>{name}</b>",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font=dict(size=11)
    )

# Layout & axes
fig3.update_layout(
    title="Daily vs Cumulative New Users with Seasonal Shading",
    xaxis_title="Date",
    margin=dict(l=40, r=40, t=60, b=40),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig3.update_yaxes(title_text="Daily New Users", secondary_y=False)
fig3.update_yaxes(title_text="Cumulative New Users", secondary_y=True)

# Display
# st.subheader("Cumulative New Users")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("User Retention-Dropoff Rate Analysis")
from datetime import datetime, timedelta


# st.title("User Drop-off Rate Analysis")

# === Load user data ===
with open("data/cumulative_new_users.json", "r") as f:
    user_data = json.load(f)["chart_data"]

usr = pd.DataFrame(user_data)
usr["time"] = pd.to_datetime(usr["time"])
usr = usr.rename(columns={
    "daily_new_users": "daily_new_users",
    "cumulative_new_users": "cumulative_new_users"
})
usr.set_index("time", inplace=True)

# === Rolling retention logic ===
def compute_retention_window(rule):
    df = usr[["daily_new_users", "cumulative_new_users"]].resample(rule).agg({
        "daily_new_users": "sum",
        "cumulative_new_users": "last"
    })
    df["expected_new"] = df["daily_new_users"]
    df["actual_new"] = df["cumulative_new_users"].diff().fillna(df["cumulative_new_users"])
    df["dropoff"] = df["expected_new"] - df["actual_new"]
    df["dropoff_rate"] = df["dropoff"] / df["expected_new"].replace(0, pd.NA)
    return df

# === Dropdown selector ===
resample_options = {
    "Daily": compute_retention_window("D"),
    "Weekly": compute_retention_window("W-MON"),
    "Fortnightly": compute_retention_window("2W-MON"),
    "Monthly": compute_retention_window("M")
}
window = st.selectbox("Select drop-off smoothing window:", resample_options.keys())
df = resample_options[window].reset_index()

# === Debug: show table tail ===
st.subheader("Debug: Dropoff Table Tail")
st.dataframe(df.tail())

# === Season definitions ===
SEASONS = [
    ("Closed Alpha", None, "2023-10-31"),
    ("Season 1", "2023-11-01", "2024-05-01"),
    ("Season 1.5", "2024-05-01", "2024-05-28"),
    ("Season 2", "2024-05-29", "2024-09-29"),
    ("Season 2.5", "2024-09-30", "2024-11-29"),
    ("Post TGE", "2024-11-29", None)
]
season_colors = px.colors.qualitative.Pastel
latest_date = df["time"].max()
earliest_date = df["time"].min()

# === Dropoff plot ===
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["dropoff_rate"],
    mode="lines+markers",
    name="Dropoff Rate",
    line=dict(color="crimson")
))

# === Add seasonal shading ===
for i, (name, start, end) in enumerate(SEASONS):
    x0 = pd.to_datetime(start) if start else earliest_date - timedelta(days=5)
    x1 = pd.to_datetime(end) if end else latest_date + timedelta(days=5)
    fig.add_vrect(
        x0=x0,
        x1=x1,
        fillcolor=season_colors[i % len(season_colors)],
        opacity=0.2,
        layer="below",
        line_width=0
    )
    fig.add_annotation(
        x=x0 + (x1 - x0) / 2,
        y=df["dropoff_rate"].quantile(0.95),
        text=f"<b>{name}</b>",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font=dict(size=11)
    )

fig.update_layout(
    title="User Drop-off Rate Over Time",
    xaxis_title="Date",
    yaxis_title="Drop-off Rate",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)

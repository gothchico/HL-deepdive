import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("HLP Liquidator ROI & Risk Analysis")

# --- Load Data ---
@st.cache_data
def load_data():
    pnl = pd.DataFrame(json.load(open("data/hlp_liquidator_pnl"))["chart_data"])
    daily_liq = pd.DataFrame(json.load(open("data/daily_notional_liquidated_total"))["chart_data"])
    cum_liq = pd.DataFrame(json.load(open("data/cumulative_liquidated_notional"))["chart_data"])
    assets = pd.DataFrame(json.load(open("data/asset_ctxs.json"))["chart_data"])
    hlp = pd.DataFrame(json.load(open("data/hlp_positions.json"))["chart_data"])
    return pnl, daily_liq, cum_liq, assets, hlp

pnl, daily_liq, cum_liq, assets, hlp = load_data()

# --- Clean & Merge ---
pnl["time"] = pd.to_datetime(pnl["time"])
daily_liq["time"] = pd.to_datetime(daily_liq["time"])
cum_liq["time"] = pd.to_datetime(cum_liq["time"])
assets["time"] = pd.to_datetime(assets["time"])
hlp["time"] = pd.to_datetime(hlp["time"])

daily = (
    pnl
    .groupby("time")[["total_pnl"]]
    .sum()
    .reset_index()
    .rename(columns={"total_pnl": "daily_pnl"})
)

daily["cum_pnl"] = daily["daily_pnl"].cumsum()
daily["pnl_volatility"] = daily["daily_pnl"].rolling(7).std()

# Max drawdown
cum_max = daily["cum_pnl"].cummax()
drawdown = daily["cum_pnl"] - cum_max
daily["drawdown"] = drawdown
daily["drawdown_pct"] = drawdown / cum_max.replace(0, pd.NA)

# --- Estimate Vault NAV ---
hlp_agg = (
    hlp
    .groupby("time")["daily_ntl"]
    .sum()
    .cumsum()
    .reset_index(name="vault_NAV")
)

vault = hlp_agg.merge(
    daily[["time", "cum_pnl"]],
    on="time", how="outer"
).sort_values("time").fillna(method="ffill")

# --- Plot ROI Curve ---
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(
        x=daily["time"],
        y=daily["cum_pnl"],
        name="Cumulative Liquidator PnL",
        line=dict(color="orange")
    ),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=vault["time"],
        y=vault["vault_NAV"],
        name="Vault NAV (HLP Net Exposure)",
        line=dict(color="blue", dash="dot")
    ),
    secondary_y=True
)

fig.update_layout(
    title="Cumulative PnL vs Vault NAV",
    xaxis_title="Date",
    yaxis_title="Cumulative PnL (USD)",
    yaxis2=dict(title="Vault NAV", overlaying="y", side="right"),
    hovermode="x unified",
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
)
st.plotly_chart(fig, use_container_width=True)

# --- Drawdown and Volatility ---
st.subheader("PnL Risk Metrics")

fig_risk = make_subplots(specs=[[{"secondary_y": True}]])

fig_risk.add_trace(
    go.Scatter(
        x=daily["time"],
        y=daily["drawdown_pct"],
        name="Drawdown %",
        line=dict(color="crimson")
    ),
    secondary_y=False
)

fig_risk.add_trace(
    go.Scatter(
        x=daily["time"],
        y=daily["pnl_volatility"],
        name="Volatility (7d rolling)",
        line=dict(color="green")
    ),
    secondary_y=True
)

fig_risk.update_layout(
    title="Drawdown and Volatility",
    xaxis_title="Date",
    yaxis=dict(title="Drawdown %"),
    yaxis2=dict(title="Volatility", overlaying="y", side="right"),
    hovermode="x unified",
    margin=dict(l=40, r=40, t=60, b=40)
)

st.plotly_chart(fig_risk, use_container_width=True)

# --- Summary Stats ---
st.subheader("Summary Statistics")
summary = {
    "Total PnL": f"${daily['daily_pnl'].sum():,.0f}",
    "Max Drawdown": f"{daily['drawdown_pct'].min() * 100:.2f}%",
    "Volatility (7d avg)": f"{daily['pnl_volatility'].mean():,.2f}"
}
st.dataframe(pd.DataFrame(summary.items(), columns=["Metric", "Value"]))

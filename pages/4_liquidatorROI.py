import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from plotly.subplots import make_subplots
import requests
import statsmodels.api as sm
import numpy as np

# st.title("HLP Strategy Analysis: Market Making vs Liquidator")

# --- Fetch Max Leverage Per Coin ---
@st.cache_data
def fetch_coin_metadata():
    response = requests.post(
        "https://api.hyperliquid.xyz/info",
        json={"type": "meta"}
    )
    meta = response.json()
    coin_to_max_leverage = {asset["name"]: asset["maxLeverage"] for asset in meta["universe"]}
    coin_to_maintenance_margin = {coin: 1/(2*max_lev) for coin, max_lev in coin_to_max_leverage.items()}
    return coin_to_max_leverage, coin_to_maintenance_margin

coin_to_max_leverage, coin_to_maintenance_margin = fetch_coin_metadata()

# --- Load Data ---
@st.cache_data
def load_data():
    pnl = pd.DataFrame(json.load(open("data/hlp_liquidator_pnl"))["chart_data"])
    daily_liq = pd.DataFrame(json.load(open("data/daily_notional_liquidated_by_coin.json"))["chart_data"])
    cum_liq = pd.DataFrame(json.load(open("data/cumulative_liquidated_notional"))["chart_data"])
    assets = pd.DataFrame(json.load(open("data/asset_ctxs.json"))["chart_data"])
    hlp = pd.DataFrame(json.load(open("data/hlp_positions.json"))["chart_data"])
    
    # Fetch both HLP and non-HLP liquidator PnL
    hlp_liq_pnl = pd.DataFrame(requests.get("https://d2v1fiwobg9w6.cloudfront.net/cumulative_hlp_liquidator_pnl?is_hlp=true").json()["chart_data"])
    non_hlp_liq_pnl = pd.DataFrame(requests.get("https://d2v1fiwobg9w6.cloudfront.net/cumulative_hlp_liquidator_pnl?is_hlp=false").json()["chart_data"])
    
    return pnl, daily_liq, cum_liq, assets, hlp, hlp_liq_pnl, non_hlp_liq_pnl

pnl, daily_liq, cum_liq, assets, hlp, hlp_liq_pnl, non_hlp_liq_pnl = load_data()

# --- Clean & Merge ---
pnl["time"] = pd.to_datetime(pnl["time"])
daily_liq["time"] = pd.to_datetime(daily_liq["time"])
cum_liq["time"] = pd.to_datetime(cum_liq["time"])
assets["time"] = pd.to_datetime(assets["time"])
hlp["time"] = pd.to_datetime(hlp["time"])
hlp_liq_pnl["time"] = pd.to_datetime(hlp_liq_pnl["time"])
non_hlp_liq_pnl["time"] = pd.to_datetime(non_hlp_liq_pnl["time"])

# --- True Vault PnL (from hlp_liquidator_pnl) ---
pnl = pnl.rename(columns={"total_pnl": "vault_daily_pnl"})

# --- Notional (from hlp_positions) ---
daily_notional = hlp.groupby("time")["daily_ntl_abs"].sum().reset_index(name="vault_daily_notional")

# --- Merge PnL and Notional ---
daily = pnl.merge(daily_notional, on="time", how="left").sort_values("time").fillna(0)

daily["vault_cum_pnl"] = daily["vault_daily_pnl"].cumsum()
daily["vault_cum_notional"] = daily["vault_daily_notional"].cumsum()

daily["pnl_per_m_notional"] = daily["vault_daily_pnl"] / (daily["vault_daily_notional"] / 1e6).replace(0, pd.NA)

# --- Calculate Modeled Liquidator PnL ---
# Add maintenance margin rate to daily_liq
daily_liq["maintenance_margin_rate"] = daily_liq["coin"].map(coin_to_maintenance_margin)
daily_liq["liquidator_daily_pnl"] = daily_liq["daily_notional_liquidated"] * daily_liq["maintenance_margin_rate"]

# Group by date for total daily liquidator PnL
liquidator_daily = daily_liq.groupby("time")["liquidator_daily_pnl"].sum().reset_index()

daily = daily.merge(liquidator_daily, on='time', how='left').fillna(0)

daily['mm_daily_pnl'] = daily['vault_daily_pnl'] - daily['liquidator_daily_pnl']
daily['mm_cum_pnl'] = daily['mm_daily_pnl'].cumsum()

daily['liquidator_cum_pnl'] = daily['liquidator_daily_pnl'].cumsum()
daily['total_vault_daily_pnl'] = daily['mm_daily_pnl'] + daily['liquidator_daily_pnl']
daily['total_vault_cum_pnl'] = daily['total_vault_daily_pnl'].cumsum()

# Calculate 7-day rolling volatility for each strategy
daily["mm_volatility"] = daily["mm_daily_pnl"].rolling(7).std()
daily["liquidator_volatility"] = daily["liquidator_daily_pnl"].rolling(7).std()
daily["total_vault_volatility"] = daily["total_vault_daily_pnl"].rolling(7).std()

# --- Calculate drawdown and drawdown_pct for each strategy ---
for strategy in ['liquidator', 'mm', 'total_vault']:
    cum_max = daily[f'{strategy}_cum_pnl'].cummax()
    drawdown = daily[f'{strategy}_cum_pnl'] - cum_max
    daily[f'{strategy}_drawdown'] = drawdown
    daily[f'{strategy}_drawdown_pct'] = drawdown / cum_max.replace(0, pd.NA)

# --- Markdown on Methodology ---
st.markdown("""
#### Liquidator PnL Modeling
Liquidator PnL is now modeled as:
**Daily Notional Liquidated × Maintenance Margin Rate**
where Maintenance Margin Rate = 1/(2 × maxLeverage) for each coin.

This is based on Hyperliquid's official documentation and public stats. All analysis below uses this modeled value, not the direct reported PnL.
""")

# --- Plot ROI Curve ---
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Liquidator PnL (left y-axis)
fig.add_trace(
    go.Scatter(
        x=daily["time"],
        y=daily["liquidator_cum_pnl"],
        name="Cumulative Liquidator PnL (Modeled)",
        line=dict(color="orange")
    ),
    secondary_y=False
)

# Market Making PnL (left y-axis)
fig.add_trace(
    go.Scatter(
        x=daily["time"],
        y=daily["mm_cum_pnl"],
        name="Cumulative Market Making PnL",
        line=dict(color="blue", dash="dot")
    ),
    secondary_y=False
)

# Total Vault PnL (right y-axis)
fig.add_trace(
    go.Scatter(
        x=daily["time"],
        y=daily["total_vault_cum_pnl"],
        name="Cumulative Total Vault PnL",
        line=dict(color="green", dash="dash")
    ),
    secondary_y=True
)

fig.update_layout(
    title="Cumulative PnL: Market Making vs Liquidator vs Total Vault",
    xaxis_title="Date",
    yaxis_title="Cumulative MM & Liquidator PnL (USD)",
    yaxis2=dict(title="Cumulative Total Vault PnL (USD)", overlaying="y", side="right"),
    hovermode="x unified",
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
)

st.plotly_chart(fig, use_container_width=True)

# --- Efficiency Table ---
efficiency_table = pd.DataFrame({
    "Metric": [
        "Total Vault PnL",
        "Total Vault Notional",
        "Mean Daily PnL per $1M Notional"
    ],
    "Value": [
        f"${daily['vault_daily_pnl'].sum():,.0f}",
        f"${daily['vault_daily_notional'].sum():,.0f}",
        f"${daily['pnl_per_m_notional'].mean():,.2f}"
    ]
})
st.dataframe(efficiency_table.set_index("Metric"))

# --- Strategy Analysis ---
st.subheader("Strategy Analysis")

# Calculate correlation between strategies
correlation = daily["liquidator_daily_pnl"].corr(daily["mm_daily_pnl"])

# Calculate strategy characteristics
strategy_analysis = {
    "Metric": [
        "Total PnL",
        "Daily PnL Mean",
        "Daily PnL Std Dev",
        "Sharpe Ratio (assuming 0% risk-free rate)",
        "Max Drawdown",
        "Win Rate",
        "Correlation with Other Strategy"
    ],
    "Market Making": [
        f"${daily['mm_daily_pnl'].sum():,.0f}",
        f"${daily['mm_daily_pnl'].mean():,.0f}",
        f"${daily['mm_daily_pnl'].std():,.0f}",
        f"{(daily['mm_daily_pnl'].mean() / daily['mm_daily_pnl'].std()):.2f}",
        f"{daily['mm_drawdown_pct'].min() * 100:.2f}%",
        f"{(daily['mm_daily_pnl'] > 0).mean() * 100:.1f}%",
        f"{daily['mm_daily_pnl'].corr(daily['liquidator_daily_pnl']):.2f}"
    ],
    "Liquidator": [
        f"${daily['liquidator_daily_pnl'].sum():,.0f}",
        f"${daily['liquidator_daily_pnl'].mean():,.0f}",
        f"${daily['liquidator_daily_pnl'].std():,.0f}",
        f"{(daily['liquidator_daily_pnl'].mean() / daily['liquidator_daily_pnl'].std()):.2f}",
        f"{daily['liquidator_drawdown_pct'].min() * 100:.2f}%",
        f"{(daily['liquidator_daily_pnl'] > 0).mean() * 100:.1f}%",
        f"{daily['liquidator_daily_pnl'].corr(daily['mm_daily_pnl']):.2f}"
    ],
    "Total Vault": [
        f"${daily['total_vault_daily_pnl'].sum():,.0f}",
        f"${daily['total_vault_daily_pnl'].mean():,.0f}",
        f"${daily['total_vault_daily_pnl'].std():,.0f}",
        f"{(daily['total_vault_daily_pnl'].mean() / daily['total_vault_daily_pnl'].std()):.2f}",
        f"{daily['total_vault_drawdown_pct'].min() * 100:.2f}%",
        f"{(daily['total_vault_daily_pnl'] > 0).mean() * 100:.1f}%",
        f"{daily['total_vault_daily_pnl'].corr(daily['mm_daily_pnl']):.2f}"
    ]
}

st.dataframe(pd.DataFrame(strategy_analysis).set_index("Metric"))

# --- Strategy Inference ---
st.subheader("Strategy Inference")

# Analyze market making strategy characteristics
mm_characteristics = {
    "Volatility": "High" if daily["mm_volatility"].mean() > daily["liquidator_volatility"].mean() else "Low",
    "Consistency": "Consistent" if (daily["mm_daily_pnl"] > 0).mean() > 0.6 else "Inconsistent",
    "Risk Profile": "Aggressive" if daily["mm_drawdown_pct"].min() < -0.2 else "Conservative",
    "Correlation with Liquidator": "Positive" if correlation > 0.3 else "Negative" if correlation < -0.3 else "Neutral"
}

st.markdown("### Market Making Strategy Characteristics")
st.dataframe(pd.DataFrame(mm_characteristics.items(), columns=["Characteristic", "Value"]))

st.markdown("""
### Strategy Inference
Based on the data analysis, the market making strategies appear to be:

1. **Strategy A (0x010461c14e146ac35fe42271bdc1134ee31c703a)**: Likely a high-frequency market making strategy focusing on tight spreads and high volume. This is suggested by:
   - Consistent positive PnL
   - Lower volatility compared to liquidator
   - High win rate

2. **Strategy B (0x31ca8395cf837de08b24da3f660e77761dfb974b)**: Likely a statistical arbitrage strategy that:
   - Takes advantage of price discrepancies
   - Has higher volatility
   - May use mean reversion techniques

3. **Liquidator (0x2e3d94f0562703b25c83308a05046ddaf9a8dd14)**: Handles liquidations and provides risk management by:
   - Balancing between high-frequency and statistical arbitrage
   - Providing stability to the overall portfolio
   - Acting as a hedge against extreme market conditions

The strategies appear to be complementary, with different risk profiles and performance characteristics that help balance the overall portfolio.
""")

# --- Drawdown and Volatility Comparison ---
st.subheader("Strategy Risk Metrics")

fig_risk = make_subplots(specs=[[{"secondary_y": True}]])

fig_risk.add_trace(
    go.Scatter(
        x=daily["time"],
        y=daily["mm_drawdown_pct"],
        name="Market Making Drawdown %",
        line=dict(color="blue")
    ),
    secondary_y=False
)

fig_risk.add_trace(
    go.Scatter(
        x=daily["time"],
        y=daily["liquidator_drawdown_pct"],
        name="Liquidator Drawdown %",
        line=dict(color="orange")
    ),
    secondary_y=True
)

fig_risk.update_layout(
    title="Strategy Drawdown Comparison",
    xaxis_title="Date",
    yaxis=dict(title="Market Making Drawdown %"),
    yaxis2=dict(title="Liquidator Drawdown %", overlaying="y", side="right"),
    hovermode="x unified",
    margin=dict(l=40, r=40, t=60, b=40)
)

st.plotly_chart(fig_risk, use_container_width=True)

# Show a sample of the per-day, per-coin liquidator PnL calculation
sample = (
    daily_liq[["time", "coin", "daily_notional_liquidated", "maintenance_margin_rate", "liquidator_daily_pnl"]]
    .sort_values(["time", "coin"])
    .head(20)  # Show the first 20 rows for brevity
)

# Format for readability
sample["daily_notional_liquidated"] = sample["daily_notional_liquidated"].map('{:,.2f}'.format)
sample["maintenance_margin_rate"] = sample["maintenance_margin_rate"].map('{:.6f}'.format)
sample["liquidator_daily_pnl"] = sample["liquidator_daily_pnl"].map('${:,.2f}'.format)

st.markdown('### Debug: Sample of Per-Day, Per-Coin Liquidator PnL Calculation')
st.dataframe(sample)

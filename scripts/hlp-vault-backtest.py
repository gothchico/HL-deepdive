import json
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🔍 HLP Vault Backtest (20× Leverage)")

LEVERAGE = 20

@st.cache_data
def load_and_prepare():
    # — 1) Load JSONs —
    with open("data/hlp_positions.json","r") as f:
        pos_raw = json.load(f)["chart_data"]
    with open("data/funding_rate.json","r") as f:
        fr_raw = json.load(f)["chart_data"]
    with open("data/slippage.json","r") as f:
        slip_raw = json.load(f)["chart_data"]
    with open("data/prices.json","r") as f:
        price_raw = json.load(f)["chart_data"]

    # — 2) Build DataFrames —
    pos = pd.DataFrame(pos_raw)
    pos["time"] = pd.to_datetime(pos["time"])
    pos = pos.rename(columns={
        "daily_ntl":     "exposure",   # signed notional
        "daily_ntl_abs": "turnover"    # gross notional proxy
    })

    fr = pd.DataFrame(fr_raw)
    fr["time"] = pd.to_datetime(fr["time"])
    fr = fr.rename(columns={"sum_funding":"funding_rate"})

    slip = pd.DataFrame(slip_raw)
    slip["time"] = pd.to_datetime(slip["time"])
    slip = slip.rename(columns={"slippage_pct":"slippage_pct"})

    prices = pd.DataFrame(price_raw)
    prices["time"] = pd.to_datetime(prices["time"])
    prices = prices.rename(columns={"close":"price"})
    prices = prices.sort_values(["coin","time"])
    prices["return"] = prices.groupby("coin")["price"].pct_change()

    # — 3) Merge on coin×day —
    df = (
        pos
        .merge(fr,   on=["time","coin"], how="inner")
        .merge(slip, on=["time","coin"], how="left")
        .merge(prices[["time","coin","return"]], on=["time","coin"], how="left")
    )

    # — 4) Apply leverage —
    df["turnover_lv"]     = df["turnover"] * LEVERAGE
    df["exposure_prev"]   = df.groupby("coin")["exposure"].shift(1)
    df["exposure_prev_lv"]= df["exposure_prev"] * LEVERAGE

    # — 5) Compute PnL components —
    df["funding_usd"]  = df["turnover_lv"] * df["funding_rate"]
    df["slippage_usd"] = df["turnover_lv"] * df["slippage_pct"]
    df["market_pnl"]   = df["exposure_prev_lv"] * df["return"]

    # — 6) Daily aggregation —
    daily = (
        df.groupby("time", as_index=False)
          .agg({
            "funding_usd":  "sum",
            "slippage_usd": "sum",
            "market_pnl":   "sum"
          })
          .sort_values("time")
          .fillna(0)
    )

    # — 7) Build cumulative curves —
    daily["cum_funding"]  = daily["funding_usd"].cumsum()
    daily["cum_slippage"] = daily["slippage_usd"].cumsum()
    daily["cum_market"]   = daily["market_pnl"].cumsum()

    # actual = funding − slippage + market
    daily["cum_actual"] = (
        daily["cum_funding"]
      - daily["cum_slippage"]
      + daily["cum_market"]
    )
    # ideal = funding only
    daily["cum_ideal"] = daily["cum_funding"]

    return df, daily

df, daily = load_and_prepare()

# — Sample coin‑level data —
st.subheader("🚀 Sample Coin‑Day Data")
st.dataframe(df.head(50), use_container_width=True)

# — Equity curves plot —
st.subheader("📈 Cumulative PnL: Actual vs Funding‑Only")
eq = daily[["time","cum_actual","cum_ideal"]].melt(
    id_vars="time",
    value_vars=["cum_actual","cum_ideal"],
    var_name="strategy",
    value_name="cumulative_pnl"
)
strategy_labels = {
    "cum_actual": "Actual (funding−slippage + market PnL)",
    "cum_ideal":  "Ideal Funding‑Only"
}
eq["strategy"] = eq["strategy"].map(strategy_labels)

fig = px.line(
    eq, x="time", y="cumulative_pnl", color="strategy",
    labels={
      "time":"Date",
      "cumulative_pnl":"Cumulative PnL (USD)",
      "strategy":"Strategy"
    },
    title=f"HLP Backtest with {LEVERAGE}× Leverage"
)
fig.update_layout(legend=dict(orientation="h", y=1.02, x=1))
st.plotly_chart(fig, use_container_width=True)

# — Summary table —
st.subheader("📊 Backtest Summary")
summary = {
    "Total Funding Income":          daily["funding_usd"].sum(),
    "Total Slippage Cost":           daily["slippage_usd"].sum(),
    "Total Market PnL":              daily["market_pnl"].sum(),
    f"Final Actual PnL ({LEVERAGE}×)": daily["cum_actual"].iloc[-1],
    "Final Ideal Funding‑Only":      daily["cum_ideal"].iloc[-1],
}
st.table(pd.DataFrame.from_dict(summary, orient="index", columns=["USD"]))

st.markdown("""
**Notes**  
- **Funding** = `turnover × leverage × funding_rate`  
- **Slippage** = `turnover × leverage × slippage_pct`  
- **Market PnL** = `prior-day exposure × leverage × daily_return`
""")


# Loads & merges your four data sources (positions, funding rates, slippage rates, and prices).

# Computes day‑by‑day PnL components:

# Funding fee income

# Slippage cost

# Market PnL (via prior‐day exposure × return)

# Aggregates them into a daily timeseries and builds cumulative curves.

# Plots (via Plotly Express) the “Actual” equity curve vs a “Funding‑Only” baseline.

# Shows a summary table of total dollar PnL by component and final strategy returns.

# Once you confirm it works with your real JSONs, we can add:

# Fees earned (if separate)

# Turnover or bid‑ask spread charts

# Sharpe / drawdown metrics
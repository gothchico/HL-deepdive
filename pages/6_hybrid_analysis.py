import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots
import requests

# st.title("HLP Liquidator PnL Reinvestment & Effective Leverage")

# --- Shared Data Loading (copied from main script) ---
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

@st.cache_data
def load_data():
    pnl = pd.DataFrame(json.load(open("data/hlp_liquidator_pnl"))["chart_data"])
    daily_liq = pd.DataFrame(json.load(open("data/daily_notional_liquidated_by_coin.json"))["chart_data"])
    cum_liq = pd.DataFrame(json.load(open("data/cumulative_liquidated_notional"))["chart_data"])
    assets = pd.DataFrame(json.load(open("data/asset_ctxs.json"))["chart_data"])
    hlp = pd.DataFrame(json.load(open("data/hlp_positions.json"))["chart_data"])
    return pnl, daily_liq, cum_liq, assets, hlp

pnl, daily_liq, cum_liq, assets, hlp = load_data()

# --- Data Preparation (copied from main script) ---
pnl["time"] = pd.to_datetime(pnl["time"])
daily_liq["time"] = pd.to_datetime(daily_liq["time"])
cum_liq["time"] = pd.to_datetime(cum_liq["time"])
assets["time"] = pd.to_datetime(assets["time"])
hlp["time"] = pd.to_datetime(hlp["time"])

pnl = pnl.rename(columns={"total_pnl": "vault_daily_pnl"})
daily_notional = hlp.groupby("time")["daily_ntl_abs"].sum().reset_index(name="vault_daily_notional")
daily = pnl.merge(daily_notional, on="time", how="left").sort_values("time").fillna(0)
daily["vault_cum_pnl"] = daily["vault_daily_pnl"].cumsum()
daily["vault_cum_notional"] = daily["vault_daily_notional"].cumsum()
daily["pnl_per_m_notional"] = daily["vault_daily_pnl"] / (daily["vault_daily_notional"] / 1e6).replace(0, pd.NA)

# --- Calculate Modeled Liquidator PnL ---
daily_liq["maintenance_margin_rate"] = daily_liq["coin"].map(coin_to_maintenance_margin)
daily_liq["liquidator_daily_pnl"] = daily_liq["daily_notional_liquidated"] * daily_liq["maintenance_margin_rate"]
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

# --- Hybrid Analysis: Reinvestment Simulation (B) and Effective Leverage (C) ---

st.markdown("""
### 

#### Reinvestment Simulation
We simulate a scenario where, at each day, the liquidator PnL is added to the vault's equity base, and market making returns are compounded on this growing base. This models the feedback loop where liquidation profits are recycled into market making, unique to Hyperliquid's HLP structure.

- **Pure Market Making:** Cumulative PnL from market making, assuming a fixed notional base.
- **Hybrid (Reinvested) Market Making:** Cumulative PnL from market making, where the notional base grows as liquidator PnL is added daily.
- **Interpretation:** The difference between these lines shows the compounding effect of reinvesting liquidation profits.

---
""")

# --- B. Reinvestment Simulation ---
# Simulate hybrid compounding of liquidator PnL into market making
initial_equity = 0  # or set to starting vault equity if known
vault_equity = [initial_equity]
mm_cum_pnl_hybrid = [0]
for i in range(1, len(daily)):
    prev_equity = vault_equity[-1]
    liq_pnl = daily['liquidator_daily_pnl'].iloc[i]
    mm_pnl = daily['mm_daily_pnl'].iloc[i]
    mm_return = mm_pnl / prev_equity if prev_equity != 0 else 0
    new_equity = prev_equity + liq_pnl
    mm_pnl_today = mm_return * new_equity
    mm_cum_pnl_hybrid.append(mm_cum_pnl_hybrid[-1] + mm_pnl_today)
    vault_equity.append(new_equity + mm_pnl_today)

# --- Plot: Pure vs Hybrid Market Making ---
fig_hybrid = go.Figure()
fig_hybrid.add_trace(go.Scatter(x=daily['time'], y=daily['mm_cum_pnl'], name='Cumulative Market Making PnL (Pure)', line=dict(color='blue', dash='dot')))
fig_hybrid.add_trace(go.Scatter(x=daily['time'], y=mm_cum_pnl_hybrid, name='Cumulative Market Making PnL (Hybrid/Reinvested)', line=dict(color='purple')))
fig_hybrid.add_trace(go.Scatter(x=daily['time'], y=daily['liquidator_cum_pnl'], name='Cumulative Liquidator PnL', line=dict(color='orange')))
fig_hybrid.add_trace(go.Scatter(x=daily['time'], y=daily['total_vault_cum_pnl'], name='Cumulative Total Vault PnL', line=dict(color='green', dash='dash')))
fig_hybrid.update_layout(title='Hybrid PnL: Reinvesting Liquidator Profits into Market Making', xaxis_title='Date', yaxis_title='PnL (USD)', hovermode='x unified', margin=dict(l=40, r=40, t=60, b=40), legend=dict(orientation='h', y=1.02, x=1, xanchor='right'))
st.plotly_chart(fig_hybrid, use_container_width=True)

st.markdown("""
**Inference:**
- The hybrid (reinvested) line shows the compounding effect of recycling liquidation profits into market making.
- If the hybrid line outperforms the pure line, it demonstrates the benefit of HLP's structure for community PnL growth.
- The gap between the lines quantifies the synergy between risk management (liquidations) and active trading (market making).
---
            """)

st.markdown("""  
            #### Effective Leverage or Risk Utilization
We track the ratio of market making notional to vault equity, showing how the vault's risk profile evolves as liquidator PnL is added to the equity base.

- **Vault Equity:** Cumulative sum of liquidator and market making PnL (i.e., total vault equity).
- **Market Making Notional:** The notional deployed for market making each day.
- **Effective Leverage:** The ratio of market making notional to vault equity.
- **Interpretation:** This shows how much risk the vault is taking relative to its capital, and how this changes as liquidation profits are recycled.
---

            """)
# --- C. Effective Leverage or Risk Utilization ---
vault_equity_series = pd.Series(vault_equity[1:], index=daily.index[1:])
daily['vault_equity_hybrid'] = vault_equity_series
if 'vault_daily_notional' in daily.columns:
    daily['effective_leverage'] = daily['vault_daily_notional'] / daily['vault_equity_hybrid']
else:
    daily['effective_leverage'] = float('nan')

# Mask out extreme/invalid values for effective leverage
leverage_mask = (daily['vault_equity_hybrid'] > 1e5)  # Only show when equity > $100k
effective_leverage_plot = daily['effective_leverage'].where(leverage_mask, np.nan)
# Smooth the effective leverage
effective_leverage_smooth = effective_leverage_plot.rolling(7, min_periods=1).mean()

fig_leverage = go.Figure()
fig_leverage.add_trace(go.Scatter(
    x=daily['time'], y=daily['vault_equity_hybrid'],
    name='Vault Equity (Hybrid)', line=dict(color='green')
))
fig_leverage.add_trace(go.Scatter(
    x=daily['time'], y=daily['vault_daily_notional'],
    name='Market Making Notional', line=dict(color='blue')
))
fig_leverage.add_trace(go.Scatter(
    x=daily['time'], y=effective_leverage_smooth,
    name='Effective Leverage (smoothed)', line=dict(color='red', dash='dot'), yaxis='y2'
))
fig_leverage.add_trace(go.Scatter(
    x=daily['time'], y=[1]*len(daily), name='Leverage = 1', line=dict(color='gray', dash='dash'), yaxis='y2', showlegend=False
))
fig_leverage.update_layout(
    title='Effective Leverage: Market Making Notional vs Vault Equity',
    xaxis_title='Date',
    yaxis=dict(title='USD'),
    yaxis2=dict(title='Effective Leverage', overlaying='y', side='right', range=[0, max(2, np.nanmax(effective_leverage_smooth))]),
    hovermode='x unified',
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(orientation='h', y=1.02, x=1, xanchor='right')
)
st.plotly_chart(fig_leverage, use_container_width=True)

st.markdown("""
**Inference:**
- Effective leverage shows how aggressively the vault is deploying capital for market making relative to its equity base.
- As liquidation profits are recycled, the vault's equity grows, and effective leverage may decrease (if notional is constant) or remain stable (if notional scales with equity).
- Monitoring this ratio helps understand the vault's risk profile and capital efficiency over time.
""")

# --- End Hybrid Analysis --- 
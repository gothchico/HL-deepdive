import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")
st.title("HLP Vault Fee-Split Simulation")

# ── 1) Load daily USD volume ──
with open("data/daily_usd_volume.json") as f:
  raw = json.load(f)["chart_data"]
vol_df = pd.DataFrame(raw)
vol_df["time"] = pd.to_datetime(vol_df["time"])
vol_df = vol_df.set_index("time").sort_index()

# ── 2) Simulate perps vs spot volumes ──
# (we'll still do the "all before 2024-04-01 is perps" logic for simulation)
cutoff = pd.Timestamp("2024-04-01")
vol_df["vol_perps"] = vol_df["daily_usd_volume"].where(vol_df.index < cutoff, 0.0)
vol_df["vol_spot"] = vol_df["daily_usd_volume"].where(vol_df.index >= cutoff, 0.0)

# ── 3) Compute 14-day rolling volume for fee tiers ──
vol_df["vol_14d"] = vol_df["vol_perps"].rolling(14).sum()

thresholds = [0, 5e6, 25e6, 100e6, 500e6, 2e9, 7e9]
taker_rates = [0.00045, 0.00040, 0.00035, 0.00030, 0.00028, 0.00026, 0.00024]
maker_rates = [0.00015, 0.00012, 0.00008, 0.00004, 0.0, 0.0, 0.0]

def pick_rate(vol14, rates):
  idx = max(i for i, thr in enumerate(thresholds) if vol14 >= thr)
  return rates[idx]

df = vol_df.dropna(subset=["vol_14d"]).copy()
df["taker_rate_perps"] = df["vol_14d"].apply(lambda v: pick_rate(v, taker_rates))
df["maker_rate_perps"] = df["vol_14d"].apply(lambda v: pick_rate(v, maker_rates))
df["taker_rate_spot"] = 0.0
df["maker_rate_spot"] = 0.0

# ── 4) Simulate fees & vault net ──
df["taker_fee_perps"] = df["vol_perps"] * df["taker_rate_perps"]
df["maker_rebate_perps"] = df["vol_perps"] * df["maker_rate_perps"]
df["taker_fee_spot"] = df["vol_spot"] * df["taker_rate_spot"]
df["maker_rebate_spot"] = df["vol_spot"] * df["maker_rate_spot"]

df["total_taker"] = df["taker_fee_perps"] + df["taker_fee_spot"]
df["total_maker"] = df["maker_rebate_perps"] + df["maker_rebate_spot"]
df["vault_net"] = df["total_taker"] - df["total_maker"]

# ── 5) Plot simulated Taker vs Maker vs Vault ──
st.subheader("Simulated Daily Fee Splits")
fig1 = px.line(
  df.reset_index(),
  x="time",
  y=["total_taker", "total_maker", "vault_net"],
  labels={"variable": "Metric", "value": "USD", "time": "Date"},
  title="Simulated: Taker Fees, Maker Rebates & Vault Net"
)
st.plotly_chart(fig1, use_container_width=True)

# ── 6) Load actual Hypurrscan fee breakdown ──
fees = pd.DataFrame(json.load(open("data/fees.json")))
fees["time"] = pd.to_datetime(fees["time"], unit="s").dt.normalize()
fees["total_fees_usd"] = fees["total_fees"] / 1e6
fees["spot_fees_usd"] = fees["total_spot_fees"] / 1e6
fees["perps_fees_usd"] = fees["total_fees_usd"] - fees["spot_fees_usd"]

# ── 7) Plot actual Perps vs Spot vs Total fees ──
st.subheader("Actual Fees from Hypurrscan")
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

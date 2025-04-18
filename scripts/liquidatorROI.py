# Liquidator ROI & Risk. To reconstruct the liquidator’s PnL and risk metrics, here’s what I’ll need:

# 2️⃣ Liquidator ROI & Risk Audit
# A. Data inputs
# Daily Liquidation Events
# A JSON or CSV with rows like:

# csv
# Copy
# date,coin,collateral_seized_usd,margin_returned_usd,liquidation_fee_usd
# 2023-06-13,ETH,120000,100000,2000
# 2023-06-13,BTC,300000,290000,5000
# …
# collateral_seized_usd: value of collateral liquidated

# margin_returned_usd: value paid back to the vault/user

# liquidation_fee_usd: net fee income to the liquidator

# (Optional) Funding & Price Data
# — if you want side‑by‑side comparisons with funding income or market moves.

# B. What we’ll build
# Daily Liquidator PnL Series

# daily_pnl
#   
# =
#   
# ∑
# (
# collateral_seized
#   
# −
#   
# margin_returned
# )
#   
# +
#   
# ∑
# liquidation_fee
# daily_pnl=∑(collateral_seized−margin_returned)+∑liquidation_fee
# Cumulative ROI Curve
# – Plot cumulative liquidator PnL over time.

# Risk Metrics
# – Compute daily PnL volatility, max drawdown, VaR.

# Correlation with Vault Drawdowns
# – Overlay liquidations vs vault NAV drops to see timing/impact.

# C. Next step
# Could you supply a file like data/liquidations.json (or CSV) with at least those four columns (date, coin, collateral_seized_usd, margin_returned_usd, liquidation_fee_usd)? Once I have that, I’ll scaffold a Streamlit + Plotly‐Express app to:

# Load & merge

# Compute daily and cumulative PnL

# Display interactive charts and summary tables


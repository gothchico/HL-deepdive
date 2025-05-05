import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io

# Set page config
st.set_page_config(page_title="Vault Analysis", layout="wide")
# st.title("Vault Analysis")

# Load liquidation PnL data
@st.cache_data
def load_liquidation_pnl():
    with open('data/hlp_liquidator_pnl', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['chart_data'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

# Load and aggregate positions data
@st.cache_data
def load_positions():
    with open('data/hlp_positions.json', 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'chart_data' in data:
        df = pd.DataFrame(data['chart_data'])
    else:
        df = pd.DataFrame(data)
    if 'time' not in df.columns:
        st.error("No 'time' column found in the positions data.")
        return None
    df['time'] = pd.to_datetime(df['time'])
    # Aggregate by time (sum daily_ntl for each day)
    agg_df = df.groupby('time', as_index=True).agg({'daily_ntl': 'sum'})
    return agg_df

# Calculate drawdown
def calculate_drawdown(series):
    rolling_max = series.expanding().max()
    drawdown = (series - rolling_max) / rolling_max * 100
    return drawdown

# Calculate rolling volatility
def calculate_volatility(series, window=7):
    returns = series.pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252) * 100
    return volatility

try:
    pnl_df = load_liquidation_pnl()
    positions_df = load_positions()
    if positions_df is None:
        st.error("Failed to load positions data")
        st.stop()
    # Align positions_df to pnl_df index
    aligned_positions = positions_df.reindex(pnl_df.index)
    # Calculate metrics
    cumulative_pnl = pnl_df['total_pnl'].cumsum()
    drawdown = calculate_drawdown(cumulative_pnl)
    volatility = calculate_volatility(cumulative_pnl)
    # Display summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Drawdown", f"{drawdown.min():.2f}%")
    with col2:
        st.metric("Average Volatility", f"{volatility.mean():.2f}%")
    with col3:
        st.metric("Total PnL", f"${cumulative_pnl.iloc[-1]:,.2f}")

    # --- Markdown analysis for Daily Notional ---
    st.markdown("""
    **Daily Notional Analysis**  
    The chart below shows the aggregated daily notional traded. Spikes may indicate periods of high trading activity or volatility. Sustained low values may reflect reduced market participation or risk-off behavior.
    """)

    # --- Daily Notional Bar Chart ---
    # (Separate from main chart)
    if 'daily_ntl' in aligned_positions.columns:
        fig_notional = go.Figure()
        fig_notional.add_trace(
            go.Bar(
                x=aligned_positions.index,
                y=aligned_positions['daily_ntl'],
                name="Daily Notional",
                marker_color=np.where(aligned_positions['daily_ntl'] >= 0, 'green', 'red')
            )
        )
        fig_notional.update_layout(
            title="Daily Notional Traded",
            xaxis_title="Date",
            yaxis_title="Daily Notional",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_notional, use_container_width=True)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    import traceback
    st.write("Traceback:", traceback.format_exc())

# --- Drawdown and Volatility (PnL Risk Metrics) at the bottom ---
try:
    with open('data/hlp_liquidator_pnl', 'r') as f:
        hlp_liquidator_pnl_data = json.load(f)
    hlp_liquidator_pnl_df = pd.DataFrame(hlp_liquidator_pnl_data['chart_data'])
    hlp_liquidator_pnl_df['time'] = pd.to_datetime(hlp_liquidator_pnl_df['time'])
    hlp_liquidator_pnl_df.set_index('time', inplace=True)
    hlp_cum_pnl = hlp_liquidator_pnl_df['total_pnl'].cumsum()
    hlp_drawdown = calculate_drawdown(hlp_cum_pnl)
    hlp_volatility = calculate_volatility(hlp_cum_pnl)
    # --- Markdown analysis for PnL Risk Metrics ---
    st.markdown("""
    **PnL Risk Metrics**  
    The following chart shows the drawdown percentage and 7-day rolling volatility for the HLP liquidator PnL. Persistent drawdowns or high volatility can signal increased risk or unstable performance. Use this to monitor risk and adapt strategies as needed.
    """)
    st.markdown("## Drawdown and Volatility")
    st.markdown("**PnL Risk Metrics**")
    hlp_fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Drawdown % (left y-axis)
    hlp_fig.add_trace(
        go.Scatter(
            x=hlp_drawdown.index,
            y=hlp_drawdown,
            name="Drawdown %",
            line=dict(color='crimson')
        ),
        secondary_y=False
    )
    # Cumulative PnL (right y-axis)
    hlp_fig.add_trace(
        go.Scatter(
            x=hlp_cum_pnl.index,
            y=hlp_cum_pnl,
            name="Cumulative PnL",
            line=dict(color='orange')
        ),
        secondary_y=True
    )
    # Volatility (overlay, hidden y-axis)
    hlp_fig.add_trace(
        go.Scatter(
            x=hlp_volatility.index,
            y=hlp_volatility,
            name="Volatility (7d rolling)",
            line=dict(color='green'),
            yaxis='y3'
        )
    )
    # Add a hidden y3 axis for volatility
    hlp_fig.update_layout(
        showlegend=True,
        height=600,
        yaxis3=dict(
            overlaying='y',
            visible=False
        )
    )
    hlp_fig.update_yaxes(title_text="Drawdown %", secondary_y=False)
    hlp_fig.update_yaxes(title_text="Cumulative PnL", secondary_y=True)
    st.plotly_chart(hlp_fig, use_container_width=True)
except Exception as e:
    st.error(f"Error loading hlp_liquidator_pnl.json: {str(e)}")
    import traceback
    st.write("Traceback:", traceback.format_exc()) 
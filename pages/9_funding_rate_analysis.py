import time  # to simulate a real time data, time loop
from datetime import datetime, timedelta
import numpy as np 
import pandas as pd  
import plotly.express as px  
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st  # 🎈 data web app development
import requests
from hyperliquid.info import Info
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Funding Rate Analysis",
    page_icon="💯",
    layout="wide",
)

def datetime_to_milliseconds(dt):
    return int(dt.timestamp() * 1000)

def convert_to_numeric(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def calculate_funding_pnl(funding_rate, notional, leverage, days):
    """Calculate PnL from funding rate arbitrage"""
    return funding_rate * notional * leverage * days

# Dashboard title and description
st.title("Funding Rate Analysis")
st.markdown("""
This dashboard provides comprehensive analysis of perpetual futures funding rates, including:
- Current and historical funding rates
- Funding rate arbitrage opportunities
- Market statistics and correlations
""")

# Initialize Hyperliquid API
info = Info(skip_ws=False)

# Fetch market data
res = info.meta_and_asset_ctxs()
universe_data = res[0]['universe']
OI_data = res[1]

# Combine and process data
data = [a | b for a, b in zip(universe_data, OI_data)]
df1 = pd.json_normalize(data)

# Convert numeric columns
numeric_columns = ['funding', 'openInterest', 'prevDayPx', 'dayNtlVlm', 'premium', 'oraclePx', 'markPx', 'midPx', 'dayBaseVlm']
for col in numeric_columns:
    if col in df1.columns:
        df1[col] = df1[col].apply(convert_to_numeric)

available_coins = pd.unique(df1["name"])

# Sidebar for controls
st.sidebar.header("Controls")
token = st.sidebar.selectbox("Select token", available_coins)

# Date range selection
start_date = datetime(2022, 1, 1).date()
end_date = datetime.now().date()
max_days_range = timedelta(days=90)

if "slider_range" not in st.session_state:
    st.session_state.slider_range = (end_date - max_days_range, end_date)

slider_range = st.sidebar.slider(
    "Select a date range (max 3 months)",
    min_value=start_date,
    max_value=end_date,
    value=st.session_state.slider_range,
    step=timedelta(days=1),
)

selected_start_date, selected_end_date = slider_range

if (selected_end_date - selected_start_date) > max_days_range:
    st.warning("Please select a date range within the maximum allowed span of 3 months. Showing last 3 months from the selected end date")
    selected_start_date = selected_end_date - max_days_range
    st.session_state.slider_range = (selected_start_date, selected_end_date)

# Fetch funding rate history
fr_response = info.funding_history(
    name=token, 
    startTime=datetime_to_milliseconds(datetime.combine(selected_start_date, datetime.min.time())), 
    endTime=datetime_to_milliseconds(datetime.combine(selected_end_date, datetime.max.time()))
)

df2 = pd.json_normalize(fr_response)

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Funding Analysis", "Market Overview", "Arbitrage Analysis"])

with tab1:
    st.header("Funding Rate Analysis")
    if {"time", "fundingRate", "premium"}.issubset(df2.columns):
        if df2["time"].dtype != "datetime64[ns]":
            df2["time"] = pd.to_datetime(df2["time"], unit="ms")
        
        # Convert funding rate and premium to numeric
        df2["fundingRate"] = df2["fundingRate"].apply(convert_to_numeric)
        df2["premium"] = df2["premium"].apply(convert_to_numeric)
        
        # Create subplot for funding rate and premium
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=df2["time"], y=df2["fundingRate"], name="Funding Rate"),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=df2["time"], y=df2["premium"], name="Premium"),
            secondary_y=True,
        )
        
        fig.update_layout(
            title=f'{token} Funding Rate vs Premium',
            xaxis_title="Time",
            legend_title="Metrics",
            height=600
        )
        
        fig.update_yaxes(title_text="Funding Rate", secondary_y=False)
        fig.update_yaxes(title_text="Premium", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional funding rate statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Funding Rate", f"{df2['fundingRate'].mean():.6f}")
        with col2:
            st.metric("Max Funding Rate", f"{df2['fundingRate'].max():.6f}")
        with col3:
            st.metric("Min Funding Rate", f"{df2['fundingRate'].min():.6f}")

with tab2:
    st.header("Market Overview")
    
    # Calculate and display market statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Selected Token Statistics")
        token_stats = df1[df1['name'] == token].iloc[0]
        
        st.metric("Current Funding Rate", f"{float(token_stats.get('funding', 0)):.6f}")
        st.metric("Open Interest", f"${float(token_stats.get('openInterest', 0)):,.2f}")
        st.metric("24h Volume", f"${float(token_stats.get('dayNtlVlm', 0)):,.2f}")
        st.metric("Mark Price", f"${float(token_stats.get('markPx', 0)):,.2f}")
    
    with col2:
        st.subheader("Market Overview")
        # Sort by funding rate
        funding_data = df1.sort_values('funding', ascending=False)
        
        fig = px.bar(
            funding_data.head(10),
            x='name',
            y='funding',
            title='Top 10 Tokens by Current Funding Rate'
        )
        
        fig.update_layout(
            xaxis_title="Token",
            yaxis_title="Funding Rate",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Funding Rate Arbitrage Analysis")
    
    # Arbitrage parameters
    col1, col2 = st.columns(2)
    
    with col1:
        notional = st.number_input("Notional Value (USD)", min_value=1000, value=10000, step=1000)
        leverage = st.selectbox("Leverage", options=[1, 2, 3, 4, 5, 10, 20, 50, 100])
        days = st.number_input("Holding Period (Days)", min_value=1, value=7, step=1)
    
    # Calculate potential PnL for each token
    df1['potential_pnl'] = df1.apply(
        lambda x: calculate_funding_pnl(
            float(x.get('funding', 0)),
            notional,
            leverage,
            days
        ),
        axis=1
    )
    
    # Calculate divergence ratio: (markPx - oraclePx) / oraclePx
    df1['divergence_ratio'] = df1.apply(
        lambda x: (float(x.get('markPx', 0)) - float(x.get('oraclePx', 0))) / float(x.get('oraclePx', 1)) if float(x.get('oraclePx', 0)) != 0 else 0,
        axis=1
    )
    
    # Sort by potential PnL
    arbitrage_opportunities = df1.sort_values('potential_pnl', ascending=False)

    # --- Interactive Filtering ---
    st.subheader("Filter Opportunities")
    min_funding, max_funding = float(arbitrage_opportunities['funding'].min()), float(arbitrage_opportunities['funding'].max())
    min_pnl, max_pnl = float(arbitrage_opportunities['potential_pnl'].min()), float(arbitrage_opportunities['potential_pnl'].max())
    min_div, max_div = float(arbitrage_opportunities['divergence_ratio'].min()), float(arbitrage_opportunities['divergence_ratio'].max())
    
    funding_range = st.slider("Funding Rate Range", min_funding, max_funding, (min_funding, max_funding), step=(max_funding-min_funding)/100 or 1e-6, format="%.6f")
    pnl_range = st.slider("Potential PnL Range", min_pnl, max_pnl, (min_pnl, max_pnl), step=(max_pnl-min_pnl)/100 or 1e-2, format="%.2f")
    div_range = st.slider("Divergence Ratio Range", float(min_div), float(max_div), (float(min_div), float(max_div)), step=(max_div-min_div)/100 or 1e-6, format="%.4f")
    
    filtered = arbitrage_opportunities[
        (arbitrage_opportunities['funding'] >= funding_range[0]) &
        (arbitrage_opportunities['funding'] <= funding_range[1]) &
        (arbitrage_opportunities['potential_pnl'] >= pnl_range[0]) &
        (arbitrage_opportunities['potential_pnl'] <= pnl_range[1]) &
        (arbitrage_opportunities['divergence_ratio'] >= div_range[0]) &
        (arbitrage_opportunities['divergence_ratio'] <= div_range[1])
    ]

    with col2:
        st.subheader("Top Arbitrage Opportunities")
        st.dataframe(
            filtered[['name', 'funding', 'potential_pnl', 'divergence_ratio']].head(10),
            column_config={
                "name": "Token",
                "funding": st.column_config.NumberColumn("Current Funding Rate", format="%.6f"),
                "potential_pnl": st.column_config.NumberColumn("Potential PnL (USD)", format="$%.2f"),
                "divergence_ratio": st.column_config.NumberColumn("Divergence Ratio", format=".4%")
            }
        )

    # --- Enhanced scatter plot: Funding Rate vs Potential PnL ---
    st.subheader("Funding Rate vs Potential PnL")
    # Color by sign, size by abs(PnL), alpha for overlap
    scatter_colors = np.where(filtered['potential_pnl'] > 0, 'green', np.where(filtered['potential_pnl'] < 0, 'red', 'gray'))
    marker_sizes = 10 + 20 * (np.abs(filtered['potential_pnl']) / (np.abs(filtered['potential_pnl']).max() or 1))
    
    # Top 5 positive and negative PnL for static labels
    top_pos = filtered.nlargest(5, 'potential_pnl')
    top_neg = filtered.nsmallest(5, 'potential_pnl')
    label_mask = filtered.index.isin(top_pos.index) | filtered.index.isin(top_neg.index)
    
    # Regression line
    if len(filtered) > 1:
        m, b = np.polyfit(filtered['funding'], filtered['potential_pnl'], 1)
        reg_x = np.linspace(filtered['funding'].min(), filtered['funding'].max(), 100)
        reg_y = m * reg_x + b
    else:
        reg_x, reg_y = [], []

    fig = go.Figure()
    # Main scatter
    fig.add_trace(go.Scatter(
        x=filtered['funding'],
        y=filtered['potential_pnl'],
        mode='markers',
        marker=dict(color=scatter_colors, size=marker_sizes, opacity=0.6, line=dict(width=1, color='white')),
        hovertemplate=(
            'Token: %{text}<br>Funding Rate: %{x:.6f}<br>Potential PnL: %{y:.2f}<br>' +
            'Divergence Ratio: %{customdata:.4%}'
        ),
        text=filtered['name'],
        customdata=filtered['divergence_ratio'],
        showlegend=False
    ))
    # Static labels for top 5 positive/negative
    fig.add_trace(go.Scatter(
        x=filtered[label_mask]['funding'],
        y=filtered[label_mask]['potential_pnl'],
        mode='text',
        text=filtered[label_mask]['name'],
        textposition='top center',
        showlegend=False
    ))
    # Regression line
    if len(reg_x) > 0:
        fig.add_trace(go.Scatter(
            x=reg_x, y=reg_y, mode='lines', line=dict(color='blue', dash='dash'),
            name='Regression Line', showlegend=True
        ))
    # Center axes at (0,0)
    fig.update_layout(
        title=f'Funding Rate vs Potential PnL (Notional: ${notional:,.0f}, Leverage: {leverage}x)',
        xaxis_title="Funding Rate",
        yaxis_title="Potential PnL (USD)",
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- New plot: PnL vs Divergence Ratio ---
    st.subheader("Potential PnL vs Divergence Ratio")
    # Regression line for this plot
    if len(filtered) > 1:
        m2, b2 = np.polyfit(filtered['divergence_ratio'], filtered['potential_pnl'], 1)
        reg_x2 = np.linspace(filtered['divergence_ratio'].min(), filtered['divergence_ratio'].max(), 100)
        reg_y2 = m2 * reg_x2 + b2
    else:
        reg_x2, reg_y2 = [], []
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=filtered['divergence_ratio'],
        y=filtered['potential_pnl'],
        mode='markers',
        marker=dict(color=scatter_colors, size=marker_sizes, opacity=0.6, line=dict(width=1, color='white')),
        hovertemplate=(
            'Token: %{text}<br>Divergence Ratio: %{x:.4%}<br>Potential PnL: %{y:.2f}<br>' +
            'Funding Rate: %{customdata:.6f}'
        ),
        text=filtered['name'],
        customdata=filtered['funding'],
        showlegend=False
    ))
    # Static labels for top 5 positive/negative
    fig2.add_trace(go.Scatter(
        x=filtered[label_mask]['divergence_ratio'],
        y=filtered[label_mask]['potential_pnl'],
        mode='text',
        text=filtered[label_mask]['name'],
        textposition='top center',
        showlegend=False
    ))
    # Regression line
    if len(reg_x2) > 0:
        fig2.add_trace(go.Scatter(
            x=reg_x2, y=reg_y2, mode='lines', line=dict(color='blue', dash='dash'),
            name='Regression Line', showlegend=True
        ))
    fig2.update_layout(
        title=f'Potential PnL vs Divergence Ratio (Notional: ${notional:,.0f}, Leverage: {leverage}x)',
        xaxis_title="Divergence Ratio (Perp - Spot) / Spot",
        yaxis_title="Potential PnL (USD)",
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Risk metrics
    st.subheader("Risk Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Max Drawdown", f"${abs(filtered['potential_pnl'].min()):,.2f}")
    with col2:
        st.metric("Average PnL", f"${filtered['potential_pnl'].mean():,.2f}")
    with col3:
        st.metric("PnL Volatility", f"${filtered['potential_pnl'].std():,.2f}")

# Auto-refresh functionality
count = st_autorefresh(
    interval=5 * 1000,  # 5 seconds
    key="data_refresh",
)

# Footer
st.markdown("---")
st.markdown("""
### Data Sources
- Funding rates and market data: Hyperliquid API
- Real-time updates every 5 seconds
""")

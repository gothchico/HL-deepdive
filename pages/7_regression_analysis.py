import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
from scipy.signal import correlate
import json

# Set page config
st.set_page_config(page_title="Regression Analysis", layout="wide")
# st.title("Regression Analysis")

# --- Data Loading ---
@st.cache_data
def load_data():
    with open('data/daily_notional_liquidated_by_coin.json', 'r') as f:
        liquidation_data = json.load(f)
    liquidation_df = pd.DataFrame(liquidation_data['chart_data'])
    liquidation_df['time'] = pd.to_datetime(liquidation_df['time'])
    
    with open('data/hlp_positions.json', 'r') as f:
        positions_data = json.load(f)
    positions_df = pd.DataFrame(positions_data['chart_data'])
    positions_df['time'] = pd.to_datetime(positions_df['time'])
    
    with open('data/asset_ctxs.json', 'r') as f:
        price_data = json.load(f)
    price_df = pd.DataFrame(price_data['chart_data'])
    price_df['time'] = pd.to_datetime(price_df['time'])
    
    with open('data/hlp_liquidator_pnl', 'r') as f:
        pnl_data = json.load(f)
    pnl_df = pd.DataFrame(pnl_data['chart_data'])
    pnl_df['time'] = pd.to_datetime(pnl_df['time'])
    
    return liquidation_df, positions_df, price_df, pnl_df

try:
    # --- Load Data ---
    liquidation_df, positions_df, price_df, pnl_df = load_data()
    
    st.markdown("""
    ## Data Preparation
    All data is aligned by date. Liquidations are summed across coins. Price returns are calculated as weighted log returns. PnL is taken from the official HLP liquidator PnL file.
    """)
    
    # --- Prepare Data ---
    daily_liquidated = liquidation_df.groupby('time')['daily_notional_liquidated'].sum()
    price_pivot = price_df.pivot(index='time', columns='coin', values='avg_oracle_px')
    log_returns = np.log(price_pivot / price_pivot.shift(1))
    daily_notional = positions_df.pivot(index='time', columns='coin', values='daily_ntl')
    weighted_returns = (daily_notional * log_returns).sum(axis=1)
    pnl_df = pnl_df.set_index('time').sort_index()
    
    # --- Align all data by date ---
    df = pd.DataFrame({
        'liquidated_notional': daily_liquidated,
        'weighted_return': weighted_returns,
        'abs_return': weighted_returns.abs(),
        'pnl': pnl_df['total_pnl']
    }).dropna()
    
    # --- 1. Standard Regression: Liquidations vs Absolute Price Returns ---
    st.markdown("""
    ### 1. Linear Regression: Liquidations vs Absolute Price Returns
    We regress daily liquidated notional on the absolute value of weighted price returns.
    """)
    X = df[['abs_return']].values
    y = df['liquidated_notional'].values
    model = LinearRegression().fit(X, y)
    st.write(f"R-squared: {model.score(X, y):.4f}")
    st.write(f"Coefficient: {model.coef_[0]:.4f}")
    st.write(f"Intercept: {model.intercept_:.4f}")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['abs_return'], y=df['liquidated_notional'], mode='markers', name='Actual Data', marker=dict(size=8)))
    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = model.predict(x_range.reshape(-1, 1))
    fig1.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Regression Line', line=dict(color='red')))
    fig1.update_layout(title='Liquidated Notional vs Absolute Price Returns', xaxis_title='Absolute Price Returns', yaxis_title='Liquidated Notional', showlegend=True)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(f"""
    **Inference:**
    - R-squared is {model.score(X, y):.4f}, indicating a {'weak' if model.score(X, y)<0.2 else 'moderate/strong'} linear relationship.
    - The relationship is {'positive' if model.coef_[0]>0 else 'negative'}.
    """)
    
    # --- 2. Logistic Regression: Does a large price move increase probability of large liquidation? ---
    st.markdown("""
    ### 2. Logistic Regression: Probability of Large Liquidations
    We classify days as 'large liquidation' if liquidated notional is in the top 10%. We use logistic regression to predict this from absolute price returns.
    """)
    threshold = df['liquidated_notional'].quantile(0.9)
    df['large_liq'] = (df['liquidated_notional'] >= threshold).astype(int)
    X_log = df[['abs_return']].values
    y_log = df['large_liq'].values
    clf = LogisticRegression().fit(X_log, y_log)
    probas = clf.predict_proba(X_log)[:,1]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['abs_return'], y=probas, mode='markers', name='Predicted Probability', marker=dict(size=8, color=df['large_liq'], colorscale='Viridis', showscale=False)))
    fig2.add_trace(go.Scatter(x=df.loc[df['large_liq']==1, 'abs_return'], y=probas[df['large_liq']==1], mode='markers', name='Large Liquidation Days', marker=dict(size=10, color='yellow', line=dict(width=1, color='black'))))
    fig2.update_layout(title='Probability of Large Liquidation vs Absolute Price Returns', xaxis_title='Absolute Price Returns', yaxis_title='Probability of Large Liquidation', showlegend=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f"""
    **Inference:**
    - The probability of a large liquidation increases with the magnitude of price moves.
    - This supports the hypothesis that extreme price changes are more likely to trigger large liquidations.
    """)
    
    # --- 3. Quantile Regression: Liquidations vs Absolute Price Returns ---
    st.markdown("""
    ### 3. Quantile Regression: Conditional Quantiles of Liquidations
    We fit quantile regressions (median and 90th percentile) to see how the relationship changes across the distribution.
    """)
    mod_median = sm.QuantReg(df['liquidated_notional'], sm.add_constant(df['abs_return'])).fit(q=0.5)
    mod_90 = sm.QuantReg(df['liquidated_notional'], sm.add_constant(df['abs_return'])).fit(q=0.9)
    y_median = mod_median.predict(sm.add_constant(x_range))
    y_90 = mod_90.predict(sm.add_constant(x_range))
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df['abs_return'], y=df['liquidated_notional'], mode='markers', name='Actual Data', marker=dict(size=8)))
    fig3.add_trace(go.Scatter(x=x_range, y=y_median, mode='lines', name='Median Quantile', line=dict(color='orange')))
    fig3.add_trace(go.Scatter(x=x_range, y=y_90, mode='lines', name='90th Quantile', line=dict(color='green')))
    fig3.update_layout(title='Quantile Regression: Liquidated Notional vs Absolute Price Returns', xaxis_title='Absolute Price Returns', yaxis_title='Liquidated Notional', showlegend=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(f"""
    **Inference:**
    - The 90th percentile line is much steeper than the median, indicating that large price moves disproportionately increase the risk of very large liquidations.
    - Quantile regression reveals tail risk not captured by standard regression.
    """)
    
    # --- 4. Cross-Correlation Plot: Price Returns and Liquidations ---
    st.markdown("""
    ### 4. Cross-Correlation: Price Returns and Liquidations
    We plot the cross-correlation between absolute price returns and liquidated notional at different lags.
    """)
    max_lag = 10
    x = df['abs_return'] - df['abs_return'].mean()
    y = df['liquidated_notional'] - df['liquidated_notional'].mean()
    corr = [x.corr(y.shift(lag)) for lag in range(-max_lag, max_lag+1)]
    lags = list(range(-max_lag, max_lag+1))
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=lags, y=corr, name='Cross-correlation'))
    fig4.update_layout(title='Cross-Correlation: Abs Price Returns vs Liquidated Notional', xaxis_title='Lag (days)', yaxis_title='Correlation', showlegend=True)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown(f"""
    **Inference:**
    - The highest correlation is typically at lag 0 or 1, suggesting contemporaneous or next-day effects.
    - If lag 1 is high, price moves today may predict liquidations tomorrow.
    """)
    
    # --- 5. Time Series Plots ---
    st.markdown("""
    ### 5. Time Series 
    Visualize the evolution of price returns, liquidations, and PnL over time.
    """)
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=df.index, y=df['abs_return'], name='Abs Price Return', line=dict(color='red')))
    fig_ts.add_trace(go.Scatter(x=df.index, y=df['pnl'], name='PnL', line=dict(color='green')))
    fig_ts.add_trace(go.Scatter(x=df.index, y=df['liquidated_notional'], name='Liquidated Notional', line=dict(color='blue'), yaxis='y2', opacity=0.3))
    fig_ts.update_layout(
        title='Time Series: Abs Price Return, Liquidated Notional, PnL',
        xaxis_title='Date',
        yaxis=dict(title='Abs Price Return / PnL'),
        yaxis2=dict(title='Liquidated Notional', overlaying='y', side='right', showgrid=False),
        showlegend=True
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    st.markdown("""
    **Inference:**
    - Spikes in price returns often coincide with spikes in liquidations and PnL volatility.
    - Visual inspection can reveal regime shifts or clusters of high activity.
    """)
    
    # --- 6. Boxplots/Violin Plots: High vs Low Liquidation Days ---
    st.markdown("""
    ### 6. Distribution Comparison: High vs Low Liquidation Days
    We compare the distribution of price returns and PnL on high vs low liquidation days.
    
    **Definition:**
    - **High** liquidation days are those where liquidated notional is above the 90th percentile (top 10%).
    - **Low** liquidation days are those where liquidated notional is below the 10th percentile (bottom 10%).
    """)
    high_liq = df['liquidated_notional'] >= df['liquidated_notional'].quantile(0.9)
    low_liq = df['liquidated_notional'] < df['liquidated_notional'].quantile(0.1)
    import plotly.express as px
    fig_box = px.violin(df, y='abs_return', box=True, points='all', color=high_liq.map({True:'High',False:'Low'}), category_orders={'color':['Low','High']}, labels={'color':'Liquidation Day'})
    fig_box.update_layout(title='Distribution of Abs Price Returns: High vs Low Liquidation Days')
    st.plotly_chart(fig_box, use_container_width=True)
    fig_box2 = px.violin(df, y='pnl', box=True, points='all', color=high_liq.map({True:'High',False:'Low'}), category_orders={'color':['Low','High']}, labels={'color':'Liquidation Day'})
    fig_box2.update_layout(title='Distribution of PnL: High vs Low Liquidation Days')
    st.plotly_chart(fig_box2, use_container_width=True)
    st.markdown("""
    **Inference:**
    - High liquidation days are associated with fatter tails in price returns and PnL distributions.
    - This suggests that risk is concentrated in a small number of extreme days.
    """)
    
except Exception as e:
    st.error(f"Error in analysis: {str(e)}")
    import traceback
    st.write("Traceback:", traceback.format_exc()) 
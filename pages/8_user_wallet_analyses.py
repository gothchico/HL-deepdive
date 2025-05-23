import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import numpy as np
import json

st.set_page_config(page_title="User-Wallet Analyses", page_icon="📊", layout="wide")



@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_all_data():
    endpoints = {
        'daily_volume': "https://d2v1fiwobg9w6.cloudfront.net/daily_usd_volume_by_user",
        'largest_users': "https://d2v1fiwobg9w6.cloudfront.net/largest_users_by_usd_volume",
        'liquidations': "https://d2v1fiwobg9w6.cloudfront.net/largest_liquidated_notional_by_user",
        'trade_count': "https://d2v1fiwobg9w6.cloudfront.net/largest_user_trade_count"
    }
    
    data = {}
    for key, url in endpoints.items():
        response = requests.get(url)
        if response.status_code == 200:
            if key == 'daily_volume':
                data[key] = pd.DataFrame(response.json()['chart_data'])
                data[key]['time'] = pd.to_datetime(data[key]['time'])
            else:
                data[key] = pd.DataFrame(response.json()['table_data'])
                data[key].columns = ['user', 'value']
    # Load total daily volume (all users)
    with open('data/daily_usd_volume.json', 'r') as f:
        total_vol = pd.DataFrame(json.load(f)["chart_data"])
        total_vol['time'] = pd.to_datetime(total_vol['time'])
    data['total_daily_volume'] = total_vol
    return data

def main():
    data = fetch_all_data()
    
    if not data:
        st.error("Failed to fetch data. Please try again later.")
        return
    
    # Key Metrics Overview
    st.subheader("Key Metrics Overview")
    st.markdown("""
    These metrics provide a high-level overview of trading activity on the platform:
    - **Total Users**: Number of unique traders
    - **Total Volume**: Combined trading volume in USD
    - **Total Trades**: Total number of trades executed
    - **Avg Trade Size**: Average size of each trade in USD
    """)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    metric_style = "font-size: 2.2em; font-weight: bold; max-width: 100%; overflow-wrap: anywhere; margin-top: 0.2em; margin-bottom: 0.2em; display: inline-block;"
    with col1:
        st.markdown(f"""
        <div style='text-align: center;'>
            <span style='font-size: 1.2em;'>Total Users</span><br>
            <span style='{metric_style}'>
                {len(data['daily_volume']['user'].unique())}
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='text-align: center;'>
            <span style='font-size: 1.2em;'>Total Volume (USD)</span><br>
            <span style='{metric_style}'>
                ${data['daily_volume']['daily_usd_volume'].sum():,.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style='text-align: center;'>
            <span style='font-size: 1.2em;'>Total Trades</span><br>
            <span style='{metric_style}'>
                {int(data['trade_count']['value'].sum()):,}
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_trade_size = data['daily_volume']['daily_usd_volume'].sum() / data['trade_count']['value'].sum()
        st.markdown(f"""
        <div style='text-align: center;'>
            <span style='font-size: 1.2em;'>Avg Trade Size</span><br>
            <span style='{metric_style}'>
                ${avg_trade_size:,.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(""" --------------------------------------------------------------------------------------------------- """)    
     
    # Top Users Analysis Tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Volume Analysis", "Liquidation Analysis", "Trading Behavior", "Advanced Metrics", "User Frequency"])
    
    with tab1:
        # --- Top Wallets vs Total Daily Volume ---
        st.subheader("Top Wallets vs Total Daily Volume")
        st.markdown("""
            Compare the daily trading volume of the top N wallets (by global volume or activity) or top percentile to the total daily volume across all users. Use the controls below to customize the cohort and chart type.
            """)

        col1, col2, col3, col4, col5 = st.columns([2,2,2,2,2])
        with col1:
            cohort_mode = st.radio("Cohort Mode", ["Top N", "Top Percentile"], index=0, horizontal=True, key="cohort_mode")
        with col2:
            if cohort_mode == "Top N":
                top_n = st.selectbox("Select Top N:", [10, 100, 500, 1000], index=1, key="topn_volume_compare")
            else:
                top_percentile = st.slider("Select Top Percentile:", min_value=10, max_value=100, value=10, step=1, key="top_percentile")
        with col3:
            cohort_metric = st.radio("Rank by", ["Volume", "Activity"], index=0, horizontal=True, key="metric_volume_compare")
        with col4:
            chart_type = st.radio("Chart Type", ["Line", "Bar"], index=0, horizontal=True, key="charttype_volume_compare")
        with col5:
            exclude_top3 = st.checkbox("Exclude Top 3 Wallets", value=True, key="exclude_top3_volume_compare")

        # Compute cohort users
        user_freq = data['daily_volume'].groupby('user').agg(
            total_volume=('daily_usd_volume', 'sum'),
            active_days=('time', 'nunique')
        ).reset_index()
        if cohort_metric == "Volume":
            sorted_users = user_freq.sort_values('total_volume', ascending=False)
        else:
            sorted_users = user_freq.sort_values('active_days', ascending=False)
        if cohort_mode == "Top N":
            if exclude_top3:
                cohort_users = sorted_users.iloc[3:3+top_n]['user']
            else:
                cohort_users = sorted_users.head(top_n)['user']
            cohort_label = f"Top {top_n} by {cohort_metric}{' (Excl. Top 3)' if exclude_top3 else ''}"
        else:
            n_users = int(np.ceil(len(sorted_users) * top_percentile / 100))
            if exclude_top3:
                cohort_users = sorted_users.iloc[3:3+n_users]['user']
            else:
                cohort_users = sorted_users.head(n_users)['user']
            cohort_label = f"Top {top_percentile}% by {cohort_metric} ({n_users} users){' (Excl. Top 3)' if exclude_top3 else ''}"

        topn_daily = data['daily_volume'][data['daily_volume']['user'].isin(cohort_users)].groupby('time')['daily_usd_volume'].sum().reset_index()
        total_daily = data['total_daily_volume'][['time', 'daily_usd_volume']].rename(columns={'daily_usd_volume': 'total_volume'})
        merged = pd.merge(topn_daily, total_daily, on='time', how='inner')

        # Plot
        if chart_type == "Line":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=merged['time'], y=merged['daily_usd_volume'], mode='lines', name=cohort_label))
            fig.add_trace(go.Scatter(x=merged['time'], y=merged['total_volume'], mode='lines', name='Total Volume (All Users)', line=dict(color='lightgray')))
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=merged['time'], y=merged['daily_usd_volume'], name=cohort_label))
            fig.add_trace(go.Bar(x=merged['time'], y=merged['total_volume'], name='Total Volume (All Users)', marker_color='lightgray'))
            fig.update_layout(barmode='stack')
        fig.update_layout(title=f"{cohort_label} vs Total Daily Volume", xaxis_title="Date", yaxis_title="Volume (USD)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, key=f"top_wallets_{cohort_label}_{chart_type}_{exclude_top3}")

        # --- Percentage Share Plot Controls ---
        st.markdown("""
        ### Percentage Share of Top Wallets
        Use the controls to select cohort and see their share of total daily volume.
        """)
        col1, col2, col3, col4 = st.columns([2,2,2,2])
        with col1:
            pct_cohort_mode = st.radio("Cohort Mode", ["Top N", "Top Percentile"], index=0, horizontal=True, key="pct_cohort_mode")
        with col2:
            if pct_cohort_mode == "Top N":
                pct_n = st.selectbox("Select Top N:", [10, 100, 500, 1000], index=1, key="pctn_share")
            else:
                pct_percentile = st.slider("Select Top Percentile:", min_value=10, max_value=100, value=10, step=1, key="pct_percentile")
        with col3:
            pct_metric = st.radio("Rank by", ["Volume", "Activity"], index=0, horizontal=True, key="pct_metric")
        with col4:
            exclude_top3_pct = st.checkbox("Exclude Top 3 Wallets", value=True, key="exclude_top3_pct")

        if pct_metric == "Volume":
            sorted_pct_users = user_freq.sort_values('total_volume', ascending=False)
        else:
            sorted_pct_users = user_freq.sort_values('active_days', ascending=False)
        if pct_cohort_mode == "Top N":
            if exclude_top3_pct:
                pct_users = sorted_pct_users.iloc[3:3+pct_n]['user']
            else:
                pct_users = sorted_pct_users.head(pct_n)['user']
            pct_label = f"Top {pct_n} by {pct_metric}{' (Excl. Top 3)' if exclude_top3_pct else ''}"
        else:
            n_users = int(np.ceil(len(sorted_pct_users) * pct_percentile / 100))
            if exclude_top3_pct:
                pct_users = sorted_pct_users.iloc[3:3+n_users]['user']
            else:
                pct_users = sorted_pct_users.head(n_users)['user']
            pct_label = f"Top {pct_percentile}% by {pct_metric} ({n_users} users){' (Excl. Top 3)' if exclude_top3_pct else ''}"

        pct_daily = data['daily_volume'][data['daily_volume']['user'].isin(pct_users)].groupby('time')['daily_usd_volume'].sum().reset_index(name='cohort_volume')
        merged_pct = pd.merge(pct_daily, total_daily, on='time', how='right').fillna(0)
        merged_pct['share'] = 100 * merged_pct['cohort_volume'] / merged_pct['total_volume']
        fig_pct = px.line(merged_pct, x='time', y='share',
                             title=f"{pct_label}: Share of Total Daily Volume",
                         labels={'share': 'Share (%)', 'time': 'Date'})
        fig_pct.update_yaxes(tickformat=".2f", range=[0, 100])
        st.plotly_chart(fig_pct, use_container_width=True, key=f"pct_share_{pct_label}")

        st.info(f"{cohort_label} includes {len(cohort_users)} users. Share is calculated as the sum of their daily volume divided by total daily volume. If the cohort is larger than the number of active users on a day, share will be 100%.")

        # --- User Cohort Analysis and CSV Export ---
        st.subheader("User Cohort Analysis & Export")
        st.markdown("""
        Analyze and export daily/cumulative volume and activity frequency for top users. Select cohort size and ranking metric below.
        """)
        cohort_options = [10, 100, 1000]
        metric_options = {'Total Volume': 'volume', 'Active Days': 'activity'}
        cohort_size = st.selectbox("Select cohort size:", cohort_options, index=2)
        ranking_metric = st.selectbox("Rank users by:", list(metric_options.keys()), index=0)

        # Prepare per-wallet daily and cumulative volume
        user_daily = data['daily_volume'].copy()
        user_daily = user_daily.sort_values(['user', 'time'])
        user_daily['cumulative_usd_volume'] = user_daily.groupby('user')['daily_usd_volume'].cumsum()
        # User frequency: number of unique days active
        user_freq = user_daily.groupby('user').agg(
            total_volume=('daily_usd_volume', 'sum'),
            active_days=('time', 'nunique'),
            first_active=('time', 'min'),
            last_active=('time', 'max')
        ).reset_index()
        # Merge in trade count and liquidation if available
        if 'trade_count' in data:
            user_freq = user_freq.merge(data['trade_count'], on='user', how='left', suffixes=('', '_trades'))
            user_freq = user_freq.rename(columns={'value': 'trade_count'})
        if 'liquidations' in data:
            user_freq = user_freq.merge(data['liquidations'], on='user', how='left', suffixes=('', '_liquidated'))
            user_freq = user_freq.rename(columns={'value': 'liquidation_notional'})
        # Ranking
        if metric_options[ranking_metric] == 'volume':
            top_users = user_freq.sort_values('total_volume', ascending=False).head(cohort_size)['user']
        else:
            top_users = user_freq.sort_values('active_days', ascending=False).head(cohort_size)['user']
        cohort_daily = user_daily[user_daily['user'].isin(top_users)].copy()
        # Save to CSV
        csv_cols = ['user', 'time', 'daily_usd_volume', 'cumulative_usd_volume']
        if 'trade_count' in user_freq:
            csv_cols.append('trade_count')
        if 'liquidation_notional' in user_freq:
            csv_cols.append('liquidation_notional')
        cohort_daily = cohort_daily.merge(user_freq[['user', 'trade_count', 'liquidation_notional']].drop_duplicates(), on='user', how='left')
        cohort_daily[csv_cols].to_csv(f'user_cohort_{cohort_size}_{metric_options[ranking_metric]}.csv', index=False)
        st.success(f"Saved user_cohort_{cohort_size}_{metric_options[ranking_metric]}.csv to disk.")

        st.markdown("""
        ### Top Volume Traders
        This visualization identifies the most active traders by total volume, highlighting:
        - Market makers and major participants
        - Potential institutional traders
        - Key market influencers
        """)
        # Add toggle to exclude top 3 wallets
        exclude_top3_traders = st.checkbox("Exclude Top 3 Wallets", value=True, key="exclude_top3_traders")
        # Top users by volume (scatterplot + table)
        top_users_all = data['largest_users'].copy()
        if exclude_top3_traders:
            top_users = top_users_all.iloc[3:13].copy()
        else:
            top_users = top_users_all.head(10).copy()
        # Merge in trade count and liquidation ratio if available
        if 'trade_count' in data and 'liquidations' in data:
            top_users = top_users.merge(data['trade_count'], on='user', how='left', suffixes=('', '_trades'))
            top_users = top_users.merge(data['liquidations'], on='user', how='left', suffixes=('', '_liquidated'))
            top_users['trade_count'] = top_users['value_trades']
            top_users['liquidation_ratio'] = top_users['value_liquidated'] / top_users['value']
        # Shorten wallet addresses for x-axis
        def short_addr(addr):
            return addr[:6] + '...' + addr[-4:]
        top_users['short_user'] = top_users['user'].apply(short_addr)
        fig = px.scatter(
            top_users,
            x='short_user',
            y='value',
            size='value',
            color='value',
            color_continuous_scale='Blues',
            title="Top 10 Users by Total Volume",
            labels={'short_user': 'User Address', 'value': 'Total Volume (USD)'}
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, key=f"top_volume_traders_{exclude_top3_traders}")
        st.markdown("""
        **Top 10 Wallets Table**
        <div style='overflow-x:auto;'>
        """, unsafe_allow_html=True)
        table_cols = ['user', 'value']
        col_rename = {'user': 'Wallet Address', 'value': 'Total Volume (USD)'}
        if 'trade_count' in top_users:
            table_cols.append('trade_count')
            col_rename['trade_count'] = 'Trade Count'
        if 'liquidation_ratio' in top_users:
            table_cols.append('liquidation_ratio')
            col_rename['liquidation_ratio'] = 'Liquidation Ratio'
        st.dataframe(
            top_users[table_cols].rename(columns=col_rename).style.format({'Total Volume (USD)': '{:,.0f}', 'Liquidation Ratio': '{:.2%}'}),
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Liquidation Analysis")
        st.markdown("""
        ### Liquidation Patterns
        This analysis examines liquidation events to understand:
        - Risk management effectiveness
        - Market volatility impact
        - User risk profiles
        - Potential market stress points
        """)
        
        # --- Controls for Top Users by Liquidation Amount ---
        st.markdown("#### Top Users by Liquidation Amount")
        col1, col2, col3, col4, col5 = st.columns([2,2,2,2,2])
        with col1:
            liq_mode = st.radio("Cohort Mode", ["Top N", "Top Percentile"], index=0, horizontal=True, key="liq_mode")
        with col2:
            if liq_mode == "Top N":
                liq_n = st.selectbox("Select Top N:", [10, 100, 500, 1000], index=3, key="liq_n")
            else:
                liq_percentile = st.slider("Select Top Percentile:", min_value=1, max_value=100, value=10, step=1, key="liq_percentile")
        with col3:
            liq_metric = st.radio("Rank by", ["Volume", "Activity"], index=0, horizontal=True, key="liq_metric")
        with col4:
            exclude_top1_liq = st.checkbox("Exclude Top 1 Wallet", value=False, key="exclude_top1_liq")
        with col5:
            liq_chart_type = st.radio("Chart Type", ["Bar", "Bubble"], index=0, horizontal=True, key="liq_chart_type")

        # Select cohort from user_freq
        if liq_metric == "Volume":
            sorted_liq_users = user_freq.sort_values('total_volume', ascending=False)
        else:
            sorted_liq_users = user_freq.sort_values('active_days', ascending=False)
        if liq_mode == "Top N":
            if exclude_top1_liq:
                cohort_liq_users = sorted_liq_users.iloc[1:1+liq_n]['user']
                n_excluded = 1
            else:
                cohort_liq_users = sorted_liq_users.head(liq_n)['user']
                n_excluded = 0
            liq_label = f"Top {liq_n} by {liq_metric}{' (Excl. Top 1)' if exclude_top1_liq else ''}"
        else:
            n_liq_users = int(np.ceil(len(sorted_liq_users) * liq_percentile / 100))
            if exclude_top1_liq:
                cohort_liq_users = sorted_liq_users.iloc[1:1+n_liq_users]['user']
                n_excluded = 1
            else:
                cohort_liq_users = sorted_liq_users.head(n_liq_users)['user']
                n_excluded = 0
            liq_label = f"Top {liq_percentile}% by {liq_metric} ({n_liq_users} users){' (Excl. Top 1)' if exclude_top1_liq else ''}"

        # Notice for excluded top 1 wallet
        if n_excluded > 0:
            st.info(f"Excluded the top {n_excluded} wallet(s) by {liq_metric} from the cohort as per your selection.")

        # Merge with liquidation data, keep only users present in both
        liq_data = data['liquidations'].copy()
        cohort_liq_df = pd.DataFrame({'user': cohort_liq_users})
        n_cohort = len(cohort_liq_df)
        merged_liq = pd.merge(cohort_liq_df, liq_data, on='user', how='inner')
        n_after_merge = merged_liq['user'].nunique()
        n_discarded = n_cohort - n_after_merge
        if n_discarded > 0:
            st.warning(f"{n_discarded} wallet(s) in your selected cohort had no liquidation data and were excluded from this analysis.")
        merged_liq = merged_liq.merge(user_freq, on='user', how='left')
        merged_liq['short_user'] = merged_liq['user'].apply(lambda addr: addr[:6] + '...' + addr[-4:])
        merged_liq = merged_liq.sort_values('value', ascending=True)
        n_both = merged_liq['user'].nunique()
        st.markdown(f"<div style='margin-bottom: 0.5em;'><b>{n_both} wallets</b> are present in both the user and liquidation tables. <br>This analysis is only for this intersection of wallets. Only these wallets have both user activity and liquidation data available, and all visualizations and tables below are restricted to this universe.</div>", unsafe_allow_html=True)
        st.dataframe(merged_liq[['user', 'value', 'total_volume', 'active_days']].rename(columns={'user': 'Wallet Address', 'value': 'Liquidation Amount (USD)', 'total_volume': 'Total Volume', 'active_days': 'Active Days'}).style.format({'Liquidation Amount (USD)': '{:,.0f}', 'Total Volume': '{:,.0f}'}), use_container_width=True)

        # Chart
        if liq_chart_type == "Bar":
            fig_liq = px.bar(
                merged_liq,
                y='short_user',
                x='value',
                orientation='h',
                title=f"{liq_label} by Liquidation Amount",
                labels={'short_user': 'User Address', 'value': 'Liquidation Amount (USD)'}
            )
        else:
            fig_liq = px.scatter(
                merged_liq,
                x='total_volume',
                y='value',
                size='active_days',
                color='value',
                color_continuous_scale='Blues',
                hover_name='short_user',
                title=f"{liq_label} by Liquidation Amount (Bubble Chart)",
                labels={'total_volume': 'Total Volume (USD)', 'value': 'Liquidation Amount (USD)'}
            )
        st.plotly_chart(fig_liq, use_container_width=True, key=f"liq_amt_{liq_label}_{liq_chart_type}")

        # --- Liquidation to Volume Ratio: Intersection Universe Only ---
        st.markdown("#### Liquidation to Volume Ratio")
        # Find intersection of wallets in both daily volume and liquidations
        # Apply exclusion logic to intersection as well
        if liq_metric == "Volume":
            sorted_intersection = user_freq[user_freq['user'].isin(data['liquidations']['user'])].sort_values('total_volume', ascending=False)
        else:
            sorted_intersection = user_freq[user_freq['user'].isin(data['liquidations']['user'])].sort_values('active_days', ascending=False)
        if exclude_top1_liq:
            intersection_wallets = list(sorted_intersection.iloc[1:]['user'])
        else:
            intersection_wallets = list(sorted_intersection['user'])
        n_intersection = len(intersection_wallets)
        n_liq_only = len(set(data['liquidations']['user']) - set(data['daily_volume']['user']))
        n_vol_only = len(set(data['daily_volume']['user']) - set(data['liquidations']['user']))
        if n_liq_only > 0 or n_vol_only > 0:
            st.info(f"{n_liq_only} wallet(s) have liquidation data but no trading data, and {n_vol_only} wallet(s) have trading data but no liquidation data. Only the {n_intersection} wallet(s) present in both are included in this analysis." + (" (Excluding top 1 wallet)" if exclude_top1_liq else ""))
        st.markdown(f"<div style='margin-bottom: 0.5em;'><b>{n_intersection} wallets</b> are present in both the user and liquidation tables for this metric. <br>This analysis is only for this intersection of wallets. Only these wallets have both user activity and liquidation data available, and all visualizations and tables below are restricted to this universe.</div>", unsafe_allow_html=True)
        # Prepare merged data for these wallets
        merged_data = pd.merge(data['largest_users'], data['liquidations'], on='user', suffixes=('_volume', '_liquidated'))
        merged_data['liquidation_ratio'] = merged_data['value_liquidated'] / merged_data['value_volume']
        merged_data = merged_data[merged_data['user'].isin(intersection_wallets)]
        merged_data = merged_data.merge(user_freq, on='user', how='left')
        merged_data['short_user'] = merged_data['user'].apply(lambda addr: addr[:6] + '...' + addr[-4:])
        merged_data = merged_data.sort_values('liquidation_ratio', ascending=True)
        st.dataframe(merged_data[['user', 'liquidation_ratio', 'value_volume', 'active_days']].rename(columns={'user': 'Wallet Address', 'liquidation_ratio': 'Liquidation/Volume Ratio', 'value_volume': 'Total Volume', 'active_days': 'Active Days'}).style.format({'Liquidation/Volume Ratio': '{:.4f}', 'Total Volume': '{:,.0f}'}), use_container_width=True)
        # Chart (bar only for clarity)
        fig_ratio = px.bar(
            merged_data,
            y='short_user',
            x='liquidation_ratio',
            orientation='h',
            title=f"Intersection Wallets: Liquidation/Volume Ratio",
            labels={'short_user': 'User Address', 'liquidation_ratio': 'Liquidation/Volume Ratio'}
        )
        st.plotly_chart(fig_ratio, use_container_width=True, key=f"liq_ratio_intersection_bar")
    
    with tab3:
        st.subheader("Trading Behavior Analysis")
        st.markdown("""
        ### Trading Frequency Distribution
        This histogram shows the distribution of trade counts across users:
        - Identifies typical trading patterns
        - Shows the range of trading activity
        - Helps understand user engagement levels
        """)
        
        # Trade count distribution
        fig = px.histogram(data['trade_count'], x='value',
                          title="Distribution of Trade Counts",
                          labels={'value': 'Number of Trades'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Volume per Trade Analysis
        This metric reveals trading efficiency and strategy:
        - Higher values indicate larger, potentially more strategic trades
        - Lower values suggest more frequent, smaller trades
        - Helps identify different trading styles
        """)

        # Add controls for Top N/Percentile and Rank by
        col1, col2, col3 = st.columns([2,2,2])
        with col1:
            vpt_cohort_mode = st.radio("Cohort Mode", ["Top N", "Top Percentile"], index=0, horizontal=True, key="vpt_cohort_mode")
        with col2:
            if vpt_cohort_mode == "Top N":
                vpt_n = st.selectbox("Select Top N:", [10, 100, 500, 1000], index=0, key="vpt_n")
            else:
                vpt_percentile = st.slider("Select Top Percentile:", min_value=1, max_value=100, value=10, step=1, key="vpt_percentile")
        with col3:
            vpt_metric = st.radio("Rank by", ["Volume", "Activity"], index=0, horizontal=True, key="vpt_metric")

        # Compute cohort for volume per trade
        if vpt_metric == "Volume":
            sorted_vpt_users = user_freq.sort_values('total_volume', ascending=False)
        else:
            sorted_vpt_users = user_freq.sort_values('active_days', ascending=False)
        if vpt_cohort_mode == "Top N":
            vpt_cohort_users = sorted_vpt_users.head(vpt_n)['user']
            vpt_label = f"Top {vpt_n} by {vpt_metric}"
        else:
            n_vpt_users = int(np.ceil(len(sorted_vpt_users) * vpt_percentile / 100))
            vpt_cohort_users = sorted_vpt_users.head(n_vpt_users)['user']
            vpt_label = f"Top {vpt_percentile}% by {vpt_metric} ({n_vpt_users} users)"

        # Volume per trade analysis for selected cohort
        merged_trade_data = pd.merge(data['largest_users'], data['trade_count'], 
                                   on='user', suffixes=('_volume', '_trades'))
        merged_trade_data['volume_per_trade'] = merged_trade_data['value_volume'] / merged_trade_data['value_trades']
        merged_trade_data = merged_trade_data[merged_trade_data['user'].isin(vpt_cohort_users)]
        merged_trade_data['short_user'] = merged_trade_data['user'].apply(lambda addr: addr[:6] + '...' + addr[-4:])
        # Display all users in the cohort, sorted by volume per trade
        merged_trade_data = merged_trade_data.sort_values('volume_per_trade', ascending=False)
        fig = px.bar(merged_trade_data, x='short_user', y='volume_per_trade',
                     title=f"{vpt_label}: Users by Volume per Trade",
                     labels={'volume_per_trade': 'Volume per Trade (USD)', 'short_user': 'User Address'})
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
        # Show a table for the full cohort
        st.dataframe(merged_trade_data[['user', 'volume_per_trade', 'value_volume', 'value_trades']].rename(columns={
            'user': 'Wallet Address',
            'volume_per_trade': 'Volume per Trade (USD)',
            'value_volume': 'Total Volume',
            'value_trades': 'Trade Count'
        }).style.format({'Volume per Trade (USD)': '{:,.2f}', 'Total Volume': '{:,.0f}', 'Trade Count': '{:,.0f}'}), use_container_width=True)
    
    with tab4:
        st.subheader("Advanced Combined Analysis")
        st.markdown("""
        ### Risk-Adjusted Performance
        This analysis combines multiple metrics to create a comprehensive view of trading performance:
        - Considers volume, liquidation risk, and trading frequency
        - Higher scores indicate more efficient trading
        - Helps identify consistently successful traders
        """)

        # Add controls for Top N/Percentile and Rank by
        col1, col2, col3 = st.columns([2,2,2])
        with col1:
            rap_cohort_mode = st.radio("Cohort Mode", ["Top N", "Top Percentile"], index=0, horizontal=True, key="rap_cohort_mode")
        with col2:
            if rap_cohort_mode == "Top N":
                rap_n = st.selectbox("Select Top N:", [10, 100, 500, 1000], index=0, key="rap_n")
            else:
                rap_percentile = st.slider("Select Top Percentile:", min_value=1, max_value=100, value=10, step=1, key="rap_percentile")
        with col3:
            rap_metric = st.radio("Rank by", ["Volume", "Activity"], index=0, horizontal=True, key="rap_metric")

        # Create a comprehensive user profile
        user_profile = pd.merge(data['largest_users'], data['liquidations'], 
                              on='user', suffixes=('_volume', '_liquidated'))
        user_profile = pd.merge(user_profile, data['trade_count'], 
                              on='user', suffixes=('', '_trades'))
        user_profile['volume_per_trade'] = user_profile['value_volume'] / user_profile['value']
        user_profile['liquidation_ratio'] = user_profile['value_liquidated'] / user_profile['value_volume']
        user_profile['risk_score'] = (user_profile['value_volume'] * (1 - user_profile['liquidation_ratio'])) / user_profile['value']

        # Compute cohort for risk-adjusted score
        if rap_metric == "Volume":
            sorted_rap_users = user_profile.sort_values('value_volume', ascending=False)
        else:
            sorted_rap_users = user_profile.sort_values('value', ascending=False)
        if rap_cohort_mode == "Top N":
            rap_cohort = sorted_rap_users.head(rap_n)
            rap_label = f"Top {rap_n} by {rap_metric}"
        else:
            n_rap_users = int(np.ceil(len(sorted_rap_users) * rap_percentile / 100))
            rap_cohort = sorted_rap_users.head(n_rap_users)
            rap_label = f"Top {rap_percentile}% by {rap_metric} ({n_rap_users} users)"

        rap_cohort['short_user'] = rap_cohort['user'].apply(lambda addr: addr[:6] + '...' + addr[-4:])
        # Display all users in the cohort, sorted by risk-adjusted score
        rap_cohort = rap_cohort.sort_values('risk_score', ascending=False)
        fig = px.bar(rap_cohort, x='short_user', y='risk_score',
                     title=f"{rap_label}: Users by Risk-Adjusted Volume Score",
                     labels={'risk_score': 'Risk-Adjusted Score', 'short_user': 'User Address'})
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
        # Show a table for the full cohort
        st.dataframe(rap_cohort[['user', 'risk_score', 'value_volume', 'value_liquidated', 'value', 'volume_per_trade', 'liquidation_ratio']].rename(columns={
            'user': 'Wallet Address',
            'risk_score': 'Risk-Adjusted Score',
            'value_volume': 'Total Volume',
            'value_liquidated': 'Liquidated Amount',
            'value': 'Trade Count',
            'volume_per_trade': 'Volume per Trade',
            'liquidation_ratio': 'Liquidation Ratio'
        }).style.format({'Risk-Adjusted Score': '{:,.2f}', 'Total Volume': '{:,.0f}', 'Liquidated Amount': '{:,.0f}', 'Trade Count': '{:,.0f}', 'Volume per Trade': '{:,.2f}', 'Liquidation Ratio': '{:.2%}'}), use_container_width=True)
        
        st.markdown("""
        ### Volume vs Liquidation Analysis
        This scatter plot reveals the relationship between trading volume and liquidation risk:
        - Size of points indicates number of trades
        - Color represents volume per trade
        - Helps identify optimal trading strategies
        """)
        
        # Scatter plot of volume vs liquidation ratio
        fig = px.scatter(user_profile, x='value_volume', y='liquidation_ratio',
                        size='value', color='volume_per_trade',
                        title="Volume vs Liquidation Ratio",
                        labels={'value_volume': 'Total Volume (USD)',
                               'liquidation_ratio': 'Liquidation Ratio',
                               'volume_per_trade': 'Volume per Trade'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### User Clustering Analysis
        This 3D visualization groups users into clusters based on their trading behavior:
        - Identifies distinct trading patterns
        - Shows relationships between volume, liquidations, and trade frequency
        - Helps understand market participant types
        """)
        
        # User clustering analysis
        st.write("User Clustering Analysis")
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        # Prepare data for clustering
        cluster_data = user_profile[['value_volume', 'value_liquidated', 'value', 'volume_per_trade']].copy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        user_profile['cluster'] = kmeans.fit_predict(scaled_data)
        
        # Visualize clusters
        fig = px.scatter_3d(user_profile, 
                           x='value_volume', 
                           y='value_liquidated', 
                           z='value',
                           color='cluster',
                           title="User Clusters by Trading Behavior",
                           labels={'value_volume': 'Total Volume',
                                 'value_liquidated': 'Liquidated Amount',
                                 'value': 'Trade Count'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Cluster Characteristics
        This table shows the average metrics for each user cluster:
        - Helps understand different trading styles
        - Identifies risk and performance patterns
        - Provides insights into market participant behavior
        """)
        
        # Cluster characteristics
        st.write("Cluster Characteristics")
        cluster_stats = user_profile.groupby('cluster').agg({
            'value_volume': 'mean',
            'value_liquidated': 'mean',
            'value': 'mean',
            'volume_per_trade': 'mean',
            'liquidation_ratio': 'mean'
        }).round(2)
        st.dataframe(cluster_stats)

    with tab5:
        st.subheader("User Frequency Analysis")
        st.markdown("""
        ### User Frequency Analysis
        - **Active Days**: Number of days a user was active
        - **First/Last Active**: User's first and last trading day
        - **Trade Count** and **Liquidation Notional** (if available)
        """)
        st.dataframe(user_freq.sort_values('active_days', ascending=False).head(20), use_container_width=True)
        fig_freq = px.histogram(user_freq, x='active_days', nbins=50, title='Distribution of User Active Days', labels={'active_days': 'Active Days'})
        st.plotly_chart(fig_freq, use_container_width=True)

if __name__ == "__main__":
    main() 
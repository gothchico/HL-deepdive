# airdrop_analysis.py
import os
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime



# ── 0) Fetch & save fees.json from Hypurrscan ──
FEES_ENDPOINT = "https://abc.hypurrscan.io/fees"
os.makedirs("data", exist_ok=True)
try:
    resp = requests.get(FEES_ENDPOINT)
    resp.raise_for_status()
    fees_list = resp.json()
    with open("data/fees.json", "w") as f:
        json.dump(fees_list, f)
    st.sidebar.success("Fetched latest fees data")
except Exception as e:
    st.sidebar.error(f"Failed to fetch fees: {e}")
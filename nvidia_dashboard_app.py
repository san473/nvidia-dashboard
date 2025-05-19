pip install streamlit yfinance plotly pandas
streamlit run nvidia_dashboard_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="NVIDIA Stock Dashboard", layout="wide")

# Title
st.title("📊 NVIDIA Stock Dashboard - Real-Time Analysis & Valuation")

# Sidebar
st.sidebar.header("Configuration")
selected_interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "1h", "1d"])
selected_range = st.sidebar.selectbox("Select Date Range", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"])

# Load stock data
ticker = "NVDA"
stock = yf.Ticker(ticker)
df = stock.history(interval=selected_interval, period=selected_range)

# Real-Time Market Data
st.subheader("📈 Real-Time Market Data")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Price", f"${stock.info['currentPrice']}")
    st.metric("52-Week High", f"${stock.info['fiftyTwoWeekHigh']}")
    st.metric("52-Week Low", f"${stock.info['fiftyTwoWeekLow']}")
with col2:
    st.metric("Market Cap", f"${round(stock.info['marketCap']/1e9, 2)}B")
    st.metric("Beta", f"{stock.info['beta']}")
    st.metric("Volume", f"{stock.info['volume']}")
with col3:
    st.metric("Average Volume", f"{stock.info['averageVolume']}")
    st.metric("Bid", f"{stock.info['bid']}")
    st.metric("Ask", f"{stock.info['ask']}")

# Candlestick chart with volume
st.subheader("📊 Intraday Candlestick Chart with Volume")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Candlestick'
))
fig.update_layout(xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

# Placeholder sections for further development
st.subheader("📉 Financial Metrics & Ratios")
st.info("This section will display live financial statements and key ratios (P/E, ROE, etc.).")

st.subheader("💰 Valuation Models (DCF, Comps, Transactions)")
st.info("This section will feature an automated DCF model and peer comparison tools.")

st.subheader("📰 Sentiment & News Analysis")
st.info("Real-time news feed, earnings call sentiment, and social media analysis coming soon.")

st.subheader("🔍 Analyst Insights")
st.info("Consensus recommendations, price targets, and historical forecast accuracy.")

st.subheader("🌱 ESG & Risk Metrics")
st.info("ESG scoring, volatility charts, and regulatory/geopolitical risk updates.")

st.subheader("📈 Interactive Visualizations & Scenario Analysis")
st.info("Revenue breakdown, geographic exposure, and what-if analysis tools.")

st.subheader("🔔 Alerts & Recommendations")
st.info("Dynamic alerts based on technical/fundamental indicators and valuation gaps.")


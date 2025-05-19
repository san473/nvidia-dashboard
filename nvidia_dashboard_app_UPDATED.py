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

# ------------------------ FINANCIAL METRICS & RATIOS ------------------------
st.subheader("📉 Financial Metrics & Ratios")

try:
    info = stock.info
    ratios = {
        "Market Cap": info.get("marketCap"),
        "Trailing P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "PEG Ratio": info.get("pegRatio"),
        "Price to Book": info.get("priceToBook"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "Profit Margin": info.get("profitMargins"),
        "Gross Margins": info.get("grossMargins"),
        "Operating Margins": info.get("operatingMargins"),
        "Debt to Equity": info.get("debtToEquity"),
    }
    for k, v in ratios.items():
        if v is not None:
            st.write(f"**{k}:** {v:,.2f}" if isinstance(v, float) else f"**{k}:** {v}")
        else:
            st.write(f"**{k}:** Data not available")
except Exception as e:
    st.error(f"Failed to load financial metrics: {e}")


st.subheader("💰 Valuation Models (DCF, Comps, Transactions)")

# ------------------------ DCF VALUATION ------------------------
st.subheader("💰 Valuation Models (DCF)")

try:
    cashflow = stock.cashflow
    st.write("🧾 Available cash flow rows:", cashflow.index.tolist())

    if 'Total Cash From Operating Activities' in cashflow.index:
        op_cash = cashflow.loc['Total Cash From Operating Activities']
    else:
        op_cash = cashflow.loc[cashflow.index.str.contains("Operating", case=False)].iloc[0]

    avg_cashflow = op_cash.mean()
    discount_rate = 0.10
    growth_rate = 0.05
    intrinsic_value = avg_cashflow * (1 + growth_rate) / (discount_rate - growth_rate)

    st.success(f"Intrinsic Value Estimate (DCF): **${intrinsic_value:,.2f}**")
except Exception as e:
    st.warning(f"DCF valuation failed: {e}")


st.subheader("📰 Sentiment & News Analysis")

# ------------------------ NEWS ANALYSIS ------------------------
st.subheader("📰 Real-Time News Feed")

try:
    news_items = stock.news
    if not news_items:
        raise ValueError("No news returned")

    for item in news_items[:5]:
        title = item.get("title", "No Title")
        link = item.get("link", "#")
        publisher = item.get("publisher", "Unknown Publisher")
        publish_time = item.get("providerPublishTime", "")
        st.markdown(f"**[{title}]({link})**  
{publisher} - {publish_time}")
except Exception as e:
    st.info("No recent news available.")


st.subheader("🔍 Analyst Insights")
st.info("Consensus recommendations, price targets, and historical forecast accuracy.")

st.subheader("🌱 ESG & Risk Metrics")
st.info("ESG scoring, volatility charts, and regulatory/geopolitical risk updates.")

st.subheader("📈 Interactive Visualizations & Scenario Analysis")

# ------------------------ INTERACTIVE VISUALIZATIONS ------------------------
st.subheader("📈 Interactive Visualizations")

try:
    income_stmt = stock.financials
    revenue = income_stmt.loc['Total Revenue'].dropna()
    revenue.index = pd.to_datetime(revenue.index)
    revenue = revenue.sort_index()

    st.subheader("📈 Revenue Trend")
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(x=revenue.index, y=revenue.values / 1e9, mode='lines+markers', name='Revenue'))
    fig_rev.update_layout(title="Revenue Trend (in Billions)", xaxis_title="Year", yaxis_title="Revenue (B USD)")
    st.plotly_chart(fig_rev, use_container_width=True)

    st.subheader("📊 Financial Ratios")
    ratios_for_chart = {
        "P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "Price to Book": info.get("priceToBook"),
        "ROA": info.get("returnOnAssets"),
        "ROE": info.get("returnOnEquity"),
        "Profit Margin": info.get("profitMargins"),
        "Debt to Equity": info.get("debtToEquity")
    }
    ratios_cleaned = {k: v for k, v in ratios_for_chart.items() if v is not None}
    fig_ratios = go.Figure([go.Bar(x=list(ratios_cleaned.keys()), y=list(ratios_cleaned.values()), marker_color='indianred')])
    fig_ratios.update_layout(title="Key Financial Ratios", xaxis_title="Ratio", yaxis_title="Value")
    st.plotly_chart(fig_ratios, use_container_width=True)

    st.subheader("🎛️ KPI Toggles")
    kpi_options = ["Revenue Growth", "Net Margin", "Sales"]
    selected_kpi = st.selectbox("Select KPI", kpi_options)

    if selected_kpi == "Revenue Growth":
        growth = revenue.pct_change().dropna() * 100
        fig_kpi = go.Figure()
        fig_kpi.add_trace(go.Bar(x=growth.index, y=growth.values, name="Revenue Growth (%)"))
        fig_kpi.update_layout(title="Revenue Growth (%)", xaxis_title="Year", yaxis_title="Growth %")
        st.plotly_chart(fig_kpi, use_container_width=True)
    elif selected_kpi == "Net Margin":
        net_income = income_stmt.loc['Net Income']
        net_margin = (net_income / income_stmt.loc['Total Revenue']) * 100
        net_margin = net_margin.sort_index()
        fig_kpi = go.Figure()
        fig_kpi.add_trace(go.Scatter(x=net_margin.index, y=net_margin.values, name="Net Margin", mode='lines+markers'))
        fig_kpi.update_layout(title="Net Margin (%)", xaxis_title="Year", yaxis_title="Net Margin %")
        st.plotly_chart(fig_kpi, use_container_width=True)
    elif selected_kpi == "Sales":
        fig_kpi = go.Figure()
        fig_kpi.add_trace(go.Scatter(x=revenue.index, y=revenue.values / 1e9, mode='lines+markers', name="Sales"))
        fig_kpi.update_layout(title="Sales Over Time", xaxis_title="Year", yaxis_title="Sales (B USD)")
        st.plotly_chart(fig_kpi, use_container_width=True)
except Exception as e:
    st.warning(f"Unable to display visualizations: {e}")


st.subheader("🔔 Alerts & Recommendations")
st.info("Dynamic alerts based on technical/fundamental indicators and valuation gaps.")


import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

NEWSAPI_KEY = st.secrets["newsapi"]
def fetch_news(ticker):
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWSAPI_KEY}&sortBy=publishedAt&language=en"
        response = requests.get(url)
        articles = response.json().get("articles", [])[:5]
        return articles
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        return []


st.set_page_config(page_title="📈 Stock Dashboard", layout="wide")

# ------------------------ HEADER ------------------------
st.title("📊 Comprehensive Stock Dashboard")
ticker = st.text_input("Enter stock ticker (e.g., AAPL, NVDA, MSFT)", value="AAPL")

@st.cache_data
def get_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1y")
    

    return info, hist

def get_financial_ratios(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    ratios = {
        "Market Cap": info.get("marketCap"),
        "Trailing P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "PEG Ratio": info.get("pegRatio"),
        "Price to Book": info.get("priceToBook"),
        "Enterprise to Revenue": info.get("enterpriseToRevenue"),
        "Enterprise to EBITDA": info.get("enterpriseToEbitda"),
        "Return on Assets (ROA)": info.get("returnOnAssets"),
        "Return on Equity (ROE)": info.get("returnOnEquity"),
        "Profit Margin": info.get("profitMargins"),
        "Gross Margins": info.get("grossMargins"),
        "Operating Margins": info.get("operatingMargins"),
        "Current Ratio": info.get("currentRatio"),
        "Quick Ratio": info.get("quickRatio"),
        "Debt to Equity": info.get("debtToEquity"),
    }

    return ratios

def format_large_number(n):
    if n is None:
        return "N/A"
    elif n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    else:
        return str(n)

# Fetch data
try:
    info, hist = get_data(ticker)
    ratios = get_financial_ratios(ticker)

    st.subheader(f"{info.get('longName', ticker)} ({ticker})")
    st.markdown(f"**Sector:** {info.get('sector', 'N/A')}  ")
    st.markdown(f"**Industry:** {info.get('industry', 'N/A')}  ")
    st.markdown(f"**Market Cap:** ${info.get('marketCap', 0):,}  ")

    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close'))
    fig.update_layout(title=f"{ticker} Price Chart (1Y)", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Failed to fetch data for ticker {ticker}. Error: {e}")

    # ------------------------ GEOGRAPHIC REVENUE / HQ INFO ------------------------
with st.expander("🌍 Geographic & Business Overview"):
    try:
        country = info.get("country", "N/A")
        city = info.get("city", "")
        state = info.get("state", "")
        address = f"{city}, {state}, {country}" if city else country
        st.markdown(f"**Headquarters:** {address}")

        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
        st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")

        summary = info.get("longBusinessSummary", "Business description not available.")
        st.markdown(f"**Business Description:**  \n{summary}")
    except Exception as e:
        st.warning("Geographic and company summary data not available.")




# -------------------- Peer Comparison --------------------
st.header("🔍 Peer Comparison")

# ✅ Curated list of valid peer tickers and labels
peer_options = {
    "AMD": "Advanced Micro Devices (AMD)",
    "INTC": "Intel Corporation (INTC)",
    "AVGO": "Broadcom Inc. (AVGO)",
    "QCOM": "Qualcomm Inc. (QCOM)",
    "TSM": "Taiwan Semiconductor (TSM)",
    "TXN": "Texas Instruments (TXN)",
    "MU": "Micron Technology (MU)",
    "AMAT": "Applied Materials (AMAT)"
}

selected_peers = st.multiselect(
    "Select peer companies to compare:",
    options=list(peer_options.keys()),
    format_func=lambda x: peer_options[x],
    default=["AMD", "INTC"]  # Optional default selection
)

if selected_peers:
    comparison_data = []

    for peer_ticker in selected_peers:
        peer = yf.Ticker(peer_ticker)
        info = peer.info

        comparison_data.append({
            "Company": peer_options[peer_ticker],
            "Ticker": peer_ticker,
            "Price": info.get("currentPrice"),
            "Market Cap (B)": info.get("marketCap", 0) / 1e9,
            "PE Ratio": info.get("trailingPE"),
            "Revenue (B)": info.get("totalRevenue", 0) / 1e9,
            "Net Margin (%)": (info.get("netMargins", 0) or 0) * 100,
            "Return on Equity (%)": (info.get("returnOnEquity", 0) or 0) * 100,
        })

    df_peers = pd.DataFrame(comparison_data)
    st.dataframe(df_peers.set_index("Ticker"), use_container_width=True)

else:
    st.warning("Please select one or more companies to compare.")


# ------------------------ REAL-TIME NEWS ------------------------
st.subheader("📰 Real-Time News Feed")
news_articles = fetch_news(ticker)

if news_articles:
    for article in news_articles:
        st.markdown(f"**[{article['title']}]({article['url']})**")
        st.caption(f"{article['source']['name']} - {article['publishedAt']}")
        st.write(article['description'] or "No summary available.")
        st.markdown("---")
else:
    st.info("No recent news available.")


# ---------------------- INTERACTIVE VISUALIZATIONS ----------------------
st.header("📈 Interactive Financial Visualizations")

@st.cache_data
def get_quarterly_financials(ticker):
    stock = yf.Ticker(ticker)
    income_stmt = stock.quarterly_financials
    cashflow_stmt = stock.quarterly_cashflow
    return income_stmt, cashflow_stmt

try:
    income_stmt, cashflow_stmt = get_quarterly_financials(ticker)

    # Convert to regular format
    income_df = income_stmt.T
    cashflow_df = cashflow_stmt.T

    # --- Operating Cash Flow ---
    if 'Total Cash From Operating Activities' in cashflow_df.columns:
        fig_ocf = go.Figure()
        fig_ocf.add_trace(go.Scatter(
            x=cashflow_df.index, 
            y=cashflow_df['Total Cash From Operating Activities'], 
            mode='lines+markers', 
            name='Operating Cash Flow',
            line=dict(color='royalblue')
        ))
        fig_ocf.update_layout(
            title="Operating Cash Flow (Quarterly)", 
            yaxis_title="USD", 
            xaxis_title="Date"
        )
        st.plotly_chart(fig_ocf, use_container_width=True)

    # --- Free Cash Flow (Operating Cash Flow - CapEx) ---
    if 'Total Cash From Operating Activities' in cashflow_df.columns and 'Capital Expenditures' in cashflow_df.columns:
        fcf = cashflow_df['Total Cash From Operating Activities'] - cashflow_df['Capital Expenditures']
        fig_fcf = go.Figure()
        fig_fcf.add_trace(go.Scatter(
            x=cashflow_df.index, 
            y=fcf, 
            mode='lines+markers', 
            name='Free Cash Flow',
            line=dict(color='green')
        ))
        fig_fcf.update_layout(
            title="Free Cash Flow (Quarterly)", 
            yaxis_title="USD", 
            xaxis_title="Date"
        )
        st.plotly_chart(fig_fcf, use_container_width=True)

    # --- Net Profit Margin vs Sales ---
    if 'Total Revenue' in income_df.columns and 'Net Income' in income_df.columns:
        net_margin = income_df['Net Income'] / income_df['Total Revenue'] * 100
        fig_margin = go.Figure()

        fig_margin.add_trace(go.Scatter(
            x=income_df.index, 
            y=income_df['Total Revenue'], 
            mode='lines+markers',
            name='Total Revenue',
            yaxis='y1',
            line=dict(color='orange')
        ))

        fig_margin.add_trace(go.Scatter(
            x=income_df.index, 
            y=net_margin, 
            mode='lines+markers',
            name='Net Profit Margin (%)',
            yaxis='y2',
            line=dict(color='red')
        ))

        fig_margin.update_layout(
            title="Net Profit Margin vs Sales",
            xaxis_title="Date",
            yaxis=dict(
                title="Revenue (USD)",
                side='left'
            ),
            yaxis2=dict(
                title="Net Margin (%)",
                overlaying='y',
                side='right'
            ),
            legend=dict(x=0.01, y=1.15, orientation='h')
        )
        st.plotly_chart(fig_margin, use_container_width=True)

except Exception as e:
    st.warning(f"Unable to load financial visualizations: {e}")

# -------- Revenue Trends --------
try:
    st.subheader("📈 Revenue Trend")
    income_stmt = stock.financials
    if not income_stmt.empty:
        revenue = income_stmt.loc['Total Revenue'].dropna()
        revenue.index = pd.to_datetime(revenue.index)
        revenue = revenue.sort_index()
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(x=revenue.index, y=revenue.values / 1e9, mode='lines+markers', name='Revenue'))
        fig_rev.update_layout(title="Revenue Trend (in Billions)", xaxis_title="Year", yaxis_title="Revenue (B USD)")
        st.plotly_chart(fig_rev, use_container_width=True)
    else:
        st.warning("Revenue data not available.")
except Exception as e:
    st.error(f"Failed to load revenue trend: {e}")


# -------------------- FINANCIAL METRICS & KEY RATIOS --------------------
st.header("📉 Financial Metrics & Key Ratios")

@st.cache_data
def get_clean_ratios(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    try:
        total_debt = stock.balance_sheet.loc['Total Debt'].iloc[0]
        cash = stock.balance_sheet.loc['Cash'].iloc[0]
        total_equity = stock.balance_sheet.loc['Total Stockholder Equity'].iloc[0]
        ebit = stock.financials.loc['Ebit'].iloc[0]
        operating_income = stock.financials.loc['Operating Income'].iloc[0]
        total_assets = stock.balance_sheet.loc['Total Assets'].iloc[0]

        roic = (operating_income / (total_assets - cash)) if total_assets and cash else None
        net_debt = total_debt - cash
        net_debt_to_equity = (net_debt / total_equity) if total_equity else None
        op_margin = info.get("operatingMargins")

    except:
        roic = None
        net_debt_to_equity = None
        op_margin = None

    ratios = {
        "P/E Ratio": info.get("trailingPE"),
        "EV/EBITDA": info.get("enterpriseToEbitda"),
        "Price to Book": info.get("priceToBook"),
        "ROIC (%)": roic * 100 if roic else None,
        "Net Debt to Equity (%)": net_debt_to_equity * 100 if net_debt_to_equity else None,
        "Operating Profit Margin (%)": op_margin * 100 if op_margin else None,
    }
    return ratios

ratios = get_clean_ratios(ticker)

# --- Group ratios ---
percent_ratios = {k: v for k, v in ratios.items() if "%" in k and v is not None}
multiple_ratios = {k: v for k, v in ratios.items() if "%" not in k and v is not None}

# --- Plot: Percentage-based ratios ---
if percent_ratios:
    st.subheader("📊 Percentage-Based Ratios")
    fig_percent = go.Figure([go.Bar(
        x=list(percent_ratios.keys()),
        y=list(percent_ratios.values()),
        marker_color='seagreen'
    )])
    fig_percent.update_layout(
        title="Key Performance (%-based)",
        yaxis_title="Percent",
        xaxis_title="Metric"
    )
    st.plotly_chart(fig_percent, use_container_width=True)

# --- Plot: Multiples ---
if multiple_ratios:
    st.subheader("📊 Market Valuation Multiples")
    fig_mult = go.Figure([go.Bar(
        x=list(multiple_ratios.keys()),
        y=list(multiple_ratios.values()),
        marker_color='indianred'
    )])
    fig_mult.update_layout(
        title="Market Valuation Metrics (Multiples)",
        yaxis_title="Value",
        xaxis_title="Ratio"
    )
    st.plotly_chart(fig_mult, use_container_width=True)


# -------------------- KPI DASHBOARD --------------------
st.subheader("📌 Key Performance Indicators (KPIs)")

kpi_options = [
    "Revenue", 
    "Net Profit Margin (%)", 
    "EPS (Earnings Per Share)", 
    "Free Cash Flow (FCF)", 
    "Return on Equity (%)", 
    "Revenue Growth (%)"
]

selected_kpis = st.multiselect("Select KPIs to display:", options=kpi_options, default=kpi_options)

# Safely extract KPIs
def safe_get(source, keys):
    for key in keys:
        if key in source:
            return source[key]
    return None

info = stock.info
financials = stock.financials
cashflow = stock.cashflow
balance_sheet = stock.balance_sheet

# Compute KPIs
kpis = {
    "Revenue": safe_get(info, ['totalRevenue']),
    "Net Profit Margin (%)": safe_get(info, ['netMargins']) * 100 if safe_get(info, ['netMargins']) is not None else None,
    "EPS (Earnings Per Share)": safe_get(info, ['trailingEps']),
    "Free Cash Flow (FCF)": (
        safe_get(cashflow, ['Total Cash From Operating Activities']) - 
        safe_get(cashflow, ['Capital Expenditures']) 
        if all(k in cashflow.index for k in ['Total Cash From Operating Activities', 'Capital Expenditures']) 
        else None
    ),
    "Return on Equity (%)": safe_get(info, ['returnOnEquity']) * 100 if safe_get(info, ['returnOnEquity']) is not None else None,
    "Revenue Growth (%)": safe_get(info, ['revenueGrowth']) * 100 if safe_get(info, ['revenueGrowth']) is not None else None,
}

# Render KPI cards
cols = st.columns(len(selected_kpis))
for col, kpi in zip(cols, selected_kpis):
    val = kpis.get(kpi)
    if val is not None:
        if "($" in kpi or "Revenue" in kpi or "FCF" in kpi:
            val_str = f"${val:,.2f}B" if abs(val) > 1e9 else f"${val:,.2f}M"
        elif "%" in kpi:
            val_str = f"{val:.2f}%"
        else:
            val_str = f"${val:.2f}" if isinstance(val, (float, int)) else str(val)
    else:
        val_str = "N/A"
    col.metric(label=kpi, value=val_str)

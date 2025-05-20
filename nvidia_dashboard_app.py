import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import plotly.express as px


NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]



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



# -------------------- Real-Time News Feed --------------------
import requests
import yfinance as yf
from datetime import datetime

st.header("📰 Real-Time News Feed")

def fetch_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        query = stock.info.get("longName", ticker)  # Use full company name if available
        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={st.secrets['NEWSAPI_KEY']}"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])[:5]
        return articles
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        return []

articles = fetch_news(ticker)

if articles:
    for article in articles:
        published_at = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
        st.markdown(f"**[{article['title']}]({article['url']})**  \n*{article['source']['name']} - {published_at.strftime('%b %d, %Y %H:%M')}*  \n{article['description']}\n")
        st.divider()
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
    stock = yf.Ticker(ticker)
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

# 📊 Market Valuation Multiples
st.subheader("📊 Market Valuation Multiples")

# Select only relevant valuation multiples
valuation_keys = {
    "P/E Ratio": "Trailing P/E",
    "Price to Book": "Price to Book",
    "EV/EBITDA": "Enterprise to EBITDA"
}

valuation_data = {}
for label, key in valuation_keys.items():
    val = ratios.get(key)
    if val is not None:
        valuation_data[label] = val

if valuation_data:
    fig1 = px.bar(
        x=list(valuation_data.keys()),
        y=list(valuation_data.values()),
        labels={"x": "", "y": "Value"},
        text_auto=True
    )
    fig1.update_layout(title="Market Valuation Metrics (Multiples)", xaxis_title="", yaxis_title="Value")
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("⚠️ No market valuation data available.")



# 📈 Profitability & Leverage Ratios
st.subheader("📈 Profitability & Leverage Ratios")

# Define the ratios you want to show
profitability_keys = {
    "Return on Equity (ROE)": "Return on Equity (ROE)",
    "Operating Margin (%)": "Operating Margins"
}
leverage_keys = {
    "Net Debt to Equity": "Debt to Equity"
}

# Extract values
profitability_data = {}
leverage_data = {}

for label, key in profitability_keys.items():
    val = ratios.get(key)
    if val is not None:
        profitability_data[label] = val

for label, key in leverage_keys.items():
    val = ratios.get(key)
    if val is not None:
        leverage_data[label] = val

# Plot side-by-side charts only if data exists
if profitability_data or leverage_data:
    col1, col2 = st.columns(2)

    with col1:
        if profitability_data:
            fig_profit = px.bar(
                x=list(profitability_data.keys()),
                y=list(profitability_data.values()),
                labels={"x": "", "y": "%"},
                text_auto=True
            )
            fig_profit.update_layout(title="Profitability Ratios")
            st.plotly_chart(fig_profit, use_container_width=True)
        else:
            st.warning("⚠️ No profitability data available.")

    with col2:
        if leverage_data:
            fig_leverage = px.bar(
                x=list(leverage_data.keys()),
                y=list(leverage_data.values()),
                labels={"x": "", "y": "×"},
                text_auto=True
            )
            fig_leverage.update_layout(title="Leverage Ratios")
            st.plotly_chart(fig_leverage, use_container_width=True)
        else:
            st.warning("⚠️ No leverage data available.")
else:
    st.warning("⚠️ No profitability or leverage data available.")



# -------------------- KPI DASHBOARD --------------------
st.subheader("📌 Key Performance Indicators (KPIs)")

stock = yf.Ticker(ticker)
info = stock.info
cashflow = stock.cashflow
financials = stock.financials

# Safely compute FCF
def get_fcf(cashflow):
    try:
        if "Total Cash From Operating Activities" in cashflow.index and "Capital Expenditures" in cashflow.index:
            op = cashflow.loc["Total Cash From Operating Activities"].dropna()
            capex = cashflow.loc["Capital Expenditures"].dropna()
            if not op.empty and not capex.empty:
                return op.iloc[0] - capex.iloc[0]
        elif "Operating Cash Flow" in cashflow.index and "Capital Expenditures" in cashflow.index:
            op = cashflow.loc["Operating Cash Flow"].dropna()
            capex = cashflow.loc["Capital Expenditures"].dropna()
            if not op.empty and not capex.empty:
                return op.iloc[0] - capex.iloc[0]
    except:
        return None


# Fallback net profit margin calculation
def get_net_profit_margin(info, financials):
    net_income = financials.loc["Net Income"].dropna().iloc[0] if "Net Income" in financials.index else None
    revenue = info.get("totalRevenue", None)
    if net_income is not None and revenue:
        return net_income / revenue
    return info.get("netMargins", None)

kpi_data = {
    "Revenue": {
        "value": info.get("totalRevenue"),
        "format": "${:,.2f}B",
        "divisor": 1e9
    },
    "Net Profit Margin (%)": {
        "value": get_net_profit_margin(info, financials),
        "format": "{:.2%}",
        "divisor": 1
    },
    "EPS (Earnings Per Share)": {
        "value": info.get("trailingEps", None),
        "format": "${:,.2f}",
        "divisor": 1
    },
    "Free Cash Flow (FCF)": {
        "value": get_fcf(cashflow),
        "format": "${:,.2f}B",
        "divisor": 1e9
    },
    "Return on Equity (%)": {
        "value": info.get("returnOnEquity", None),
        "format": "{:.2%}",
        "divisor": 1
    },
    "Revenue Growth (%)": {
        "value": info.get("revenueGrowth", None),
        "format": "{:.2%}",
        "divisor": 1
    },
}

# Dropdown for selection
selected_kpis = st.multiselect("Select KPIs to display:", list(kpi_data.keys()), default=list(kpi_data.keys()))

# Display KPIs
cols = st.columns(len(selected_kpis))
for i, kpi in enumerate(selected_kpis):
    with cols[i]:
        val = kpi_data[kpi]["value"]
        if val is not None and pd.notnull(val):
            formatted = kpi_data[kpi]["format"].format(val / kpi_data[kpi]["divisor"])
            st.metric(kpi, formatted)
        else:
            st.metric(kpi, "N/A")

# --- Investment Thesis & Upside Potential ---
# --- Investment Thesis & Upside Potential ---
st.markdown("## 💡 Investment Thesis & Upside Potential")

if 'intrinsic_value' in locals() and 'current_price' in locals() and intrinsic_value and current_price:
    upside_pct = ((intrinsic_value - current_price) / current_price) * 100
    if upside_pct > 20:
        st.success(f"""
        **Strong Upside**: Based on our DCF valuation, the intrinsic value (${intrinsic_value:,.2f}) is **{upside_pct:.1f}%** higher than the current price (${current_price:,.2f}).

        This indicates significant upside potential driven by robust fundamentals, growth expectations, and market leadership.
        """)
    elif upside_pct > 0:
        st.info(f"""
        **Modest Upside**: The intrinsic value (${intrinsic_value:,.2f}) is slightly higher than the current price (${current_price:,.2f}), with an upside of **{upside_pct:.1f}%**.

        There is potential for long-term gains, especially if current growth trajectories hold.
        """)
    else:
        st.warning(f"""
        **Limited Upside**: The intrinsic value (${intrinsic_value:,.2f}) is below the current price (${current_price:,.2f}), suggesting downside risk of **{abs(upside_pct):.1f}%**.

        Investors should proceed with caution or wait for a better entry point.
        """)
else:
    st.warning("⚠️ Intrinsic value or current price is not available to assess upside potential.")


# --- Risks & Concerns ---
# --- Risks & Concerns ---
st.markdown("## ⚠️ Risks & Concerns")

try:
    debt_equity = key_metrics.get("Debt to Equity", None)  # or ratios.get(...) depending on your setup

    if debt_equity and debt_equity > 1:
        st.error(f"""
        **High Leverage Risk**: The company has a Debt-to-Equity ratio of **{debt_equity:.2f}**, indicating it is significantly leveraged.

        High debt levels could expose it to refinancing risks, especially in a rising interest rate environment.
        """)
    else:
        st.info("No significant debt-related risks detected based on current ratios.")

except Exception as e:
    st.warning(f"Could not assess risk metrics: {e}")

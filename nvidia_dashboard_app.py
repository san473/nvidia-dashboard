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

# --- Investment Thesis ---
st.markdown("## 💡 Investment Thesis & Upside Potential")

try:
    if stock and stock.info:
        name = stock.info.get("longName", ticker)
        sector = stock.info.get("sector", "N/A")
        rev_growth = stock.info.get("revenueGrowth", 0)
        net_margin = stock.info.get("profitMargins", 0)
        roe = stock.info.get("returnOnEquity", 0)
        roic = stock.info.get("returnOnAssets", 0)
        fwd_pe = stock.info.get("forwardPE", None)
        peg = stock.info.get("pegRatio", None)

        thesis = []

        if rev_growth and rev_growth > 0.1:
            thesis.append(f"- **Strong Revenue Growth**: Revenue is growing at **{rev_growth*100:.1f}% YoY**, indicating business expansion.")
        if net_margin and net_margin > 0.15:
            thesis.append(f"- **Healthy Profitability**: Net profit margins are **{net_margin*100:.1f}%**, signaling operational strength.")
        if roe and roe > 0.15:
            thesis.append(f"- **High Return on Equity**: ROE of **{roe*100:.1f}%** implies effective use of shareholder capital.")
        if roic and roic > 0.1:
            thesis.append(f"- **Efficient Capital Use**: ROIC at **{roic*100:.1f}%** reflects good return on invested assets.")
        if peg and peg < 1:
            thesis.append(f"- **Undervalued on PEG**: PEG ratio of **{peg:.2f}** may indicate undervaluation relative to growth.")
        if sector == "Technology":
            thesis.append(f"- **Strategic Moat**: As a tech sector leader, {name} is positioned to benefit from long-term digital trends like AI, cloud, and automation.")

        if thesis:
            st.success("**Investment Thesis:**\n" + "\n".join(thesis))
        else:
            st.info("No compelling thesis could be generated from available data.")
    else:
        st.warning("⚠️ Company info not available.")
except Exception as e:
    st.warning(f"Could not generate investment thesis: {e}")

# --- Extract Key Financial Ratios (required for risk assessment) ---
try:
    fin_ratios = stock.info  # or wherever you assign stock info via yfinance
    key_metrics = {
        "Debt to Equity": fin_ratios.get("debtToEquity", None),
        "Profit Margin": fin_ratios.get("profitMargins", None),
        "Current Ratio": fin_ratios.get("currentRatio", None),
        "Return on Equity": fin_ratios.get("returnOnEquity", None),
    }
except Exception as e:
    key_metrics = {}
    st.warning(f"Could not extract key metrics: {e}")

# --- Risks & Concerns ---
st.markdown("## ⚠️ Risks & Concerns")

try:
    risk_points = []

    # --- Fallback-safe access to financial metrics ---
    debt_equity = key_metrics.get("Debt to Equity", None)
    profit_margin = key_metrics.get("Profit Margin", None)
    current_ratio = key_metrics.get("Current Ratio", None)

    # Risk 1: High leverage
    if debt_equity is not None and debt_equity > 1.0:
        risk_points.append(f"- **High Leverage**: Debt-to-Equity is **{debt_equity:.2f}**, suggesting reliance on debt.")

    # Risk 2: Margin Compression
    if profit_margin is not None and profit_margin < 0.1:
        risk_points.append(f"- **Low Profitability**: Profit margin is **{profit_margin:.2%}**, indicating margin pressure.")

    # Risk 3: Weak Liquidity
    if current_ratio is not None and current_ratio < 1.0:
        risk_points.append(f"- **Liquidity Risk**: Current ratio is **{current_ratio:.2f}**, suggesting short-term financial stress.")

    # Risk 4: Free Cash Flow (with safe lookup)
    try:
        cf_df = stock.cashflow
        op_cash = cf_df.loc[cf_df.index.str.contains("Total Cash From Operating Activities", case=False)].dropna()
        capex = cf_df.loc[cf_df.index.str.contains("Capital Expenditures", case=False)].dropna()

        if not op_cash.empty and not capex.empty:
            fcf_latest = op_cash.iloc[0] - abs(capex.iloc[0])
            if fcf_latest < 0:
                risk_points.append(f"- **Negative Free Cash Flow**: Latest FCF is **${fcf_latest:,.0f}**, indicating cash burn.")
    except Exception:
        pass  # Safe fallback if missing cashflow

    # Risk 5: Peer comparison (optional)
    if peers_df is not None and not peers_df.empty:
        try:
            industry_avg_margin = peers_df["Profit Margin"].dropna().mean()
            if profit_margin is not None and industry_avg_margin and profit_margin < industry_avg_margin * 0.8:
                risk_points.append("- **Competitive Disadvantage**: Profit margin is significantly below industry peers.")
        except Exception:
            pass

    # Output block
    if risk_points:
        st.error("**Identified Risk Factors:**\n\n" + "\n".join(risk_points))
    else:
        st.info("No material red flags identified in debt, liquidity, or free cash flow.")

except Exception as e:
    st.warning(f"⚠️ Could not generate risk section: {e}")

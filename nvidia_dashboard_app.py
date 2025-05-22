import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import plotly.express as px
from datetime import datetime
from streamlit.runtime.caching import cache_data
import requests

import pandas as pd

st.set_page_config(page_title="📈 Stock Dashboard", layout="wide")
st.cache_data.clear()
@st.cache_data
# REMOVE cache temporarily to force reload
def load_sp500_data():
    df = pd.read_excel("sp500_companies.xlsx")
    df.columns = df.columns.str.strip()
    st.write("✅ Columns loaded from Excel:", df.columns.tolist())  # DEBUG
    return df

sp500_df = load_sp500_data()
# Check if 'Market Cap' exists
if 'Market Cap' not in sp500_df.columns:
    st.error("❌ 'Market Cap' column not found. Here are the columns:")
    for col in sp500_df.columns:
        st.write(repr(col))  # Shows hidden characters






sp500_df = load_sp500_data()
sp500_df.columns = sp500_df.columns.str.strip().str.lower()





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

# ------------------- DCF VALUATION -------------------
st.header("💰 Discounted Cash Flow (DCF) Valuation")

# --- User-Defined Assumptions ---
st.subheader("📈 DCF Assumptions")
col1, col2, col3 = st.columns(3)
with col1:
    forecast_years = st.slider("Forecast Years", min_value=3, max_value=10, value=5)
with col2:
    growth_rate = st.slider("FCF Growth Rate (%)", min_value=0.0, max_value=20.0, value=10.0, step=0.5) / 100
with col3:
    discount_rate = st.slider("Discount Rate / WACC (%)", min_value=5.0, max_value=15.0, value=9.0, step=0.5) / 100

# --- DCF Calculation ---
def calculate_dcf(ticker, forecast_years, growth_rate, discount_rate):
    try:
        stock = yf.Ticker(ticker)
        cashflow = stock.cashflow

        st.write("📌 Available Cashflow Rows:", list(cashflow.index))

        if 'Total Cash From Operating Activities' in cashflow.index and 'Capital Expenditures' in cashflow.index:
            fcf = cashflow.loc['Total Cash From Operating Activities'] - cashflow.loc['Capital Expenditures']
        elif 'Operating Cash Flow' in cashflow.index and 'Capital Expenditures' in cashflow.index:
            fcf = cashflow.loc['Operating Cash Flow'] - cashflow.loc['Capital Expenditures']
            st.info("Using fallback row: **Operating Cash Flow**")
        else:
            st.warning("DCF valuation failed: Required cash flow rows not found.")
            return None

        fcf = fcf.dropna()
        if fcf.empty:
            st.warning("DCF valuation failed: FCF data is empty after dropna.")
            return None

        latest_fcf = fcf.iloc[0]
        terminal_growth_rate = 0.03

        projected_fcfs = [latest_fcf * (1 + growth_rate) ** i for i in range(1, forecast_years + 1)]
        discounted_fcfs = [val / (1 + discount_rate) ** i for i, val in enumerate(projected_fcfs, start=1)]
        terminal_value = projected_fcfs[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
        discounted_terminal = terminal_value / (1 + discount_rate) ** forecast_years

        enterprise_value = sum(discounted_fcfs) + discounted_terminal
        debt = stock.balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in stock.balance_sheet.index else 0
        cash = stock.balance_sheet.loc['Cash'].iloc[0] if 'Cash' in stock.balance_sheet.index else 0
        shares_outstanding = stock.info.get('sharesOutstanding', 0)

        equity_value = enterprise_value - debt + cash
        fair_value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else None

        return {
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'fair_value_per_share': fair_value_per_share
        }

    except Exception as e:
        st.error(f"DCF valuation failed due to error: {e}")
        return None

# --- Run and Display DCF ---
dcf_result = calculate_dcf(ticker, forecast_years, growth_rate, discount_rate)

if dcf_result and dcf_result['fair_value_per_share']:
    current_price = info.get('currentPrice', 0)
    fair_value = dcf_result['fair_value_per_share']

    st.markdown(f"""
    ### 💡 DCF Summary  
    - Forecast Period: `{forecast_years}` years  
    - FCF Growth Rate: `{growth_rate * 100:.1f}%`  
    - Discount Rate (WACC): `{discount_rate * 100:.1f}%`  
    - Terminal Growth Rate: `3.0%`  
    """)

    st.success(f"**Intrinsic Value Estimate (DCF): ${dcf_result['equity_value']:,.2f}**")
    st.info(f"**Current Market Price:** ${current_price:,.2f}")
    st.success(f"**Intrinsic Value per Share:** ${fair_value:,.2f}")

    if fair_value > current_price:
        st.markdown("✅ The stock appears **undervalued** based on DCF.")
    else:
        st.markdown("⚠️ The stock appears **overvalued** based on DCF.")
else:
    st.warning("DCF valuation data not available.")



# --------------------- Dynamic Peer Comparison ---------------------
# 🧠 Dynamic Peer Comparison (Updated to match cleaned column names)
with st.container():
    st.markdown("## 📊 Peer Comparison")

    try:
        # Standardize the column names
        sp500_df.columns = sp500_df.columns.str.strip().str.lower()

        # Ensure all string columns are lowercase for matching
        sp500_df['symbol'] = sp500_df['symbol'].str.upper()
        sp500_df['sector'] = sp500_df['sector'].str.strip()

        current_row = sp500_df[sp500_df['symbol'] == selected_ticker.upper()]
        
        if not current_row.empty:
            current_sector = current_row['sector'].values[0]
            current_marketcap = float(str(current_row['marketcap'].values[0]).replace(",", "").replace("₹", "").replace("$", "").strip())

            # Filter peers within same sector and ±30% market cap
            peers = sp500_df[
                (sp500_df['sector'] == current_sector) &
                (sp500_df['symbol'] != selected_ticker.upper())
            ].copy()

            peers['marketcap_numeric'] = peers['marketcap'].astype(str).str.replace(",", "").astype(float)
            peers = peers[
                (peers['marketcap_numeric'] >= current_marketcap * 0.7) &
                (peers['marketcap_numeric'] <= current_marketcap * 1.3)
            ]

            if peers.empty:
                st.warning("⚠️ No comparable peers found.")
            else:
                peer_symbols = peers['symbol'].tolist()
                selected_peers = st.multiselect("Select Peers", peer_symbols, default=peer_symbols[:3])
                
                if selected_peers:
                    st.dataframe(peers[peers['symbol'].isin(selected_peers)].set_index("symbol"))
                else:
                    st.info("ℹ️ Please select at least one peer to compare.")
        else:
            st.warning(f"⚠️ Ticker '{selected_ticker}' not found in the S&P 500 list.")

    except KeyError as e:
        st.error(f"❌ Column not found: {e}. Here are the available columns:\n\n{sp500_df.columns.tolist()}")
    except Exception as e:
        st.error(f"❌ Error finding peers: {e}")



# -------------------- Real-Time News Feed --------------------
st.header("🧠 Market News Summary")
ticker = st.session_state.get("selected_ticker", "AAPL")

@st.cache_data(ttl=60)
def fetch_news_articles(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWSAPI_KEY}&sortBy=publishedAt&language=en&pageSize=5"
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
        return articles
    except Exception:
        return []

def simple_logical_summary(articles):
    if not articles:
        return "No recent news found."

    titles = [a['title'] for a in articles if a.get('title')]
    highlights = "\n".join([f"- {title}" for title in titles[:5]])

    return f"""
### ✅ Key Headlines for {ticker}
{highlights}

Note: Full AI summarization has been temporarily disabled to reduce memory usage.
"""

articles = fetch_news_articles(ticker)
summary = simple_logical_summary(articles)

st.markdown("### 🧠 Summary of Key Headlines")
st.markdown(summary)

st.subheader("📰 Full Headlines")
if articles:
    for article in articles:
        st.markdown(f"- [{article['title']}]({article['url']}) — `{article['source']['name']}`")
else:
    st.warning("No news articles found.")




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

# --- ⚠️ Risks & Concerns ---
st.markdown("## ⚠️ Risks & Concerns")

try:
    # Define fallback peer tickers if not already defined
    if 'peer_tickers' not in locals():
        peer_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # fallback group
    st.caption(f"Peer group used: {', '.join(peer_tickers)}")

    # Load peer data (ensure yf.download or cache is already available)
    peer_data = yf.download(peer_tickers, period="1y", group_by='ticker', progress=False)
    
    # Calculate average revenue growth and margin for peers
    peer_revenue_growth = []
    peer_net_margins = []

    for pt in peer_tickers:
        pt_ticker = yf.Ticker(pt)
        pt_financials = pt_ticker.financials
        pt_income = pt_financials if isinstance(pt_financials, pd.DataFrame) else pt_financials.get('incomeStatementHistory', {})
        if pt_income is not None and "Total Revenue" in pt_income:
            rev = pt_income.loc["Total Revenue"].values
            if len(rev) >= 2 and rev[0] and rev[1]:
                growth = (rev[0] - rev[1]) / rev[1]
                peer_revenue_growth.append(growth)

        if pt_income is not None and "Net Income" in pt_income and "Total Revenue" in pt_income:
            net_income = pt_income.loc["Net Income"].values[0]
            total_rev = pt_income.loc["Total Revenue"].values[0]
            if net_income and total_rev:
                margin = net_income / total_rev
                peer_net_margins.append(margin)

    avg_peer_growth = np.mean(peer_revenue_growth) if peer_revenue_growth else None
    avg_peer_margin = np.mean(peer_net_margins) if peer_net_margins else None

    # Load target company data
    target = yf.Ticker(ticker)
    income_stmt = target.financials
    balance_sheet = target.balance_sheet
    cashflow = target.cashflow

    # 1. Leverage Risk
    if key_metrics.get("Debt to Equity", 0) > 2:
        st.error(f"**Leverage Risk:** Debt-to-Equity ratio is **{key_metrics['Debt to Equity']:.2f}**, indicating potential over-leverage.")

    # 2. Earnings Volatility (based on quarterly EPS)
    try:
        eps = target.earnings
        if not eps.empty and len(eps) > 4:
            earnings_std = eps['Earnings'].pct_change().std()
            if earnings_std > 0.5:
                st.error(f"**Earnings Volatility Risk:** High variability in quarterly earnings. Std Dev: **{earnings_std:.2f}**")
    except:
        pass

    # 3. Revenue Decline
    try:
        revenue = income_stmt.loc["Total Revenue"]
        if revenue.iloc[0] < revenue.iloc[1]:
            st.error(f"**Revenue Risk:** Revenue declined from **${revenue.iloc[1]:,.0f}** to **${revenue.iloc[0]:,.0f}** YoY.")
    except:
        pass

    # 4. Margin Compression vs peers
    try:
        net_income = income_stmt.loc["Net Income"].iloc[0]
        total_revenue = income_stmt.loc["Total Revenue"].iloc[0]
        net_margin = net_income / total_revenue
        if avg_peer_margin and net_margin < avg_peer_margin * 0.8:
            st.error(f"**Margin Risk:** Net margin is **{net_margin:.2%}**, significantly below peer average **{avg_peer_margin:.2%}**.")
    except:
        pass

    # 5. Shareholder Dilution (check share count increase)
    try:
        shares = balance_sheet.loc["Ordinary Shares Number"]
        if shares.iloc[0] > shares.iloc[1]:
            dilution_pct = (shares.iloc[0] - shares.iloc[1]) / shares.iloc[1]
            st.error(f"**Dilution Risk:** Share count increased by **{dilution_pct:.1%}**, indicating potential dilution.")
    except:
        pass

    # 6. Liquidity Risk (current ratio)
    try:
        current_assets = balance_sheet.loc["Total Current Assets"].iloc[0]
        current_liab = balance_sheet.loc["Total Current Liabilities"].iloc[0]
        current_ratio = current_assets / current_liab
        if current_ratio < 1:
            st.error(f"**Liquidity Risk:** Current Ratio is **{current_ratio:.2f}**, indicating short-term liquidity pressure.")
    except:
        pass

except Exception as e:
    st.warning(f"⚠️ Could not generate risk section: {e}")

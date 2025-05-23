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

# REMOVE cache temporarily to force reload
# @st.cache_data
def load_sp500_data():
    df = pd.read_excel("sp500_companies.xlsx")
    return df

# Ticker input and validation
ticker = None
ticker_obj = None

ticker_input = st.text_input("Enter Ticker Symbol")

if ticker_input:
    ticker = ticker_input.upper()
    try:
        ticker_obj = yf.Ticker(ticker)
        # Try fetching cash flow to validate data availability
        if ticker_obj.cashflow is None or ticker_obj.cashflow.empty:
            st.warning("⚠️ Cash flow data not available for this ticker.")
            ticker_obj = None
    except Exception as e:
        st.error(f"❌ Failed to load ticker data: {e}")
        ticker_obj = None
COLUMN_DISPLAY_NAMES = {
    "symbol": "Symbol",
    "exchange": "Exchange",
    "shortname": "Short Name",
    "longname": "Long Name",
    "sector": "Sector",
    "industry": "Industry",
    "currentprice": "Current Price",
    "marketcap": "Market Cap",
    "ebitda": "EBITDA",
    "revenuegrowth": "Revenue Growth",
    "state": "State",
    "country": "Country",
    "weight": "Weight",
}

def display_dataframe_pretty(df, columns):
    return df[columns].rename(columns={col: COLUMN_DISPLAY_NAMES.get(col, col) for col in columns})
sp500_df = load_sp500_data()








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

ticker_input = st.text_input("Enter stock ticker (e.g., AAPL, NVDA, MSFT)", value="AAPL").upper()

ticker = ticker_input
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
    st.error(f"Failed to fetch data for ticker {ticker_input}. Error: {e}")



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

# === DYNAMIC KPI, INVESTMENT THESIS, RISKS BLOCK ===
try:
    info = yf.Ticker(ticker).info

    st.markdown("## 📊 Key Financial Metrics")
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Market Cap", f"${info.get('marketCap', 0):,}")
    kpi_cols[1].metric("PE Ratio (TTM)", f"{info.get('trailingPE', 'N/A')}")
    kpi_cols[2].metric("PEG Ratio", f"{info.get('pegRatio', 'N/A')}")

    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Price to Book", f"{info.get('priceToBook', 'N/A')}")
    kpi_cols[1].metric("Return on Equity", f"{info.get('returnOnEquity', 'N/A'):.2%}" if info.get("returnOnEquity") else "N/A")
    kpi_cols[2].metric("Debt to Equity", f"{info.get('debtToEquity', 'N/A')}")

    # --- Investment Thesis ---
    st.markdown("## 🧠 Investment Thesis & Upside Potential")
    thesis_points = []

    if info.get("growthQuarterlyRevenueYoy") and info["growthQuarterlyRevenueYoy"] > 0.1:
        thesis_points.append("• Strong revenue growth in recent quarters.")
    if info.get("returnOnEquity", 0) > 0.15:
        thesis_points.append("• High return on equity indicates efficient capital usage.")
    if info.get("grossMargins", 0) > 0.5:
        thesis_points.append("• Robust gross margins suggest solid product/service economics.")
    if info.get("totalCash", 0) > info.get("totalDebt", 0):
        thesis_points.append("• Healthy balance sheet with more cash than debt.")
    if info.get("forwardPE", 0) < info.get("trailingPE", 999):
        thesis_points.append("• Forward PE lower than trailing PE implies expected earnings growth.")

    if thesis_points:
        for point in thesis_points:
            st.markdown(point)
    else:
        st.markdown("*No strong upside signals identified based on current data.*")

    # --- Risks & Concerns ---
    st.markdown("## ⚠️ Risks & Concerns")
    risk_points = []

    if info.get("operatingMargins", 0) < 0.1:
        risk_points.append("• Low operating margins may signal profitability challenges.")
    if info.get("debtToEquity", 0) > 1.0:
        risk_points.append("• High debt-to-equity ratio suggests potential leverage risk.")
    if info.get("revenueGrowth", 0) < 0.05:
        risk_points.append("• Weak revenue growth could limit long-term upside.")
    if info.get("profitMargins", 0) < 0.05:
        risk_points.append("• Thin profit margins may be vulnerable to cost pressures.")
    if not info.get("freeCashflow"):
        risk_points.append("• Missing or negative free cash flow data.")

    if risk_points:
        for point in risk_points:
            st.markdown(point)
    else:
        st.markdown("*No major risks identified based on current financials.*")

except Exception as e:
    st.warning(f"Unable to generate KPI/Thesis/Risks: {e}")



import yfinance as yf
import pandas as pd

# ---------------------- Debug: Cash Flow Data Rows ----------------------
st.markdown("### 🔍 Debug: Available Financial Rows")
st.markdown("**Cash Flow Statement Rows:**")

if not ticker:
    st.warning("⚠️ Debug failed: No ticker entered.")
else:
    try:
        yf_ticker = yf.Ticker(ticker)
        cashflow = yf_ticker.cashflow

        if cashflow.empty:
            st.warning("⚠️ Debug failed: No cash flow data found for this ticker.")
        else:
            # Normalize index names (row labels)
            cashflow.index = cashflow.index.str.lower()
            detected_rows = list(cashflow.index)
            st.success("✅ Cash flow data retrieved.")
            st.write("Available rows:", detected_rows)

            # Keywords to detect FCF-related rows
            fcf_keywords = [
                "free cash flow", 
                "operating cash flow",
                "total cash from operating activities",
                "capital expenditures",
                "cash from operations",
                "net cash provided by operating activities"
            ]

            matched_rows = [row for row in detected_rows if any(key in row for key in fcf_keywords)]

            if matched_rows:
                st.success(f"✅ Detected FCF-related rows: {matched_rows}")
            else:
                st.warning("⚠️ Cash flow retrieved, but no Free Cash Flow rows matched expected patterns.")
    
    except Exception as e:
        st.error(f"❌ Debug failed: {e}")



# ------------------- DCF VALUATION -------------------
# Discounted Cash Flow (DCF) Valuation
with st.container():
    st.markdown("### 📊 Discounted Cash Flow (DCF) Valuation")

    try:
        cashflow = ticker.financials
        cashflow_data = cashflow.fillna(0)

        # Try multiple likely row names for FCF
        possible_fcf_labels = ['free cash flow', 'total free cash flow', 'freecashflow']
        fcf_row = next(
            (row for row in cashflow_data.index if any(label in row.lower() for label in possible_fcf_labels)),
            None
        )

        if not fcf_row:
            st.warning("⚠️ Could not locate 'Free Cash Flow' in the financial data.")
        else:
            free_cash_flows = cashflow_data.loc[fcf_row].dropna()
            fcf_values = free_cash_flows.values

            if len(fcf_values) == 0:
                st.warning("⚠️ No free cash flow data available to perform DCF.")
            else:
                avg_fcf = np.mean(fcf_values)

                # Original Toggle Interface (Restored)
                with st.expander("⚙️ Adjust DCF Assumptions"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        forecast_years = st.slider("Forecast Years", 3, 10, 5)
                    with col2:
                        growth_rate = st.slider("FCF Growth Rate (%)", 0, 20, 8)
                    with col3:
                        discount_rate = st.slider("Discount Rate (%)", 5, 15, 10)

                # Project FCFs
                projected_fcfs = [avg_fcf * ((1 + growth_rate / 100) ** i) for i in range(1, forecast_years + 1)]
                discounted_fcfs = [fcf / ((1 + discount_rate / 100) ** i) for i, fcf in enumerate(projected_fcfs, 1)]
                terminal_value = (projected_fcfs[-1] * (1 + growth_rate / 100)) / (discount_rate / 100)
                discounted_terminal = terminal_value / ((1 + discount_rate / 100) ** forecast_years)

                intrinsic_value = sum(discounted_fcfs) + discounted_terminal

                shares_outstanding = ticker.info.get("sharesOutstanding", None)
                if shares_outstanding:
                    intrinsic_per_share = intrinsic_value / shares_outstanding
                    current_price = ticker.info.get("currentPrice", 0)
                    delta = intrinsic_per_share - current_price
                    pct = delta / current_price * 100 if current_price else 0

                    st.metric(
                        label="📈 Intrinsic Value per Share",
                        value=f"${intrinsic_per_share:,.2f}",
                        delta=f"{pct:+.2f}%" if current_price else "N/A"
                    )
                else:
                    st.write(f"💰 **Intrinsic Value:** ${intrinsic_value:,.2f}")
                    st.info("Note: Shares outstanding not available to compute per-share value.")

    except Exception as e:
        st.error(f"Error in DCF calculation: {str(e)}")



# --------------------- Dynamic Peer Comparison ---------------------
# ----------------------- PEER COMPARISON -----------------------
st.header("📊 Peer Comparison")

try:
    selected_ticker = ticker_input  # or use: ticker = st.session_state.get("selected_ticker", "AAPL")
    current_row = sp500_df[sp500_df["symbol"].str.upper() == selected_ticker.upper()]
    
    if current_row.empty:
        st.warning(f"No data found for {selected_ticker}.")
    else:
        selected_industry = current_row["industry"].values[0]
        selected_sector = current_row["sector"].values[0]
        selected_marketcap = current_row["marketcap"].values[0]

        lower_bound = selected_marketcap * 0.4
        upper_bound = selected_marketcap * 2.5

        peer_df = sp500_df[
            (sp500_df["symbol"].str.upper() != selected_ticker.upper()) &
            (sp500_df["industry"] == selected_industry) &
            (sp500_df["marketcap"] >= lower_bound) &
            (sp500_df["marketcap"] <= upper_bound)
        ]

        # Fallback: sector-based peers if not enough industry matches
        if peer_df.empty:
            st.warning("⚠️ No comparable peers found using market cap range. Showing sector-based peers.")
            peer_df = sp500_df[
                (sp500_df["symbol"].str.upper() != selected_ticker.upper()) &
                (sp500_df["sector"] == selected_sector) &
                (sp500_df["marketcap"] >= lower_bound) &
                (sp500_df["marketcap"] <= upper_bound)
            ]

        if peer_df.empty:
            st.warning("⚠️ Still no comparable peers found.")
        else:
            peer_symbols = peer_df["symbol"].unique().tolist()
            selected_peers = st.multiselect("Select Peers", peer_symbols, default=peer_symbols[:5])

            if selected_peers:
                # Fetch real-time financial data from yfinance
                data = []
                for sym in selected_peers:
                    try:
                        stock = yf.Ticker(sym)
                        info = stock.info
                        fast_info = stock.fast_info
                        pe_ratio = info.get("trailingPE", None)
                        pb_ratio = info.get("priceToBook", None)
                        peg_ratio = info.get("pegRatio", None)
                        earnings_yield = (1 / pe_ratio) if pe_ratio and pe_ratio > 0 else None
                        ev_to_ebitda = info.get("enterpriseToEbitda", None)
                        ev_to_ebit = info.get("enterpriseToEbit", None)

                        data.append({
                            "symbol": sym,
                            "longname": info.get("longName", "N/A"),
                            "sector": info.get("sector", "N/A"),
                            "industry": info.get("industry", "N/A"),
                            "currentprice": fast_info.get("lastPrice", None),
                            "marketcap": info.get("marketCap", None),
                            "P/E": pe_ratio,
                            "P/B": pb_ratio,
                            "PEG": peg_ratio,
                            "Earnings Yield": earnings_yield,
                            "EV/EBITDA": ev_to_ebitda,
                            "EV/EBIT": ev_to_ebit,
                        })
                    except Exception as e:
                        st.error(f"Error loading data for {sym}: {e}")

                peer_comparison_df = pd.DataFrame(data)
                st.dataframe(peer_comparison_df)

except Exception as e:
    st.error(f"Error finding peers: {e}")

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

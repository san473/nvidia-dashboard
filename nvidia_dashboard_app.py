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
import altair as alt
import nltk
nltk.download('vader_lexicon')


st.set_page_config(page_title="📈 Stock Dashboard", layout="wide")
st.cache_data.clear()

# REMOVE cache temporarily to force reload
# @st.cache_data


def load_sp500_data():
    df = pd.read_excel("sp500_companies.xlsx")
    return df


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


NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    st.warning("Missing NewsAPI key. Using cached/stubbed response.")



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

ticker = None
ticker_obj = None

# Input field for ticker
ticker_input = st.text_input(
    "Enter stock ticker (e.g., AAPL, NVDA, MSFT)", value="AAPL"
).upper()
if ticker_input:
    ticker = ticker_input

# Attempt to fetch ticker data
try:
    ticker_obj = yf.Ticker(ticker)
    # Validate cash flow data
    if ticker_obj.cashflow is None or ticker_obj.cashflow.empty:
        st.warning("⚠️ Cash flow data not available for this ticker.")
        ticker_obj = None
except Exception as e:
    st.error(f"❌ Failed to load ticker data: {e}")
    ticker_obj = None

# News block function
def news_block(ticker):
    st.subheader("📰 Recent News")

    query = f"{ticker} stock"
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&sortBy=publishedAt&language=en&pageSize=5&apiKey={NEWSAPI_KEY}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])

        if not articles:
            st.info("No news articles found.")
            return

        for article in articles:
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.write(article["description"] or "")
            st.caption(f"Published at: {article['publishedAt'][:10]}")
            st.markdown("---")

    except Exception as e:
        st.error(f"❌ Failed to load news data: {e}")

# Show news if ticker is valid
if ticker:
    news_block(ticker)

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
import yfinance as yf
import streamlit as st


ticker_obj = yf.Ticker(ticker)
info = ticker_obj.info

st.subheader("📌 Key Performance Indicators (KPI)")

kpi_options = {
    "Revenue (TTM)": "revenue",
    "Net Profit Margin": "net_profit_margin",
    "EPS (TTM)": "eps",
    "ROE": "roe",
    "Earnings Growth (YoY)": "earnings_growth"
}

selected_kpis = st.multiselect("Select KPIs to display:", list(kpi_options.keys()), default=list(kpi_options.keys()))

# Fetch base metrics
revenue = info.get("totalRevenue")
net_income = info.get("netIncomeToCommon")
eps = info.get("trailingEps")
roe = info.get("returnOnEquity")
earnings_growth = info.get("earningsQuarterlyGrowth")

# Calculate net profit margin safely
net_profit_margin = (net_income / revenue) if revenue and net_income else None

# Display KPI cards
kpi_values = {
    "Revenue (TTM)": f"${revenue / 1e9:.2f}B" if revenue else "N/A",
    "Net Profit Margin": f"{net_profit_margin:.2%}" if net_profit_margin is not None else "N/A",
    "EPS (TTM)": f"${eps:.2f}" if eps else "N/A",
    "ROE": f"{roe:.2%}" if roe else "N/A",
    "Earnings Growth (YoY)": f"{earnings_growth:.2%}" if earnings_growth else "N/A"
}

cols = st.columns(len(selected_kpis))
for i, kpi in enumerate(selected_kpis):
    with cols[i]:
        st.metric(label=kpi, value=kpi_values[kpi])

# Ensure ticker is fetched safely
try:
    ticker_obj = yf.Ticker(ticker)
    financial_data = ticker_obj.info
    fin = financial_data
except Exception as e:
    st.error("Failed to load financial data.")
    fin = {}
    financial_data = {}
import streamlit as st

# ========== Extract Key Metrics ==========
long_name = financial_data.get("longName", ticker)
sector = financial_data.get("sector", "N/A")
industry = financial_data.get("industry", "N/A")
market_cap = financial_data.get("marketCap", 0)
revenue = financial_data.get("totalRevenue", 0)
eps = financial_data.get("trailingEps", None)
roe = financial_data.get("returnOnEquity", None)
earnings_growth = financial_data.get("earningsQuarterlyGrowth", None)
profit_margin = financial_data.get("profitMargins", None)
debt_to_equity = financial_data.get("debtToEquity", None)
current_ratio = financial_data.get("currentRatio", None)



# Load Lighthouse Canton commentary
@st.cache_data
def load_lighthouse_commentary():
    df = pd.read_excel("CIO Stocks.xlsx")
    df.columns = df.columns.str.strip().str.lower()
    return df

cio_df = load_lighthouse_commentary()
ticker_upper = ticker.upper()

# Match current ticker
matched = cio_df[cio_df["ticker"].str.upper().str.strip() == ticker_upper]

st.subheader("🧭 Lighthouse Canton View")

if not matched.empty:
    view = matched["lighthouse canton view"].values[0]
    target_price = matched["target price"].values[0]

    # Get current market price from yfinance
    try:
        ticker_data = yf.Ticker(ticker)
        current_price = ticker_data.info.get("regularMarketPrice", None)
    except Exception:
        current_price = None

    # Compute upside/downside
    if pd.notna(target_price) and current_price:
        upside_pct = ((target_price - current_price) / current_price) * 100
        direction = "Upside" if upside_pct >= 0 else "Downside"
        color = "green" if upside_pct >= 0 else "red"
    else:
        upside_pct = None

    # Display content
    st.markdown(f"**📌 View:** {view}")
    st.markdown(f"**🎯 Target Price:** ${target_price:,.2f}")
    
    if current_price:
        st.markdown(f"**💵 Current Price:** ${current_price:,.2f}")
        st.markdown(f"**📈 {direction} Potential:** :{color}[{upside_pct:.2f}%]")
    else:
        st.markdown("⚠️ Could not fetch current price.")
else:
    st.info("This stock is not present in the Lighthouse Canton coverage.")


# ========== Investment Thesis Summary ==========
thesis_points = []

if revenue and revenue > 1e9:
    thesis_points.append(f" Strong topline performance with trailing revenue of ${revenue/1e9:.2f}B suggests robust demand.")

if eps and eps > 0:
    thesis_points.append(f" Solid EPS of ${eps:.2f} indicates strong earnings power.")

if roe and roe > 0.15:
    thesis_points.append(f" Healthy Return on Equity (ROE) of {roe*100:.1f}% highlights efficient capital allocation.")

if earnings_growth and earnings_growth > 0:
    thesis_points.append(f" Positive quarterly earnings growth of {earnings_growth*100:.1f}% supports upside potential.")

if market_cap and market_cap > 50e9:
    thesis_points.append(f" Large-cap stability: {long_name} operates at a market cap above $50B, enhancing institutional confidence.")

if len(thesis_points) < 3:
    thesis_points.append(" Limited available financial highlights — further analysis recommended.")

# ========== Risk Summary ==========
risk_points = []

if earnings_growth is not None and earnings_growth < 0:
    risk_points.append(f" Negative earnings growth ({earnings_growth*100:.1f}%) may signal performance headwinds.")

if profit_margin is not None and profit_margin < 0.05:
    risk_points.append(f" Thin profit margins ({profit_margin*100:.1f}%) could limit scalability.")

if roe is not None and roe < 0.05:
    risk_points.append(f" Weak Return on Equity ({roe*100:.1f}%) may suggest inefficient operations.")

if debt_to_equity is not None and debt_to_equity > 100:
    risk_points.append(f" Elevated debt-to-equity ratio ({debt_to_equity:.0f}%) increases financial risk.")

if current_ratio is not None and current_ratio < 1:
    risk_points.append(f" Current ratio below 1.0 raises concerns over short-term liquidity.")

if len(risk_points) < 3:
    risk_points.append(" No major red flags from available metrics — monitor quarterly updates.")

# ========== Render in Streamlit ==========
st.markdown("### 📈 Investment Thesis & Upside Potential")
for point in thesis_points:
    st.markdown(f"- {point}")

st.markdown("### ⚠️ Risks & Concerns")
for point in risk_points:
    st.markdown(f"- {point}")


   


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
# -------- CONFIG --------
NEWS_API_KEY = st.secrets["NEWSAPI_KEY"]

NEWS_API_URL = "https://newsapi.org/v2/everything"

# -------- COMPANY NAME --------
@st.cache_data(ttl=3600)
def get_company_name_from_ticker(ticker):
    try:
        return yf.Ticker(ticker).info.get("longName", ticker)
    except Exception:
        return ticker

# -------- FETCH NEWS --------
@st.cache_data(ttl=60)
def get_news_articles(company_name):
    today = datetime.today()
    last_week = today - timedelta(days=7)
    params = {
        "q": company_name,
        "from": last_week.strftime("%Y-%m-%d"),
        "sortBy": "relevancy",
        "language": "en",
        "apiKey": NEWS_API_KEY,
        "pageSize": 20,
    }
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])
    return []

# -------- CATEGORIZE NEWS --------
def categorize_articles(articles):
    positive, negative, emerging = [], [], []
    for article in articles:
        title = article["title"].lower()
        description = article.get("description", "").lower()

        text = f"{title} {description}"
        if any(word in text for word in ["beats", "growth", "record", "surge", "upgrade", "strong", "gain", "boost"]):
            positive.append(article)
        elif any(word in text for word in ["misses", "decline", "drop", "downgrade", "loss", "concern", "crisis", "lawsuit"]):
            negative.append(article)
        else:
            emerging.append(article)

    return positive, negative, emerging

# -------- MAIN BLOCK --------
def news_block(ticker):
    company_name = get_company_name_from_ticker(ticker)

    st.markdown("### 🧠 Market News Summary")

    articles = get_news_articles(company_name)

    if not articles:
        st.markdown(f"**Summary of Key Headlines**")
        st.warning(f"No recent news found for **{company_name}**.")
        return

    st.markdown("### 🧠 Summary of Key Headlines")

    # Categorize articles
    pos, neg, emerging = categorize_articles(articles)

    def render_section(title, icon, items):
        st.markdown(f"#### {icon} {title}")
        if not items:
            st.info(f"No {title.lower()} found.")
        else:
            for article in items[:3]:
                st.markdown(f"- **{article['title']}** ({article['source']['name']})")

    render_section("Positive Developments", "📈", pos)
    render_section("Risks & Negative Sentiment", "⚠️", neg)
    render_section("Emerging Themes", "🧩", emerging)

    # Full headline list
    st.markdown("### 📰 Full Headlines")
    if not articles:
        st.warning("No news articles found.")
    else:
        for article in articles:
            st.markdown(f"**[{article['title']}]({article['url']})**  \n*{article['source']['name']} | {article['publishedAt'][:10]}*  \n{article.get('description', '')}")
            st.markdown("---")


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






st.write("Cashflow index labels:", [str(col).lower() for col in cashflow.columns])


def find_capex_row(cashflow_df):
    """Find the correct CapEx row in a case-insensitive and flexible way."""
    capex_keywords = [
        "capital expenditures", "capital expenditure", "capex",
        "purchase of property and equipment", "purchase of property plant and equipment",
        "purchase of ppe"
    ]
    for possible in capex_keywords:
        for row in cashflow_df.index:
            if isinstance(row, str) and possible.lower() in row.lower():
                return row
    return None




st.subheader("💰 Free Cash Flow Analysis")

ticker = st.text_input("Enter Ticker Symbol", "AAPL")
stock = yf.Ticker(ticker)

def find_column_label(df, keywords):
    """Search column names for relevant keywords (case insensitive)."""
    for col in df.columns:
        for keyword in keywords:
            if keyword in col.lower():
                return col
    return None

try:
    cashflow = stock.cashflow.T
    income_stmt = stock.income_stmt.T

    if cashflow.empty or income_stmt.empty:
        st.warning("Missing financial data.")
        st.stop()

    capex_keywords = ["capital expenditures", "capex", "purchase of property", "capital expenditure"]
    opcf_keywords = ["operating cash flow", "total cash from operating activities", "net cash provided by operating activities"]

    capex_col = find_column_label(cashflow, capex_keywords)
    opcf_col = find_column_label(cashflow, opcf_keywords)

    if not capex_col or not opcf_col:
        st.write("Cashflow Columns:", list(cashflow.columns))  # TEMP DEBUG
        st.warning("❌ Unable to find required CapEx or Operating Cash Flow columns.")
        st.stop()

    capex = cashflow[capex_col]
    op_cf = cashflow[opcf_col]
    fcf = op_cf + capex  # CapEx is negative

    fcf = fcf.dropna()
    fcf.index = pd.to_datetime(fcf.index).year
    fcf = fcf.sort_index()

    # Revenue and Net Income
    revenue = None
    net_income = None
    for r in ["Total Revenue", "Revenue"]:
        if r in income_stmt.columns:
            revenue = income_stmt[r]
            break
    for n in ["Net Income", "Net Income Applicable to Common Shares"]:
        if n in income_stmt.columns:
            net_income = income_stmt[n]
            break

    if revenue is None or net_income is None:
        st.warning("Unable to retrieve revenue or net income.")
        st.stop()

    revenue = revenue.dropna()
    revenue.index = pd.to_datetime(revenue.index).year
    net_income = net_income.dropna()
    net_income.index = pd.to_datetime(net_income.index).year

    # Align all three
    common_index = fcf.index.intersection(revenue.index).intersection(net_income.index)
    if len(common_index) == 0:
        st.warning("⚠️ No overlapping years found for FCF, Revenue, and Net Income.")
        st.stop()

    fcf = fcf.reindex(common_index)
    revenue = revenue.reindex(common_index)
    net_income = net_income.reindex(common_index)

    # Metrics
    last_fcf = fcf.iloc[-1]
    last_revenue = revenue.iloc[-1]
    last_net_income = net_income.iloc[-1]

    fcf_margin = (last_fcf / last_revenue) * 100 if last_revenue != 0 else None
    conversion_rate = (last_fcf / last_net_income) * 100 if last_net_income != 0 else None

    # Optional Debug (set to False to hide)
    if True:
        st.write("🔧 Debug Info", {
            "FCF Years": fcf.index.tolist(),
            "Revenue Years": revenue.index.tolist(),
            "Net Income Years": net_income.index.tolist(),
            "Last FCF": last_fcf,
            "Last Revenue": last_revenue,
            "Last Net Income": last_net_income
        })

    # Display
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Last FCF", f"${last_fcf/1e9:.1f}B")
    col2.metric("📈 3-Year Avg", f"${fcf.tail(3).mean()/1e9:.1f}B")
    col3.metric("📎 FCF Margin", f"{fcf_margin:.0f}%" if fcf_margin is not None else "N/A")
    col4.metric("🔄 Conversion Rate", f"{conversion_rate:.0f}%" if conversion_rate is not None else "N/A")

    # Chart
    chart_data = pd.DataFrame({
        "Year": fcf.index,
        "Free Cash Flow": fcf.values / 1e9
    })

    chart = alt.Chart(chart_data).mark_bar(color="#7f8c8d").encode(
        x=alt.X("Year:O"),
        y=alt.Y("Free Cash Flow:Q")
    ).properties(
        width=700,
        height=400,
        title="🧾 Free Cash Flow Over Time"
    )

    st.altair_chart(chart, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ FCF block failed: {e}")



st.write("Income statement rows:", income_stmt.index.tolist())
st.write("Income statement columns:", income_stmt.columns.tolist())

latest_period = income_stmt.columns[0]


import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.subheader("Earnings Breakdown Waterfall")

try:
    # Use the already available income_stmt
    latest_col = income_stmt.columns[0]  # Most recent quarter

    def get_amount(row_name):
        try:
            return float(income_stmt.loc[row_name][latest_col])
        except:
            return 0.0

    line_items = {
        "Revenue": "Total Revenue",
        "Cost of Revenue": "Cost Of Revenue",
        "Gross Profit": "Gross Profit",
        "Operating Expenses": "Operating Expense",
        "Operating Income": "Operating Income",
        "Other Expenses": "Other Non Operating Income Expenses",
        "Net Income": "Net Income"
    }

    amounts = {label: get_amount(row) for label, row in line_items.items()}
    waterfall_data = []
    cumulative = 0
    for i, (label, value) in enumerate(amounts.items()):
        if i == 0:  # Revenue
            waterfall_data.append(dict(name=label, y=value, measure="relative"))
            cumulative += value
        elif label == "Net Income":
            final = cumulative + value
            waterfall_data.append(dict(name=label, y=final, measure="total"))
        else:
            waterfall_data.append(dict(name=label, y=value, measure="relative"))
            cumulative += value

    # Create subplot with chart on right and table on left
    fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6],
                        specs=[[{"type": "table"}, {"type": "waterfall"}]])

    # Table of values
    fig.add_trace(
        go.Table(
            header=dict(values=["Line Item", "Amount (USD)"]),
            cells=dict(values=[list(amounts.keys()), [f"${v:,.0f}" for v in amounts.values()]])
        ),
        row=1, col=1
    )

    # Waterfall chart
    fig.add_trace(
        go.Waterfall(
            orientation="v",
            measure=[d["measure"] for d in waterfall_data],
            x=list(line_items.keys()),
            y=[d["y"] for d in waterfall_data],
            connector={"line": {"color": "gray"}},
        ),
        row=1, col=2
    )

    fig.update_layout(title="Earnings Waterfall Chart", height=600)
    st.plotly_chart(fig, use_container_width=True, key="earnings_waterfall_chart")

except Exception as e:
    st.warning(f"⚠️ Earnings Waterfall Chart failed: {e}")




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

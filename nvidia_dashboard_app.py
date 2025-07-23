import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import plotly.express as px
from streamlit.runtime.caching import cache_data
import altair as alt
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from finvizfinance.quote import finvizfinance
import nltk
import openai
from openai import RateLimitError, APIError


# Set page config BEFORE anything else
st.set_page_config(page_title="📈 Stock Dashboard", layout="wide")

import streamlit.components.v1 as components

import streamlit as st

# — Custom CSS for hiding Streamlit chrome and branding —
st.markdown(
    """
    <style>
      /* Hide the default Streamlit header menu and footer */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}

      /* Brand header container */
      .app-header {
        display: flex;
        align-items: center;
        padding: 10px 0;
      }
      /* Logo spacing (if you ever revert to HTML img) */
      .app-header img {
        height: 50px;
        margin-right: 15px;
      }
      /* Header text styling */
      .app-header h1 {
        color: #C8102E;  /* LC Red */
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
      }
      /* Section titles */
      .section-title {
        color: #C8102E !important;
        font-size: 1.75rem !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# — Logo + Branded Header —
# Make sure 'assets/logo.png' exists and is committed
st.image("assets/logo.png", width=60)
st.markdown(
    """
    <div class="app-header">
      <h1>Lighthouse Canton Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)


def render_ticker_widget():
    ticker_html = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="margin-bottom: 20px;">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
      {
        "symbols": [
          {
            "proName": "FOREXCOM:SPXUSD",
            "title": "S&P 500 Index"
          },
          {
            "proName": "FOREXCOM:NSXUSD",
            "title": "US 100 Cash CFD"
          },
          {
            "proName": "FX_IDC:EURUSD",
            "title": "EUR to USD"
          },
          {
            "proName": "BITSTAMP:BTCUSD",
            "title": "Bitcoin"
          },
          {
            "proName": "BITSTAMP:ETHUSD",
            "title": "Ethereum"
          }
        ],
        "colorTheme": "dark",
        "locale": "en",
        "largeChartUrl": "",
        "isTransparent": false,
        "showSymbolLogo": true,
        "displayMode": "adaptive"
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    components.html(ticker_html, height=100)

# Place it right after page config and title
render_ticker_widget()

# Clear cache (optional - usually used for dev only)
st.cache_data.clear()

# Download NLTK data (only once, should ideally be outside app or cached)
nltk.download("vader_lexicon", quiet=True)

# Load OpenAI API key from secrets
from openai import OpenAI
client = OpenAI(api_key=st.secrets["openai"]["key"])



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



@st.cache_data(ttl=60)
def get_company_name(ticker):
    try:
        return yf.Ticker(ticker).info.get("longName", ticker)
    except Exception:
        return ticker

@st.cache_data(ttl=60)
def fetch_news(company_name):
    today = datetime.today()
    last_week = today - timedelta(days=7)
    params = {
        "q": company_name,
        "from": last_week.strftime("%Y-%m-%d"),
        "sortBy": "relevancy",
        "language": "en",
        "apiKey": NEWSAPI_KEY,
        "pageSize": 10,  # limit to latest 10 articles
    }
    response = requests.get("https://newsapi.org/v2/everything", params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])
    return []

def categorize_news(articles):
    positive, negative, emerging = [], [], []
    positive_keywords = ["beats", "growth", "record", "surge", "upgrade", "strong", "gain", "boost"]
    negative_keywords = ["misses", "decline", "drop", "downgrade", "loss", "concern", "crisis", "lawsuit"]

    for article in articles:
        text = (article.get("title", "") + " " + article.get("description", "")).lower()
        if any(word in text for word in positive_keywords):
            positive.append(article)
        elif any(word in text for word in negative_keywords):
            negative.append(article)
        else:
            emerging.append(article)
    return positive, negative, emerging

def summarize_section(articles, max_lines=3):
    # Generate a simple summary by extracting key points (titles combined)
    if not articles:
        return "No significant news found."
    lines = [f"- {art['title']}" for art in articles[:max_lines]]
    return "\n".join(lines)

def news_summary_block(ticker):
    company_name = get_company_name(ticker)
    articles = fetch_news(company_name)

    st.markdown(f"### 🧠 News Summary for {company_name}")

    if not articles:
        st.warning("No recent news found.")
        return

    pos, neg, emerg = categorize_news(articles)

    st.markdown("#### 📈 Positive Developments")
    st.markdown(summarize_section(pos))

    st.markdown("#### ⚠️ Risks & Negative Sentiment")
    st.markdown(summarize_section(neg))

    st.markdown("#### 🧩 Emerging Themes")
    st.markdown(summarize_section(emerg))


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

# ================= ADD THESE TRADINGVIEW FUNCTIONS AFTER YOUR EXISTING FUNCTIONS =================

def tradingview_advanced_chart(ticker, height=700):
    """Enhanced TradingView Advanced Chart Widget with larger size"""
    html_code = f"""
    <div class="tradingview-widget-container" style="height:{height}px;width:100%">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js">
      {{
      "autosize": true,
      "symbol": "NASDAQ:{ticker.upper()}",
      "interval": "D",
      "timezone": "Etc/UTC",
      "theme": "light",
      "style": "1",
      "locale": "en",
      "withdateranges": true,
      "allow_symbol_change": true,
      "calendar": false,
      "support_host": "https://www.tradingview.com",
      "width": "100%",
      "height": "{height}"
      }}
      </script>
    </div>
    """
    return components.html(html_code, height=height)


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



# — Load Lighthouse Canton commentary —
@st.cache_data
def load_lighthouse_commentary():
    df = pd.read_excel("CIO Stocks.xlsx")
    df.columns = df.columns.str.strip().str.lower()
    return df

cio_df      = load_lighthouse_commentary()
ticker_upper = ticker.upper()
matched     = cio_df[cio_df["ticker"].str.upper().str.strip() == ticker_upper]

# 🧭 Lighthouse Canton View
st.markdown('<div class="section-title">🧭 Lighthouse Canton View</div>', unsafe_allow_html=True)

if not matched.empty:
    view         = matched["lighthouse canton view"].values[0]
    target_price = matched["target price"].values[0]

    # Fetch current price
    try:
        current_price = yf.Ticker(ticker).info.get("regularMarketPrice", None)
    except Exception:
        current_price = None

    # Compute upside/downside
    upside_pct = None
    if pd.notna(target_price) and current_price:
        upside_pct = ((target_price - current_price) / current_price) * 100
        direction  = "Upside" if upside_pct >= 0 else "Downside"
        color      = "green" if upside_pct >= 0 else "red"

    st.markdown(f"**📌 View:** {view}")
    st.markdown(f"**🎯 Target Price:** ${target_price:,.2f}")
    if current_price is not None:
        st.markdown(f"**💵 Current Price:** ${current_price:,.2f}")
        st.markdown(f"**📈 {direction} Potential:** :{color}[{upside_pct:.2f}%]")
    else:
        st.markdown("⚠️ Could not fetch current price.")
else:
    st.info("This stock is not present in the Lighthouse Canton coverage.")

# 📈 Investment Thesis & Upside Potential
st.markdown('<div class="section-title">📈 Investment Thesis & Upside Potential</div>', unsafe_allow_html=True)

thesis_points = []
if revenue and revenue > 1e9:
    thesis_points.append(f"Strong topline performance with trailing revenue of ${revenue/1e9:.2f}B suggests robust demand.")
if eps and eps > 0:
    thesis_points.append(f"Solid EPS of ${eps:.2f} indicates strong earnings power.")
if roe and roe > 0.15:
    thesis_points.append(f"Healthy Return on Equity (ROE) of {roe*100:.1f}% highlights efficient capital allocation.")
if earnings_growth and earnings_growth > 0:
    thesis_points.append(f"Positive quarterly earnings growth of {earnings_growth*100:.1f}% supports upside potential.")
if market_cap and market_cap > 50e9:
    thesis_points.append(f"Large‑cap stability: {long_name} operates at a market cap above $50B, enhancing institutional confidence.")
if len(thesis_points) < 3:
    thesis_points.append("Limited available financial highlights — further analysis recommended.")

for point in thesis_points:
    st.markdown(f"- {point}")

# ⚠️ Risks & Concerns
st.markdown('<div class="section-title">⚠️ Risks & Concerns</div>', unsafe_allow_html=True)

risk_points = []
if earnings_growth is not None and earnings_growth < 0:
    risk_points.append(f"Negative earnings growth ({earnings_growth*100:.1f}%) may signal performance headwinds.")
if profit_margin is not None and profit_margin < 0.05:
    risk_points.append(f"Thin profit margins ({profit_margin*100:.1f}%) could limit scalability.")
if roe is not None and roe < 0.05:
    risk_points.append(f"Weak Return on Equity ({roe*100:.1f}%) may suggest inefficient operations.")
if debt_to_equity is not None and debt_to_equity > 100:
    risk_points.append(f"Elevated debt‑to‑equity ratio ({debt_to_equity:.0f}%) increases financial risk.")
if current_ratio is not None and current_ratio < 1:
    risk_points.append(f"Current ratio below 1.0 raises concerns over short‑term liquidity.")
if len(risk_points) < 3:
    risk_points.append("No major red flags from available metrics — monitor quarterly updates.")

for point in risk_points:
    st.markdown(f"- {point}")






import streamlit as st
import streamlit.components.v1 as components

# — TradingView widget definitions (unchanged) —

def tradingview_financials_overview(ticker, height=500):
    """TradingView Financials Overview Widget"""
    html_code = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-financials.js">
      {{
        "colorTheme": "light",
        "isTransparent": false,
        "largeChartUrl": "",
        "displayMode": "regular",
        "width": "100%",
        "height": "{height}",
        "symbol": "NASDAQ:{ticker.upper()}",
        "locale": "en"
      }}
      </script>
    </div>
    """
    return components.html(html_code, height=height)

def tradingview_fundamental_data(ticker, height=500):
    """TradingView Fundamental Data Widget for Valuations"""
    html_code = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-fundamental-data.js">
      {{
        "colorTheme": "light",
        "isTransparent": false,
        "largeChartUrl": "",
        "displayMode": "regular",
        "width": "100%",
        "height": "{height}",
        "symbol": "NASDAQ:{ticker.upper()}",
        "locale": "en"
      }}
      </script>
    </div>
    """
    return components.html(html_code, height=height)

def tradingview_earnings_estimates(ticker, height=500):
    """TradingView Earnings Estimates Widget"""
    html_code = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-earnings.js">
      {{
        "colorTheme": "light",
        "isTransparent": false,
        "largeChartUrl": "",
        "displayMode": "regular",
        "width": "100%",
        "height": "{height}",
        "symbol": "NASDAQ:{ticker.upper()}",
        "locale": "en"
      }}
      </script>
    </div>
    """
    return components.html(html_code, height=height)

def tradingview_dividends(ticker, height=400):
    """TradingView Dividends Widget"""
    html_code = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js">
      {{
        "symbol": "NASDAQ:{ticker.upper()}",
        "width": "100%",
        "locale": "en",
        "colorTheme": "light",
        "isTransparent": false,
        "height": "{height}"
      }}
      </script>
    </div>
    """
    return components.html(html_code, height=height)

def tradingview_analyst_estimates(ticker, height=500):
    """TradingView Analyst Estimates Widget"""
    html_code = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-analyst-estimates.js">
      {{
        "symbol": "NASDAQ:{ticker.upper()}",
        "width": "100%",
        "height": "{height}",
        "colorTheme": "light",
        "isTransparent": false,
        "locale": "en"
      }}
      </script>
    </div>
    """
    return components.html(html_code, height=height)

def tradingview_technical_analysis_widget(ticker, height=500):
    """TradingView Technical Analysis Widget"""
    html_code = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js">
      {{
        "interval": "1D",
        "width": "100%",
        "isTransparent": false,
        "height": "{height}",
        "symbol": "NASDAQ:{ticker.upper()}",
        "showIntervalTabs": true,
        "displayMode": "regular",
        "colorTheme": "light",
        "locale": "en"
      }}
      </script>
    </div>
    """
    return components.html(html_code, height=height)


# — Main dashboard layout —

def equities_dashboard(ticker):
    # … your previous sections (header, financials widget, DCF, etc.) …

    # 🧾 Financials Overview
    st.markdown('<div class="section-title">🧾 Financials Overview</div>', unsafe_allow_html=True)
    tradingview_financials_overview(ticker, height=500)

    # 📊 Fundamental Data
    st.markdown('<div class="section-title">📊 Fundamental Data</div>', unsafe_allow_html=True)
    tradingview_fundamental_data(ticker, height=500)

    # 📈 Earnings Estimates
    st.markdown('<div class="section-title">📈 Earnings Estimates</div>', unsafe_allow_html=True)
    tradingview_earnings_estimates(ticker, height=500)

    # 💰 Dividends
    st.markdown('<div class="section-title">💰 Dividends</div>', unsafe_allow_html=True)
    tradingview_dividends(ticker, height=400)

    # 🎯 Analyst Estimates
    st.markdown('<div class="section-title">🎯 Analyst Estimates</div>', unsafe_allow_html=True)
    tradingview_analyst_estimates(ticker, height=500)

    # 🛠 Technical Analysis
    st.markdown('<div class="section-title">🛠 Technical Analysis</div>', unsafe_allow_html=True)
    tradingview_technical_analysis_widget(ticker, height=500)

    # … any follow‑on sections …

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import plotly.graph_objects as go

# -----------------------
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
from finvizfinance.quote import finvizfinance

# ========== TRADINGVIEW PRICE CHARTS ==========

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI, RateLimitError, APIError

# — TradingView Price Charts Functions — 
def tv_advanced_chart(ticker, height=500):
    html = f"""
    <div style="height:{height}px;width:100%">
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js">
      {{
        "autosize": true,
        "symbol": "NASDAQ:{ticker.upper()}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "withdateranges": true,
        "allow_symbol_change": false,
        "calendar": false
      }}
      </script>
    </div>
    """
    components.html(html, height=height)

def tv_mini_chart(ticker, height=300):
    html = f"""
    <div style="height:{height}px;width:100%">
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js">
      {{
        "symbol": "NASDAQ:{ticker.upper()}",
        "width": "100%",
        "height": "{height}",
        "locale": "en",
        "dateRange": "12M",
        "colorTheme": "light",
        "isTransparent": false,
        "autosize": false
      }}
      </script>
    </div>
    """
    components.html(html, height=height)

def render_price_charts(ticker):
    st.markdown(
        '<div class="section-title">📈 Price Chart (TradingView)</div>',
        unsafe_allow_html=True
    )
    tv_advanced_chart(ticker)
    tv_mini_chart(ticker)

# — Main Equities Dashboard — 
def equities_dashboard(ticker):
    st.markdown(
        f"<h3 style='font-size:24px'>{ticker.upper()} Financial Dashboard</h3>",
        unsafe_allow_html=True
    )

    # 🧾 Financials Overview
    st.markdown(
        '<div class="section-title">🧾 Financials Overview (TradingView)</div>',
        unsafe_allow_html=True
    )
    components.html(f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-financials.js" async>
      {{
        "symbol": "NASDAQ:{ticker.upper()}",
        "colorTheme": "dark",
        "displayMode": "regular",
        "isTransparent": false,
        "locale": "en",
        "width": "100%",
        "height": 550
      }}
      </script>
    </div>
    """, height=600)

    # 📈 Price Charts
    render_price_charts(ticker)

# — Entry Point — 
ticker = st.text_input("Enter Ticker Symbol", value="AAPL", key="ticker_input")
if ticker:
    equities_dashboard(ticker)
else:
    st.info("Please enter a stock ticker symbol.")

from openai import OpenAI, RateLimitError, APIError
import streamlit as st

# — Initialize OpenAI client once, using your stored secret key —
client = OpenAI(api_key=st.secrets["openai"]["key"])

def gpt_dcf_prompt(ticker, rev_growth, ebit_margin, tax_rate, capex, wacc, term_growth):
    prompt = f"""
You are a financial modeling expert who performs equity valuation using only the Discounted Cash Flow (DCF) method.

- Ticker: {ticker}
- Revenue growth for next 10 years: {rev_growth}
- EBIT margin for next 10 years: {ebit_margin}
- Tax rate: {tax_rate}%
- CapEx as % of revenue: {capex}
- WACC: {wacc}%
- Terminal growth rate: {term_growth}%

Build a DCF model (FCFF), use a 10‑year projection, Gordon Growth for terminal value,
subtract net debt, divide by shares outstanding to get per‑share value.
Present:
• Projection table (Revenue, EBIT, NOPAT, FCFF each year)
• Summary DCF output
• Sensitivity table for WACC × terminal growth

Respond professionally.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a valuation expert."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        st.subheader("📊 DCF Valuation Summary")
        st.markdown(resp.choices[0].message.content)

    except RateLimitError:
        st.error("⚠️ OpenAI API rate limit reached. Please wait and try again.")
    except APIError as e:
        st.error(f"❌ OpenAI API error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error during DCF generation: {e}")

def dcf_valuation_module():
    st.markdown(
        '<div class="section-title">💸 Discounted Cash Flow (DCF) Valuation</div>',
        unsafe_allow_html=True
    )
    st.markdown("Fill in your assumptions below. We'll run the DCF model automatically:")

    # Collect assumptions
    dcf_ticker = st.text_input("Company Ticker (e.g., AAPL)", value="AAPL", key="dcf_ticker")

    rev_growth = st.text_area(
        "10‑Year Revenue Growth Forecast (%)",
        value="5,5,4.5,4,3.5,3,2.5,2,2,2",
        help="Comma‑separated annual % growth rates"
    )

    ebit_margin = st.text_area(
        "10‑Year EBIT Margin Forecast (%)",
        value="25,25,24.5,24,23.5,23,22.5,22,22,22",
        help="Comma‑separated EBIT margin % for each of the next 10 years"
    )

    tax_rate = st.number_input(
        "Effective Tax Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=15.0,
        step=0.1
    )

    capex = st.text_area(
        "CapEx Forecast (% of Revenue)",
        value="3,3,3,2.5,2.5,2.5,2,2,2,2",
        help="Comma‑separated CapEx as % of revenue for each of 10 years"
    )

    wacc = st.number_input(
        "Discount Rate (WACC) (%)",
        min_value=0.0,
        value=8.0,
        step=0.1
    )

    term_growth = st.number_input(
        "Terminal Growth Rate (%)",
        min_value=0.0,
        value=2.5,
        step=0.1
    )

    if st.button("Run DCF"):
        with st.spinner("Running DCF valuation via GPT…"):
            gpt_dcf_prompt(
                dcf_ticker,
                rev_growth,
                ebit_margin,
                tax_rate,
                capex,
                wacc,
                term_growth
            )

# — Call the DCF module inline in your app —
dcf_valuation_module()



import yfinance as yf
import pandas as pd




# ------------------- DCF Valuation (Forecast Table) -------------------
with st.container():
    st.markdown("### 📊 Discounted Cash Flow (DCF) Valuation Model")

    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        financials = yf_ticker.financials.fillna(0)
        cashflow = yf_ticker.cashflow.fillna(0)
        income_stmt = yf_ticker.income_stmt.fillna(0)

        # Normalize index names for consistency
        cashflow.index = cashflow.index.str.lower()
        financials.index = financials.index.str.lower()
        income_stmt.index = income_stmt.index.str.lower()

        # Extract trailing revenue and net income
        revenues = financials.loc["total revenue"] if "total revenue" in financials.index else None
        net_incomes = income_stmt.loc["net income"] if "net income" in income_stmt.index else None
        fcf_row = next((r for r in cashflow.index if "free cash flow" in r), None)

        if revenues is None or net_incomes is None:
            st.warning("Missing revenue or net income data.")
        else:
            # Use most recent year as base
            base_revenue = revenues.iloc[0]
            base_net_income = net_incomes.iloc[0]
            net_margin = base_net_income / base_revenue if base_revenue != 0 else 0

            if fcf_row:
                base_fcfe = cashflow.loc[fcf_row].iloc[0]
            else:
                op_row = next((r for r in cashflow.index if "operating cash flow" in r), None)
                capex_row = next((r for r in cashflow.index if "capital expenditure" in r), None)
                debt_row = next((r for r in cashflow.index if "repayment of debt" in r or "net debt repayment" in r), None)
                if op_row and capex_row:
                    base_fcfe = cashflow.loc[op_row].iloc[0] - cashflow.loc[capex_row].iloc[0]
                else:
                    st.warning("Unable to calculate FCFE from available cash flow data.")
                    base_fcfe = 0

            # --- Inputs
            with st.expander("⚙️ Adjust Forecast Assumptions", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    forecast_years = st.slider("Forecast Years", 3, 10, 5)
                with col2:
                    revenue_growth = st.slider("Annual Revenue Growth (%)", 0.0, 20.0, 8.0)
                with col3:
                    discount_rate = st.slider("Discount Rate (%)", 5.0, 15.0, 10.0)
                terminal_growth = st.slider("Terminal Growth Rate (%)", 0.0, 5.0, 2.0)

            # --- Forecast
            forecast = []
            revenue = base_revenue
            for year in range(1, forecast_years + 1):
                revenue *= (1 + revenue_growth / 100)
                net_income = revenue * net_margin
                fcfe = base_fcfe * (1 + revenue_growth / 100) ** year  # grow FCFE with revenue
                pv_fcfe = fcfe / ((1 + discount_rate / 100) ** year)

                forecast.append({
                    "Year": f"Year {year}",
                    "Revenue": revenue,
                    "Revenue Growth %": revenue_growth,
                    "Net Margin %": net_margin * 100,
                    "Net Income": net_income,
                    "FCFE": fcfe,
                    "Discount Rate %": discount_rate,
                    "Present Value": pv_fcfe
                })

            # --- Terminal Value
            terminal_fcfe = forecast[-1]["FCFE"] * (1 + terminal_growth / 100)
            terminal_value = terminal_fcfe / (discount_rate / 100 - terminal_growth / 100)
            discounted_terminal = terminal_value / ((1 + discount_rate / 100) ** forecast_years)

            forecast.append({
                "Year": "Terminal",
                "Revenue": np.nan,
                "Revenue Growth %": terminal_growth,
                "Net Margin %": np.nan,
                "Net Income": np.nan,
                "FCFE": terminal_fcfe,
                "Discount Rate %": discount_rate,
                "Present Value": discounted_terminal
            })

            df_forecast = pd.DataFrame(forecast)
            df_forecast.set_index("Year", inplace=True)
            df_forecast_display = df_forecast.T.style.format("{:,.0f}", subset=(slice(None), slice(None)))
            st.dataframe(df_forecast_display, use_container_width=True)

            intrinsic_value = df_forecast["Present Value"].sum()
            shares_out = info.get("sharesOutstanding", None)
            if shares_out:
                intrinsic_per_share = intrinsic_value / shares_out
                current_price = info.get("currentPrice", 0)
                pct_diff = (intrinsic_per_share - current_price) / current_price * 100 if current_price else 0
                st.metric("📈 Intrinsic Value per Share", f"${intrinsic_per_share:,.2f}", f"{pct_diff:+.2f}%")
            else:
                st.write(f"💰 Total Present Value of FCFE: ${intrinsic_value:,.0f}")

    except Exception as e:
        st.error(f"Error in DCF block: {e}")




import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Assuming sp500_df is your DataFrame of S&P 500 stocks with columns:
# 'symbol', 'industry', 'sector', 'marketcap'

def find_suggested_peers(ticker, sp500_df):
    """Find peers based on industry and market cap proximity."""
    ticker = ticker.upper()
    current_row = sp500_df[sp500_df["symbol"].str.upper() == ticker]
    if current_row.empty:
        return []
    selected_industry = current_row["industry"].values[0]
    selected_sector = current_row["sector"].values[0]
    selected_marketcap = current_row["marketcap"].values[0]

    lower_bound = selected_marketcap * 0.4
    upper_bound = selected_marketcap * 2.5

    peer_df = sp500_df[
        (sp500_df["symbol"].str.upper() != ticker) &
        (sp500_df["industry"] == selected_industry) &
        (sp500_df["marketcap"] >= lower_bound) &
        (sp500_df["marketcap"] <= upper_bound)
    ]

    if peer_df.empty:
        # fallback sector-based
        peer_df = sp500_df[
            (sp500_df["symbol"].str.upper() != ticker) &
            (sp500_df["sector"] == selected_sector) &
            (sp500_df["marketcap"] >= lower_bound) &
            (sp500_df["marketcap"] <= upper_bound)
        ]

    return peer_df["symbol"].str.upper().tolist()

def fetch_peer_data(tickers):
    """Fetch valuation, margin, and performance data for tickers."""
    data = []
    for sym in tickers:
        try:
            stock = yf.Ticker(sym)
            info = stock.info
            fast_info = stock.fast_info
            
            # Basic valuation metrics
            pe_ratio = info.get("trailingPE", None)
            pb_ratio = info.get("priceToBook", None)
            peg_ratio = info.get("pegRatio", None)
            earnings_yield = (1 / pe_ratio) if pe_ratio and pe_ratio > 0 else None
            ev_to_ebitda = info.get("enterpriseToEbitda", None)
            ev_to_ebit = info.get("enterpriseToEbit", None)

            # Margin: Use operatingMargins if available (operating margin)
            operating_margin = info.get("operatingMargins", None)  # decimal (e.g. 0.25)

            # Price Performance 1-year % change (approximate)
            hist = stock.history(period="1y")
            if hist.empty:
                perf_1y = None
            else:
                perf_1y = (hist["Close"][-1] / hist["Close"][0] - 1) * 100  # %

            data.append({
                "Symbol": sym,
                "Name": info.get("longName", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "Market Cap": info.get("marketCap", None),
                "Current Price": fast_info.get("lastPrice", None),
                "P/E": pe_ratio,
                "P/B": pb_ratio,
                "PEG": peg_ratio,
                "Earnings Yield": earnings_yield,
                "EV/EBITDA": ev_to_ebitda,
                "EV/EBIT": ev_to_ebit,
                "Operating Margin": operating_margin,
                "1Y Price Perf (%)": perf_1y,
            })
        except Exception as e:
            st.warning(f"Could not fetch data for {sym}: {e}")

    return pd.DataFrame(data)

def highlight_outliers(df, target_ticker):
    """
    Highlight where the target ticker's metrics notably differ from peers.
    For example, if margins are compressing in peers but stable in target,
    or vice versa.
    """
    df_numeric = df.select_dtypes(include=[np.number])
    target_row = df[df["Symbol"] == target_ticker]

    if target_row.empty:
        return df  # no target found, return as is

    target_vals = target_row.iloc[0]

    def margin_outlier(row):
        if pd.isna(row["Operating Margin"]) or pd.isna(target_vals["Operating Margin"]):
            return ""
        diff = row["Operating Margin"] - target_vals["Operating Margin"]
        # Example heuristic: margin difference > 5% (0.05) is notable
        if diff <= -0.05:
            return "🔻 Margin Compressing"
        elif diff >= 0.05:
            return "🔺 Margin Stronger"
        else:
            return ""

    df["Margin Signal"] = df.apply(margin_outlier, axis=1)

    return df

# --------- STREAMLIT UI -------------

st.header("📈 Peer Signal Tracker: Monitor Competitor Performance & Signals")

ticker_input = st.text_input("Enter Target Stock Ticker:", value="NVDA").upper()

if ticker_input:
    # 1. Find suggested peers
    suggested_peers = find_suggested_peers(ticker_input, sp500_df)

    # 2. Full universe tickers for search in multiselect
    all_tickers = sp500_df["symbol"].str.upper().sort_values().unique().tolist()

    # 3. Multiselect with suggested peers pre-selected, but full universe as options
    selected_peers = st.multiselect(
        "Select or search peers to compare:",
        options=all_tickers,
        default=suggested_peers,
        help="Search or add peers beyond suggested ones."
    )

    if not selected_peers:
        st.warning("Please select at least one peer to compare.")
    else:
        # Include target ticker in dataset for comparison
        tickers_to_fetch = list(set(selected_peers + [ticker_input]))

        # 4. Fetch data for target + peers
        df = fetch_peer_data(tickers_to_fetch)

        if df.empty:
            st.error("No data available for the selected tickers.")
        else:
            # 5. Highlight outliers / signals in margin trends
            df = highlight_outliers(df, ticker_input)

            # 6. Format columns nicely, sort by Market Cap descending
            df_display = df.copy()
            df_display["Market Cap"] = df_display["Market Cap"].apply(lambda x: f"${x/1e9:.2f}B" if pd.notna(x) else "N/A")
            df_display["Current Price"] = df_display["Current Price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
            df_display["P/E"] = df_display["P/E"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            df_display["P/B"] = df_display["P/B"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            df_display["PEG"] = df_display["PEG"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            df_display["Earnings Yield"] = df_display["Earnings Yield"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            df_display["EV/EBITDA"] = df_display["EV/EBITDA"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            df_display["EV/EBIT"] = df_display["EV/EBIT"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            df_display["Operating Margin"] = df_display["Operating Margin"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            df_display["1Y Price Perf (%)"] = df_display["1Y Price Perf (%)"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

            df_display = df_display.sort_values(by="Market Cap", ascending=False)

            # 7. Display table
            st.dataframe(df_display.reset_index(drop=True))

            # 8. Summary insights for target vs peers
            st.markdown("---")
            st.subheader(f"Insights on {ticker_input} vs Selected Peers")

            # Margin signals summary
            margin_signals = df[df["Margin Signal"] != ""]
            if not margin_signals.empty:
                for _, row in margin_signals.iterrows():
                    if row["Symbol"] == ticker_input:
                        st.success(f"✅ {ticker_input}: {row['Margin Signal']}")
                    else:
                        st.warning(f"{row['Symbol']}: {row['Margin Signal']}")
            else:
                st.info("No notable margin divergence signals detected among selected peers.")

else:
    st.info("Enter a ticker above to see peer signal tracking.")





import streamlit as st
import yfinance as yf
import pandas as pd

with st.container():
    st.markdown("## 📈 Profitability Overview")

    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        income_stmt = yf_ticker.income_stmt.fillna(0)
        balance_sheet = yf_ticker.balance_sheet.fillna(0)

        income_stmt.index = income_stmt.index.str.lower()
        balance_sheet.index = balance_sheet.index.str.lower()

        # --- Margins ---
        st.markdown("### 🧮 Margin Profile")
        col1, col2, col3 = st.columns(3)

        revenue = income_stmt.loc["total revenue"].iloc[0] if "total revenue" in income_stmt.index else None
        gross_profit = income_stmt.loc["gross profit"].iloc[0] if "gross profit" in income_stmt.index else None
        operating_income = income_stmt.loc["operating income"].iloc[0] if "operating income" in income_stmt.index else None
        net_income = income_stmt.loc["net income"].iloc[0] if "net income" in income_stmt.index else None

        if revenue:
            gross_margin = gross_profit / revenue if gross_profit else None
            op_margin = operating_income / revenue if operating_income else None
            net_margin = net_income / revenue if net_income else None

            for col, label, value in zip(
                [col1, col2, col3],
                ["Gross Margin", "Operating Margin", "Net Margin"],
                [gross_margin, op_margin, net_margin]
            ):
                with col:
                    if value is not None:
                        pct = value * 100
                        st.metric(label, f"{pct:.2f}%")
                        st.progress(min(max(value, 0), 1.0))  # Clamp 0-1
                    else:
                        st.metric(label, "N/A")
                        st.progress(0.001)  # Tiny visible bar

            # Historical margins line chart
            margin_df = pd.DataFrame({
                "Gross Margin": income_stmt.loc["gross profit"] / income_stmt.loc["total revenue"],
                "Operating Margin": income_stmt.loc["operating income"] / income_stmt.loc["total revenue"],
                "Net Margin": income_stmt.loc["net income"] / income_stmt.loc["total revenue"],
            }).T

            margin_df.columns = margin_df.columns.astype(str)  # Convert to string for plotting
            margin_df = margin_df.dropna(axis=1, how='all')

            if not margin_df.empty:
                st.line_chart(margin_df.T * 100)

        # --- Return on Capital Measures ---
        st.markdown("### 🧾 Return on Capital Measures")
        col1, col2, col3, col4 = st.columns(4)

        # Total assets
        total_assets = balance_sheet.loc["total assets"].iloc[0] if "total assets" in balance_sheet.index else None

        # Try multiple fallback keys for equity
        equity_keys = [
            "total stockholder equity",
            "stockholders equity",
            "common stock equity",
            "total equity gross minority interest"
        ]
        total_equity = None
        for key in equity_keys:
            if key in balance_sheet.index:
                total_equity = balance_sheet.loc[key].iloc[0]
                break

        # Debt
        short_long_debt = balance_sheet.loc["short long term debt"].iloc[0] if "short long term debt" in balance_sheet.index else 0
        long_term_debt = balance_sheet.loc["long term debt"].iloc[0] if "long term debt" in balance_sheet.index else 0

        # Income
        oper_inc = income_stmt.loc["operating income"].iloc[0] if "operating income" in income_stmt.index else None
        ebit = income_stmt.loc["ebit"].iloc[0] if "ebit" in income_stmt.index else oper_inc
        net_income = income_stmt.loc["net income"].iloc[0] if "net income" in income_stmt.index else None

        # Denominator fallback logic
        ta = total_assets or 0
        te = total_equity or 0
        td = short_long_debt or 0
        ld = long_term_debt or 0
        ei = ebit if ebit is not None else 0
        ni = net_income if net_income is not None else 0

        # Compute metrics
        roe = (ni / te) if (te > 0 and net_income is not None) else None
        roa = (ni / ta) if (ta > 0 and net_income is not None) else None
        roic = (ei / (te + td)) if ((te + td) > 0 and ebit is not None) else None
        roce = (ei / (te + ld)) if ((te + ld) > 0 and ebit is not None) else None

        for col, label, value in zip(
            [col1, col2, col3, col4],
            ["ROE", "ROA", "ROIC", "ROCE"],
            [roe, roa, roic, roce]
        ):
            with col:
                if value is not None:
                    pct = value * 100
                    st.metric(label, f"{pct:.2f}%")
                    st.progress(min(max(value, 0), 1.0))
                else:
                    st.metric(label, "N/A")
                    st.progress(0.001)

        # Historical ROIC, ROCE, ROE line charts
        roic_df = pd.DataFrame()
        roce_df = pd.DataFrame()
        roe_df = pd.DataFrame()

        # Prepare historical data for ROIC, ROCE, ROE across income statement and balance sheet years
        years = income_stmt.columns.astype(str)

        roe_list = []
        roic_list = []
        roce_list = []

        for year in years:
            try:
                ni_yr = income_stmt.loc["net income"][year]
                te_yr = None
                for key in equity_keys:
                    if key in balance_sheet.index:
                        te_yr = balance_sheet.loc[key][year]
                        break
                ta_yr = balance_sheet.loc["total assets"][year] if "total assets" in balance_sheet.index else None
                td_yr = balance_sheet.loc["short long term debt"][year] if "short long term debt" in balance_sheet.index else 0
                ld_yr = balance_sheet.loc["long term debt"][year] if "long term debt" in balance_sheet.index else 0
                ebit_yr = income_stmt.loc["ebit"][year] if "ebit" in income_stmt.index else income_stmt.loc["operating income"][year]

                roe_val = (ni_yr / te_yr) if (te_yr and te_yr > 0 and ni_yr is not None) else None
                roic_val = (ebit_yr / (te_yr + td_yr)) if (te_yr and (te_yr + td_yr) > 0 and ebit_yr is not None) else None
                roce_val = (ebit_yr / (te_yr + ld_yr)) if (te_yr and (te_yr + ld_yr) > 0 and ebit_yr is not None) else None

                roe_list.append(roe_val)
                roic_list.append(roic_val)
                roce_list.append(roce_val)
            except Exception:
                roe_list.append(None)
                roic_list.append(None)
                roce_list.append(None)

        historical_df = pd.DataFrame({
            "ROE": roe_list,
            "ROIC": roic_list,
            "ROCE": roce_list
        }, index=years)

        historical_df = historical_df.dropna(how='all')

        if not historical_df.empty:
            st.line_chart(historical_df * 100)

    except Exception as e:
        st.error(f"Error loading profitability section: {e}")



import streamlit as st
import requests
from datetime import datetime, timedelta
import yfinance as yf

# --- NewsAPI key ---
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    st.warning("Missing NewsAPI key. Using cached/stubbed response.")

NEWS_API_URL = "https://newsapi.org/v2/everything"

# --- Get Company Name from Ticker (cached 1 hour) ---
@st.cache_data(ttl=3600)
def get_company_name_from_ticker(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName", ticker)
    except Exception:
        return ticker

# --- Fetch latest 10 news articles for the company, cached 60s ---
@st.cache_data(ttl=60)
def get_news_articles(company_name):
    today = datetime.today()
    last_week = today - timedelta(days=7)
    params = {
        "q": company_name,
        "from": last_week.strftime("%Y-%m-%d"),
        "sortBy": "relevancy",
        "language": "en",
        "apiKey": NEWSAPI_KEY,
        "pageSize": 10,  # limit articles to 10 for memory/performance
    }
    try:
        response = requests.get(NEWS_API_URL, params=params)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            st.error(f"News API error: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        return []

# --- Categorize articles into positive, negative, emerging ---
def categorize_articles(articles):
    positive, negative, emerging = [], [], []
    for article in articles:
        text = (article.get("title", "") + " " + article.get("description", "")).lower()
        if any(word in text for word in ["beats", "growth", "record", "surge", "upgrade", "strong", "gain", "boost"]):
            positive.append(article)
        elif any(word in text for word in ["misses", "decline", "drop", "downgrade", "loss", "concern", "crisis", "lawsuit"]):
            negative.append(article)
        else:
            emerging.append(article)
    return positive, negative, emerging

# --- Generate summary text (2-3 lines) for each category ---
def generate_summary(articles):
    if not articles:
        return "No significant news in this category."
    # Just combine first few headlines/descriptions for summary
    lines = []
    for art in articles[:3]:
        title = art.get("title", "")
        desc = art.get("description", "")
        snippet = title if len(title) > 20 else title + " - " + desc
        lines.append(snippet)
    return " ".join(lines[:3])  # keep max 3 sentences combined

# --- Main News Block ---
def news_block(ticker):
    company_name = get_company_name_from_ticker(ticker)

    st.markdown("### 🧠 Market News Summary")

    articles = get_news_articles(company_name)

    if not articles:
        st.warning(f"No recent news found for **{company_name}**.")
        return

    positive, negative, emerging = categorize_articles(articles)

    st.markdown("#### 📈 Positive Developments")
    st.write(generate_summary(positive))

    st.markdown("#### ⚠️ Risks & Negative Sentiment")
    st.write(generate_summary(negative))

    st.markdown("#### 🧩 Emerging Themes")
    st.write(generate_summary(emerging))


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




import pandas as pd
import altair as alt

# Example: op_cf and fcf are Series indexed by year as string or int
# Ensure indices are strings for consistency
op_cf.index = op_cf.index.astype(str)
fcf.index = fcf.index.astype(str)

# Create combined DataFrame with outer join on years (to include all years present)
df = pd.DataFrame({
    "Operating Cash Flow": op_cf,
    "Free Cash Flow": fcf
}).reset_index().rename(columns={"index": "Year"})

# Convert Year to string (or keep as is, just consistent)
df["Year"] = df["Year"].astype(str)

# Fill missing values with 0 (to plot bars for missing years as zero)
df.fillna(0, inplace=True)

# Melt DataFrame to long format for Altair
df_long = df.melt(id_vars=["Year"], var_name="Type", value_name="Amount")

# Build the bar chart
chart = alt.Chart(df_long).mark_bar().encode(
    x=alt.X('Year:O', title='Year'),
    y=alt.Y('Amount:Q', title='Amount'),
    color=alt.Color('Type:N', legend=alt.Legend(title="Cash Flow Type")),
    tooltip=['Year', 'Type', 'Amount']
).properties(
    title="Operating Cash Flow vs Free Cash Flow"
)

# Show chart only, no tables
st.altair_chart(chart, use_container_width=True)





latest_period = income_stmt.columns[0]
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

def earnings_waterfall_section(ticker: str):
    st.markdown("## 💰 Earnings Breakdown Waterfall")

    try:
        yf_ticker = yf.Ticker(ticker)
        income_stmt = yf_ticker.financials.fillna(0)

        # yfinance financials columns are dates — pick the most recent
        latest_period = income_stmt.columns[0]

        # Extract items (use common labels - some fallback)
        def get_value(possible_names):
            for name in possible_names:
                if name in income_stmt.index:
                    return income_stmt.loc[name, latest_period]
            return 0

        revenue = get_value(["Total Revenue", "Revenue"])
        cost_of_revenue = get_value(["Cost Of Revenue", "Cost of Revenue", "Cost of Goods Sold"])
        gross_profit = get_value(["Gross Profit"])
        operating_expenses = get_value(["Total Operating Expenses", "Operating Expenses"])
        operating_income = get_value(["Operating Income", "Operating Income or Loss"])
        other_expenses = get_value(["Other Income Expense Net", "Other Expenses"])
        net_income = get_value(["Net Income", "Net Income Applicable To Common Shares"])

        # Defensive calculation if some missing
        if gross_profit == 0 and revenue and cost_of_revenue:
            gross_profit = revenue - cost_of_revenue
        if operating_income == 0 and gross_profit and operating_expenses:
            operating_income = gross_profit - operating_expenses

        # Waterfall breakdown components with labels
        breakdown = {
            "Revenue": revenue,
            "Cost of Revenue": -cost_of_revenue,
            "Gross Profit": gross_profit,
            "Operating Expenses": -operating_expenses,
            "Operating Income": operating_income,
            "Other Expenses": -other_expenses,
            "Net Income": net_income,
        }

        # Build waterfall plot
        measure = []
        y_vals = []
        text = []
        for i, (label, val) in enumerate(breakdown.items()):
            if label == "Revenue":
                measure.append("absolute")
            elif label in ["Gross Profit", "Operating Income", "Net Income"]:
                measure.append("total")
            else:
                measure.append("relative")
            y_vals.append(val)
            text.append(f"{val:,.0f}")

        fig = go.Figure(go.Waterfall(
            name = "Earnings Breakdown",
            orientation = "v",
            measure = measure,
            x = list(breakdown.keys()),
            text = text,
            textposition = "outside",
            y = y_vals,
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
            increasing = {"marker":{"color":"#2ca02c"}},
            decreasing = {"marker":{"color":"#d62728"}},
            totals = {"marker":{"color":"#1f77b4"}},
        ))

        fig.update_layout(
            title=f"Earnings Breakdown Waterfall for {ticker.upper()} (Most Recent)",
            yaxis_title="USD",
            waterfallgroupgap = 0.5,
            autosize=False,
            width=700,
            height=450,
            margin=dict(l=40, r=40, t=80, b=40),
            font=dict(size=12)
        )

        # Display side-by-side with summary box
        col1, col2 = st.columns([3, 1])

        with col1:
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Earnings Summary")
            for label, val in breakdown.items():
                st.write(f"**{label}:** {val:,.0f}")

    except Exception as e:
        st.error(f"Error loading earnings breakdown: {e}")

# Example usage in your app:
ticker = st.text_input("Ticker symbol", value="NVDA")
if ticker:
    earnings_waterfall_section(ticker)


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

def get_balance_value(df, keys):
    for key in keys:
        if key in df.index:
            return df.loc[key].iloc[0]
    return None


import matplotlib.pyplot as plt

def solvency_overview_section(ticker: str):
    st.markdown("## 🏦 Solvency and Liquidity Overview")

with st.container():
    solvency_overview_section(ticker)    

    try:
        yf_ticker = yf.Ticker(ticker)
        balance_sheet = yf_ticker.balance_sheet.fillna(0)
        income_stmt = yf_ticker.income_stmt.fillna(0)

        balance_sheet.index = balance_sheet.index.str.lower()
        income_stmt.index = income_stmt.index.str.lower()

        def try_get(df, keys):
            for key in keys:
                if key in df.index:
                    return df.loc[key].iloc[0]
            return None

        total_assets = try_get(balance_sheet, ["total assets", "totalAssets"])
        total_equity = try_get(balance_sheet, [
            "total stockholder equity", "stockholders equity", 
            "common stock equity", "total equity gross minority interest"
        ])
        total_debt = sum([
            try_get(balance_sheet, ["short long term debt", "short/long term debt", "short term debt"]) or 0,
            try_get(balance_sheet, ["long term debt"]) or 0
        ])
        cash = try_get(balance_sheet, ["cash", "cash and cash equivalents"]) or 0
        current_assets = try_get(balance_sheet, ["total current assets", "current assets", "totalCurrentAssets"]) or 0
        current_liabilities = try_get(balance_sheet, ["total current liabilities", "current liabilities"]) or 0
        inventory = try_get(balance_sheet, ["inventory"]) or 0
        ebit = try_get(income_stmt, ["ebit", "operating income"]) or 0
        interest_expense = try_get(income_stmt, ["interest expense"]) or 0
        retained_earnings = try_get(balance_sheet, ["retained earnings"]) or 0
        working_capital = current_assets - current_liabilities if current_assets and current_liabilities else 0

        # --- Calculations ---
        net_debt = total_debt - cash
        net_debt_equity = (net_debt / total_equity) if total_equity else None
        debt_asset_ratio = (total_debt / total_assets) if total_assets else None
        interest_coverage = (ebit / abs(interest_expense)) if (ebit and interest_expense) else None
        cash_ratio = (cash / current_liabilities) if current_liabilities else None
        quick_ratio = ((current_assets - inventory) / current_liabilities) if current_liabilities else None
        current_ratio = (current_assets / current_liabilities) if current_liabilities else None

        # Altman Z-score (approximate)
        try:
            z_score = (
                1.2 * (working_capital / total_assets) +
                1.4 * (retained_earnings / total_assets) +
                3.3 * (ebit / total_assets) +
                0.6 * (total_equity / total_debt) +
                1.0 * (yf_ticker.info.get("totalRevenue", 0) / total_assets)
            ) if total_assets and total_equity and total_debt else None
        except:
            z_score = None

        # Display
        metrics = {
            "Net Debt/Equity": net_debt_equity,
            "Debt/Assets": debt_asset_ratio,
            "Interest Coverage": interest_coverage,
            "Cash Ratio": cash_ratio,
            "Quick Ratio": quick_ratio,
            "Current Ratio": current_ratio,
            "Altman Z-Score": z_score
        }

        cols = st.columns(4)
        for idx, (label, value) in enumerate(metrics.items()):
            with cols[idx % 4]:
                if value is not None and value != float("inf"):
                    st.metric(label, f"{value:.2f}")
                    st.progress(min(max(value / 10, 0.01), 1.0))
                else:
                    st.metric(label, "N/A")
                    st.progress(0.01)

    except Exception as e:
        st.error(f"Failed to load solvency overview: {e}")











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

import streamlit as st
import requests
from datetime import datetime

def earnings_call_summary_section(ticker: str):
    with st.container():
        st.header("📢 Latest Earnings Call Summary")
        st.markdown(f"✅ Debug – rendering earnings call summary for: `{ticker}`")

        try:
            api_key = st.secrets["FINNHUB_API_KEY"]
            url = f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker}&token={api_key}"
            r = requests.get(url, timeout=10)
            
            # Log metadata
            st.caption(f"Debug: Finnhub response status: {r.status_code}")
            content_type = r.headers.get("content-type", "")
            st.caption(f"Debug: Finnhub content-type: {content_type}")
            
            if not r.ok or "application/json" not in content_type:
                st.warning("❌ Expected JSON response but got something else.")
                return

            earnings_data = r.json()
            if not earnings_data:
                st.info("No earnings call data found from Finnhub.")
                return

            # Show the latest earnings summary
            latest = earnings_data[0]
            report_date = latest.get("period")
            actual_eps = latest.get("actual")
            estimate_eps = latest.get("estimate")
            surprise = latest.get("surprise")
            surprise_pct = latest.get("surprisePercent")

            st.markdown(f"🗓 **Earnings Report Date:** {report_date}")
            st.markdown(f"💰 **Actual EPS:** {actual_eps}")
            st.markdown(f"🔮 **Estimate EPS:** {estimate_eps}")
            st.markdown(f"🎯 **Surprise:** {surprise} ({surprise_pct}%)")

        except Exception as e:
            st.error(f"Section crashed: {e}")

if ticker:
    earnings_call_summary_section(ticker)






import streamlit as st
import yfinance as yf

def wall_street_price_targets(ticker: str):
    with st.container():
        st.subheader("💹 Wall Street Analyst Price Targets")

        stock = yf.Ticker(ticker)
        info = stock.info

        target_low = info.get("targetLowPrice")
        target_median = info.get("targetMedianPrice")
        target_mean = info.get("targetMeanPrice")
        target_high = info.get("targetHighPrice")
        num_analysts = info.get("numberOfAnalystOpinions")
        current_price = info.get("currentPrice")

        if any([target_low, target_median, target_mean, target_high]):
            cols = st.columns([1, 1, 1, 1, 1])
            cols[0].markdown("**Current Price**")
            cols[1].markdown("**Low Target**")
            cols[2].markdown("**Median Target**")
            cols[3].markdown("**Mean Target**")
            cols[4].markdown("**High Target**")

            values = [
                current_price,
                target_low,
                target_median,
                target_mean,
                target_high,
            ]

            formatted = [f"${v:,.2f}" if v is not None else "N/A" for v in values]

            cols = st.columns([1, 1, 1, 1, 1])
            for col, val in zip(cols, formatted):
                col.markdown(val)

            st.markdown(f"**Number of Analysts:** {num_analysts if num_analysts else 'N/A'}")

            st.markdown("### Price Target Range Visualization")

            if all(v is not None for v in [target_low, target_high, current_price]):
                st.markdown(f"- **Low Target:** ${target_low:.2f}")
                st.markdown(f"- **Current Price:** ${current_price:.2f}")
                st.markdown(f"- **High Target:** ${target_high:.2f}")
            else:
                st.info("Insufficient data to display price target range visualization.")
        else:
            st.info("No Wall Street price target data available for this stock.")


# Insert this line at the end of any logical section (e.g., scenario modeling)
wall_street_price_targets(ticker)

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

def render_price_with_forecast(ticker_symbol: str):
    # 1) Load historical data (1 year)
    tkr = yf.Ticker(ticker_symbol)
    hist = tkr.history(period="1y")
    if hist.empty:
        st.warning(f"No historical data for {ticker_symbol.upper()}.")
        return

    # 2) Get current & target prices
    info = tkr.info
    current = info.get("currentPrice")
    low_tgt = info.get("targetLowPrice")
    mean_tgt= info.get("targetMeanPrice")
    high_tgt= info.get("targetHighPrice")
    if None in (current, low_tgt, mean_tgt, high_tgt):
        st.warning(f"No analyst targets for {ticker_symbol.upper()}.")
        return

    # 3) Build forecast index (12 monthly steps after last hist date)
    last_date = hist.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=12,
        freq='M'
    )

    # 4) Linearly interpolate from current to each target
    months = np.arange(1, 13)
    low_proj  = np.linspace(current, low_tgt, 12)
    mean_proj = np.linspace(current, mean_tgt, 12)
    high_proj = np.linspace(current, high_tgt, 12)

    # 5) Plotly figure
    fig = go.Figure()

    # A) Historical price
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist['Close'],
        mode='lines', name='Historical',
        line=dict(color='white', width=2),
        hovertemplate='%{x|%b %Y}: $%{y:.2f}<extra></extra>'
    ))

    # B) Forecast lines
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=low_proj,
        mode='lines', name='Low Forecast',
        line=dict(color='gray', dash='dot', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=mean_proj,
        mode='lines', name='Mean Forecast',
        line=dict(color='lightblue', dash='dot', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=high_proj,
        mode='lines', name='High Forecast',
        line=dict(color='green', dash='dot', width=2)
    ))

    # 6) Layout styling
    fig.update_layout(
        title=f"{ticker_symbol.upper()} Price & 12‑Month Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        legend=dict(bgcolor='rgba(0,0,0,0.5)'),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

ticker = st.text_input("Ticker to Forecast", "AAPL")
if ticker:
    st.markdown("## 📈 Price History + 12‑Month Analyst Forecast")
    render_price_with_forecast(ticker)









import streamlit as st
import streamlit.components.v1 as components

st.markdown("## 📊 Wall Street Forecast & Technical Rating (TradingView)")

components.html(f"""
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <div class="tradingview-widget-copyright">
    <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
      <span class="blue-text">Track all markets on TradingView</span>
    </a>
  </div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
  {{
    "interval": "1h",
    "width": "100%",
    "isTransparent": false,
    "height": "500",
    "symbol": "NASDAQ:{ticker.upper()}",
    "showIntervalTabs": true,
    "displayMode": "multiple",
    "locale": "en",
    "colorTheme": "light"
  }}
  </script>
</div>
""", height=500)



import streamlit as st
import yfinance as yf
import pandas as pd

# ----------------- INSIDER & INSTITUTIONAL OWNERSHIP -----------------
def insider_institutional_section(ticker: str):
    with st.container():
        st.header("🏛️ Insider & Institutional Ownership")
        st.subheader(f"📊 Ownership Data for: {ticker}")

        try:
            ticker_obj = yf.Ticker(ticker)

            # Major Holders Breakdown
            with st.expander("🏢 Major Holders Breakdown"):
                st.markdown("#### Ownership Distribution (as % of total shares)")

                try:
                    major_holders = ticker_obj.get_major_holders()

                    if major_holders is not None and not major_holders.empty:
                        if len(major_holders.columns) == 1:
                            major_holders.columns = ["Value"]

                        major_holders = (
                            major_holders.reset_index()
                            .rename(columns={"index": "Holder"})
                        )

                        total = major_holders["Value"].sum()
                        major_holders["% Ownership"] = (
                            major_holders["Value"] / total * 100
                        ).map("{:.2f}%".format)

                        st.dataframe(
                            major_holders[["Holder", "% Ownership"]],
                            use_container_width=True,
                        )
                    else:
                        st.info("No major holders data available.")
                except Exception as e:
                    st.warning(f"Could not load Major Holders data: {e}")

            # Institutional Holders
            with st.expander("🏦 Institutional Holders"):
                st.markdown("#### Top Institutional Holders")

                try:
                    inst_holders = ticker_obj.institutional_holders
                    if inst_holders is not None and not inst_holders.empty:
                        st.dataframe(inst_holders, use_container_width=True)
                    else:
                        st.info("No institutional holders data available.")
                except Exception as e:
                    st.error(f"Failed to load institutional holders: {e}")

            # Insider Transactions
            with st.expander("👥 Insider Transactions"):
                st.markdown("#### Insider Activity Log")

                try:
                    insider_trades = ticker_obj.insider_transactions
                    if insider_trades is not None and not insider_trades.empty:
                        st.dataframe(insider_trades, use_container_width=True)
                    else:
                        st.info("No insider transaction data available.")
                except Exception as e:
                    st.error(f"Failed to load insider transactions: {e}")

        except Exception as e:
            st.error(f"Ownership section error: {e}")

# --- Call the function after ticker is defined ---
if ticker:
    insider_institutional_section(ticker)

st.markdown("## 🗓️ Global Economic Events (TradingView)")

components.html("""
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <div class="tradingview-widget-copyright">
    <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
      <span class="blue-text">Track all markets on TradingView</span>
    </a>
  </div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
  {
    "colorTheme": "dark",
    "isTransparent": false,
    "locale": "en",
    "countryFilter": "ar,au,br,ca,cn,fr,de,in,id,it,jp,kr,mx,ru,sa,za,tr,gb,us,eu",
    "importanceFilter": "-1,0,1",
    "width": "100%",
    "height": 550
  }
  </script>
</div>
""", height=600)


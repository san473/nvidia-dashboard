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
import matplotlib.pyplot as plt
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



st.write("Income statement rows:", income_stmt.index.tolist())
st.write("Income statement columns:", income_stmt.columns.tolist())

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
import yfinance as yf
import datetime
import requests
from bs4 import BeautifulSoup

def earnings_call_section(ticker: str):
    st.markdown("## 📞 Earnings Call Summary")

    try:
        yf_ticker = yf.Ticker(ticker)
        earnings_dates = yf_ticker.calendar

        # Get last earnings date
        if not earnings_dates.empty and 'Earnings Date' in earnings_dates.index:
            earnings_date = earnings_dates.loc['Earnings Date'][0]
            earnings_date_str = earnings_date.strftime('%Y-%m-%d') if isinstance(earnings_date, datetime.datetime) else str(earnings_date)
        else:
            earnings_date_str = "Date not available"

        st.write(f"**Earnings Date:** {earnings_date_str}")

        # --- Try to find webcast/audio link from official IR page ---
        ir_links = {
            "NVDA": "https://www.nvidia.com/en-us/about-nvidia/investor-relations/financial-info/",
            "AAPL": "https://investor.apple.com/earnings/default.aspx"
        }

        audio_url = None
        if ticker.upper() in ir_links:
            response = requests.get(ir_links[ticker.upper()], timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Try to find audio/video links (simple logic for demo purposes)
            for link in soup.find_all("a", href=True):
                href = link['href']
                if any(ext in href.lower() for ext in [".mp3", ".m4a", "webcast", "audio", "eventid"]):
                    audio_url = href
                    if not audio_url.startswith("http"):
                        audio_url = "https://investor.apple.com" + audio_url if "apple" in ir_links[ticker.upper()] else "https://www.nvidia.com" + audio_url
                    break

        if audio_url:
            st.audio(audio_url)
        else:
            st.info("🎧 No earnings call audio/webcast links found.")

        # --- Static summary placeholder ---
        st.markdown("""
        ### Summary
        This section will soon include a dynamic summary of the most recent earnings call, extracted via NLP summarization from transcript or audio highlights.
        """)

    except Exception as e:
        st.error(f"Failed to load earnings call section: {e}")

# --- In your main Streamlit app ---
with st.container():
    earnings_call_section(ticker)

import streamlit as st
import yfinance as yf

st.header("📊 Portfolio Exposure & Risk Dashboard (Custom Input)")

# User inputs ticker
ticker = st.text_input("Enter Stock Ticker", value="NVDA").upper()
if ticker:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get("currentPrice", None)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        beta = info.get("beta", "N/A")
        long_name = info.get("longName", ticker)

        if not current_price:
            st.error("Unable to fetch current stock price.")
        else:
            st.subheader("1️⃣ Portfolio & Holding Inputs")

            # Portfolio input
            portfolio_value = st.number_input(
                "Total Portfolio Value ($)", value=1_000_000, step=10_000
            )
            holding_value = st.number_input(
                f"Current Holding Value in {ticker} ($)", value=50_000, step=1_000
            )
            position_pct = (holding_value / portfolio_value) * 100

            st.markdown(f"- **{ticker} position = {position_pct:.2f}% of portfolio**")
            st.markdown(f"- **Stock Sector**: {sector} | **Industry**: {industry}")
            st.markdown(f"- **Stock Beta**: {beta}")

            st.subheader("2️⃣ Sector/Industry Exposure Impact")

            current_sector_pct = st.slider(
                f"Your current exposure to {sector} (%)",
                min_value=0.0,
                max_value=100.0,
                value=12.0,
                step=0.1,
            )
            new_sector_pct = current_sector_pct + position_pct

            st.markdown(
                f"- After adding this position, **sector exposure becomes: {new_sector_pct:.2f}%**"
            )

            st.subheader("3️⃣ Upside / Downside Scenario Modeling")

            col1, col2, col3 = st.columns(3)
            with col1:
                bull_price = st.number_input(
                    "🎯 Bull Case Price", value=round(current_price * 1.3, 2)
                )
            with col2:
                base_price = st.number_input(
                    "📌 Base Case Price", value=round(current_price, 2)
                )
            with col3:
                bear_price = st.number_input(
                    "⚠️ Bear Case Price", value=round(current_price * 0.7, 2)
                )

            def scenario_output(label, target_price):
                pct_change = (target_price - current_price) / current_price * 100
                value_change = (pct_change / 100) * holding_value
                st.markdown(
                    f"**{label} Case** → Target: ${target_price:.2f} → "
                    f"Return: {pct_change:.1f}% → ${value_change:,.0f} gain/loss"
                )

            st.divider()
            st.subheader("📈 Scenario Results")
            scenario_output("🎯 Bull", bull_price)
            scenario_output("📌 Base", base_price)
            scenario_output("⚠️ Bear", bear_price)

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")

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

            valid_targets = [v for v in [target_low, target_high, current_price] if v is not None]
            if len(valid_targets) >= 2:
                min_price, max_price = min(valid_targets), max(valid_targets)
                scale = max_price - min_price if max_price > min_price else 1

                def scaled_bar(value):
                    length = int(((value - min_price) / scale) * 20)  # max 20 blocks
                    bar = "█" * length
                    return f"{bar} {value:.2f}"

                st.markdown(f"- **Low Target:** {scaled_bar(target_low) if target_low else 'N/A'}")
                st.markdown(f"- **Current Price:** {scaled_bar(current_price) if current_price else 'N/A'}")
                st.markdown(f"- **High Target:** {scaled_bar(target_high) if target_high else 'N/A'}")
            else:
                st.info("Insufficient data to display price target range visualization.")
        else:
            st.info("No Wall Street price target data available for this stock.")

# Insert this line at the end of any logical section (e.g., scenario modeling)
wall_street_price_targets(ticker)

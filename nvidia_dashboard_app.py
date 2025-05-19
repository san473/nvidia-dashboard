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

# ------------------------ DCF VALUATION ------------------------
with st.expander("💰 Discounted Cash Flow (DCF) Valuation"):
    try:
        stock = yf.Ticker(ticker)
        cashflow = stock.cashflow
        st.write("📊 Available Cashflow Index:", list(cashflow.index))

        if 'Total Cash From Operating Activities' in cashflow.index:
            op_cash = cashflow.loc['Total Cash From Operating Activities']
        else:
            op_match = [i for i in cashflow.index if "operating" in i.lower()]
            if not op_match:
                raise ValueError("Operating cash flow not found.")
            op_cash = cashflow.loc[op_match[0]]
            st.info(f"Using fallback row: **{op_match[0]}**")

        avg_cashflow = op_cash.mean()
        discount_rate = 0.10
        growth_rate = 0.05
        intrinsic_value = avg_cashflow * (1 + growth_rate) / (discount_rate - growth_rate)

        # Get current market price
        market_price = stock.history(period="1d")["Close"][-1]
        diff = intrinsic_value - market_price

        st.success(f"Intrinsic Value Estimate: **${intrinsic_value:,.2f}**")
        st.info(f"Current Market Price: **${market_price:.2f}**")

        if intrinsic_value > market_price:
            st.success("✅ The stock appears **undervalued** based on DCF.")
        else:
            st.warning("⚠️ The stock appears **overvalued** based on DCF.")

    except Exception as e:
        st.error(f"DCF valuation failed: {e}")
        st.warning("DCF valuation data not available.")


# ------------------------ PEER COMPARISON ------------------------
st.header("🔍 Peer Comparison")

@st.cache_data
def get_peers(ticker):
    stock = yf.Ticker(ticker)
    try:
        sector = stock.info.get("sector")
        if not sector:
            return None
        all_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "ADBE", "ORCL"]
        peers = []
        for peer in all_tickers:
            try:
                peer_info = yf.Ticker(peer).info
                if peer_info.get("sector") == sector:
                    peers.append({
                        "Ticker": peer,
                        "Name": peer_info.get("shortName"),
                        "Market Cap": peer_info.get("marketCap"),
                        "P/E": peer_info.get("trailingPE"),
                        "Forward P/E": peer_info.get("forwardPE"),
                        "Price to Book": peer_info.get("priceToBook"),
                        "ROE": peer_info.get("returnOnEquity"),
                        "Debt to Equity": peer_info.get("debtToEquity")
                    })
            except:
                continue
        return pd.DataFrame(peers)
    except:
        return None

peer_df = get_peers(ticker)
if peer_df is not None and not peer_df.empty:
    st.dataframe(peer_df.set_index("Ticker"))
else:
    st.warning("No peer data available for this ticker.")

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


# ---------------------------------------------
# 

# ------------------------ INTERACTIVE VISUALIZATIONS ------------------------
st.header("📊 Interactive Visualizations")

stock = yf.Ticker(ticker)

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

# -------- Financial Ratios Bar Chart --------
with st.expander("📊 Financial Ratios"):
    try:
        ratios_for_chart = {
            "P/E": ratios.get("Trailing P/E"),
            "Forward P/E": ratios.get("Forward P/E"),
            "Price to Book": ratios.get("Price to Book"),
            "ROA": ratios.get("Return on Assets (ROA)"),
            "ROE": ratios.get("Return on Equity (ROE)"),
            "Profit Margin": ratios.get("Profit Margin"),
            "Debt to Equity": ratios.get("Debt to Equity")
        }

        # Filter out non-numeric values
        ratios_cleaned = {k: v for k, v in ratios_for_chart.items() if isinstance(v, (int, float))}

        if not ratios_cleaned:
            st.warning("No valid numeric ratios available to plot.")
        else:
            fig_ratios = go.Figure([go.Bar(
                x=list(ratios_cleaned.keys()),
                y=list(ratios_cleaned.values()),
                marker_color='mediumseagreen'
            )])
            fig_ratios.update_layout(
                title="Key Financial Ratios",
                xaxis_title="Ratio",
                yaxis_title="Value",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_ratios, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to render financial ratios chart: {e}")


# -------- KPI Toggles --------
st.subheader("🎛️ KPI Toggles")
kpi_options = ["Revenue Growth", "Net Margin", "Sales"]
selected_kpi = st.selectbox("Select KPI", kpi_options)

try:
    if selected_kpi == "Revenue Growth":
        rev = income_stmt.loc['Total Revenue']
        rev = rev.sort_index()
        growth = rev.pct_change().dropna() * 100
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
        sales = income_stmt.loc['Total Revenue']
        sales = sales.sort_index()
        fig_kpi = go.Figure()
        fig_kpi.add_trace(go.Scatter(x=sales.index, y=sales.values / 1e9, mode='lines+markers', name="Sales"))
        fig_kpi.update_layout(title="Sales Over Time", xaxis_title="Year", yaxis_title="Sales (B USD)")
        st.plotly_chart(fig_kpi, use_container_width=True)
except Exception as e:
    st.warning(f"Unable to display KPI: {e}")

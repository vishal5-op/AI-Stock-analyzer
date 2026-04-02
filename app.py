import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
from newsapi import NewsApiClient

# ================== CONFIG ==================
st.set_page_config(layout="wide")
st.title("🚀 AI Stock Analysis PRO Dashboard")

# ================== FORMAT FUNCTION ==================
def format_in_cr(value):
    try:
        if value is None:
            return "-"
        if abs(value) >= 1e7:
            return f"{value/1e7:,.2f} Cr"
        elif abs(value) >= 1e5:
            return f"{value/1e5:,.2f} L"
        else:
            return f"{value:,.0f}"
    except:
        return value

# ================== INPUT ==================
stock = st.text_input("Enter Stock Symbol", "RELIANCE.NS")

if stock:
    ticker = yf.Ticker(stock)

    # ================== PRICE DATA ==================
    data = ticker.history(period="6mo")

    if data.empty:
        st.error("No data found")
        st.stop()

    latest_price = float(data['Close'].iloc[-1])

    # ================== AI MODEL ==================
    df = data.copy()
    df['Target'] = df['Close'].shift(-1)
    df = df[['Close', 'Target']].dropna()

    X = df['Close'].values.reshape(-1, 1)
    y = df['Target'].values

    model = RandomForestRegressor()
    model.fit(X, y)

    next_day = model.predict(np.array([[latest_price]]))[0]

    # ================== METRICS ==================
    col1, col2 = st.columns(2)
    col1.metric("Current Price", f"₹{latest_price:,.2f}")
    col2.metric("Predicted Price", f"₹{next_day:,.2f}")

    if next_day > latest_price:
        st.success("📈 BUY Signal")
    else:
        st.error("📉 SELL Signal")

    # ================== TABS ==================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Chart",
        "📰 News",
        "📊 Financials",
        "📈 Valuation",
        "📉 Earnings Chart"
    ])

# ================== TAB 1: CHART ==================
st.subheader("📊 Advanced Stock Chart")

# -------- TIMEFRAME SELECTOR --------
timeframe = st.selectbox(
    "Select Timeframe",
    ["1D", "15D", "1M", "3M", "6M", "1Y", "5Y", "10Y"]
)

# -------- FETCH DATA --------
if timeframe == "1D":
    data = ticker.history(period="1d", interval="5m")
else:
    period_map = {
        "15D": "15d",
        "1M": "1mo",
        "3M": "3mo",
        "6M": "6mo",
        "1Y": "1y",
        "5Y": "5y",
        "10Y": "10y"
    }
    data = ticker.history(period=period_map[timeframe])

# -------- CHECK DATA --------
if data.empty:
    st.error("No data available")
else:

    # -------- MOVING AVERAGES --------
    data['MA9'] = data['Close'].rolling(window=9).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()

    # -------- CANDLESTICK --------
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ))

    # -------- MOVING AVERAGE LINES --------
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA9'],
        mode='lines',
        name='MA9'
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA20'],
        mode='lines',
        name='MA20'
    ))

    # -------- VOLUME --------
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.3
    ))

    # -------- LAYOUT --------
    fig.update_layout(
        title=f"{stock} Chart ({timeframe})",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        yaxis2=dict(
            title="Volume",
            overlaying='y',
            side='right'
        ),
        height=700,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)
    # ================== TAB 2: NEWS ==================
    with tab2:
        st.subheader("📰 News Sentiment Analysis")

        newsapi = NewsApiClient(api_key="e90651a12ef044659d4a5ce457e544d7")

        news = newsapi.get_everything(q=stock.split(".")[0], language='en', page_size=10)

        sentiment_score = 0

        for article in news['articles']:
            title = article['title']
            polarity = TextBlob(title).sentiment.polarity

            st.write(f"{title}")
            st.write(f"Sentiment: {polarity:.2f}")
            st.write("---")

            sentiment_score += polarity

        if news['articles']:
            avg = sentiment_score / len(news['articles'])

            st.subheader(f"Overall Sentiment: {avg:.2f}")

            if avg > 0:
                st.success("🟢 Positive News → BUY Bias")
            elif avg < 0:
                st.error("🔴 Negative News → SELL Bias")
            else:
                st.warning("⚪ Neutral")

    # ================== TAB 3: FINANCIALS ==================
    with tab3:
        st.subheader("📊 Financial Statements")

        income = ticker.income_stmt
        balance = ticker.balance_sheet
        cashflow = ticker.cashflow

        for name, df in {
            "Income Statement": income,
            "Balance Sheet": balance,
            "Cash Flow": cashflow
        }.items():

            st.write(f"### {name}")

            if df is not None and not df.empty:
                df = df.copy()

                for col in df.columns:
                    df[col] = df[col].apply(format_in_cr)

                st.dataframe(df, use_container_width=True)
            else:
                st.warning(f"No data for {name}")

    # ================== TAB 4: VALUATION ==================
    with tab4:
        st.subheader("💰 Valuation Metrics")

        info = ticker.info

        col1, col2, col3 = st.columns(3)

        col1.metric("PE Ratio", info.get("trailingPE"))
        col2.metric("EPS", info.get("trailingEps"))
        col3.metric("Market Cap", format_in_cr(info.get("marketCap")))

        col4, col5 = st.columns(2)

        col4.metric("Revenue", format_in_cr(info.get("totalRevenue")))
        col5.metric("Profit Margin", info.get("profitMargins"))

    # ================== TAB 5: EARNINGS ==================
   
st.subheader("📈 Revenue & Profit Trend")

income = ticker.income_stmt

if income is not None and not income.empty:

    # -------- GET DATA --------
    try:
        revenue = income.loc["Total Revenue"].dropna()
    except:
        revenue = None

    try:
        profit = income.loc["Net Income"].dropna()
    except:
        profit = None

    fig = go.Figure()

    # -------- REVENUE LINE --------
    if revenue is not None:
        fig.add_trace(go.Scatter(
            x=revenue.index,
            y=[val/1e7 for val in revenue.values],  # convert to Cr
            mode='lines+markers',
            name='Revenue (Cr)',
            line=dict(color='cyan', width=3)
        ))

    # -------- PROFIT LINE --------
    if profit is not None:
        fig.add_trace(go.Scatter(
            x=profit.index,
            y=[val/1e7 for val in profit.values],  # convert to Cr
            mode='lines+markers',
            name='Net Profit (Cr)',
            line=dict(color='green', width=3)
        ))

    # -------- LAYOUT --------
    fig.update_layout(
        title="📊 Company Earnings Trend",
        xaxis_title="Year",
        yaxis_title="Amount (Cr)",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No earnings data available")

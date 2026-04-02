import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("📊 AI Stock Analyzer (TradingView Style)")

# Sidebar
ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"])

# Fetch Data
df = yf.download(ticker, period=period)

# Fix MultiIndex (IMPORTANT)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

df.columns = df.columns.str.capitalize()

if df.empty:
    st.error("No data found")
    st.stop()

# =========================
# 📊 Candlestick Chart
# =========================
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])

fig.update_layout(
    title=f"{ticker} Candlestick Chart",
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# 🤖 Dummy AI Logic
# =========================
latest_price = df['Close'].iloc[-1]

# Fake prediction (you can replace with ML model)
next_day = latest_price * (1 + np.random.uniform(-0.02, 0.02))

# Fake sentiment (-1 to +1)
avg_sentiment = np.random.uniform(-1, 1)

# =========================
# 📊 Top Metrics
# =========================
col1, col2, col3 = st.columns(3)

price_delta = next_day - latest_price

col1.metric("💰 Current Price", f"₹ {latest_price:.2f}")

col2.metric(
    "📈 Predicted Price",
    f"₹ {next_day:.2f}",
    f"{price_delta:.2f}"
)

if avg_sentiment > 0.2:
    sentiment_label = "🟢 Positive"
elif avg_sentiment < -0.2:
    sentiment_label = "🔴 Negative"
else:
    sentiment_label = "🟡 Neutral"

col3.metric("📰 Sentiment", sentiment_label)

# =========================
# 📢 AI Signal
# =========================
st.subheader("🤖 AI Decision")

if next_day > latest_price and avg_sentiment > 0:
    st.success("🟢 BUY Signal")
elif next_day < latest_price and avg_sentiment < 0:
    st.error("🔴 SELL Signal")
else:
    st.warning("🟡 HOLD / UNCERTAIN")
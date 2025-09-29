import streamlit as st
import pandas as pd
import yfinance as yf
from stock_analysis import fetch_stock_data, calculate_moving_average, calculate_rsi, calculate_bollinger_bands, calculate_macd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Ensuring VADER lexicon is downloaded (only required once)
nltk.download('vader_lexicon')

import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.stream import Stream
    ALPACA_AVAILABLE = True
except ImportError:
    tradeapi = None
    Stream = None
    ALPACA_AVAILABLE = False

import asyncio
import threading
import os
import smtplib
from email.mime.text import MIMEText

#########################
# EMAIL ALERT HELPER FUNCTION
#########################
def send_email_alert(subject, message, from_email, to_email, smtp_server, smtp_port, smtp_username, smtp_password):
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, to_email, msg.as_string())
    print("Email alert sent.")

#########################
# REAL-TIME STREAMING SETUP
#########################
if "latest_update" not in st.session_state:
    st.session_state.latest_update = "Waiting for real-time updates..."

async def on_trade(data):
    st.session_state.latest_update = f"Trade Update: {data}"

async def on_quote(data):
    st.session_state.latest_update = f"Quote Update: {data}"
    # Check if alert threshold and email alerts are enabled
    threshold = st.session_state.get("alert_threshold", None)
    email_enabled = st.session_state.get("email_alert_enabled", False)
    if threshold is not None and email_enabled:
        ask_price = data.get("ask_price")
        if ask_price and ask_price > threshold:
            subject = f"Price Alert: {data.get('symbol', 'Ticker')} exceeds {threshold}"
            message = f"Current ask price: {ask_price}"
            from_email = st.session_state.get("from_email")
            to_email = st.session_state.get("to_email")
            smtp_server = st.session_state.get("smtp_server")
            smtp_port = st.session_state.get("smtp_port")
            smtp_username = st.session_state.get("smtp_username")
            smtp_password = st.session_state.get("smtp_password")
            try:
                send_email_alert(subject, message, from_email, to_email, smtp_server, smtp_port, smtp_username, smtp_password)
            except Exception as e:
                print("Error sending email alert:", e)

async def run_stream():
    if not ALPACA_AVAILABLE:
        print("Alpaca not available. Streaming disabled.")
        return
    API_KEY = os.getenv("APCA_API_KEY_ID", "PK72F14MY9AE5K6H4EJE")
    API_SECRET = os.getenv("APCA_API_SECRET_KEY", "9gwRVD6idycWU5byU8TsOgj3KXz2wvO10L6yZuYl")
    BASE_URL = "https://paper-api.alpaca.markets"
    DATA_FEED = "iex"
    stream = Stream(API_KEY, API_SECRET, base_url=BASE_URL, data_feed=DATA_FEED)
    stream.subscribe_trades(on_trade, "AAPL")
    stream.subscribe_quotes(on_quote, "AAPL")
    print("Subscribed to AAPL updates. Waiting for data...")
    await stream._run_forever()

def start_streaming():
    asyncio.run(run_stream())

if "streaming_thread_started" not in st.session_state:
    st.session_state.streaming_thread_started = True
    threading.Thread(target=start_streaming, daemon=True).start()

#########################
# PORTFOLIO TRACKING SETUP
#########################
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

#########################
# Helper functions for News & Sentiment Analysis
#########################
def fetch_news(ticker, api_key):
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&pageSize=5&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        headlines = [article['title'] for article in data.get('articles', [])]
        return headlines
    else:
        return []

def analyze_sentiment(headlines):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = {headline: sia.polarity_scores(headline) for headline in headlines}
    return sentiment_scores

#########################
# SIDEBAR FEATURE SELECTION
#########################
st.sidebar.title("Features")
feature = st.sidebar.selectbox("Select Feature", 
    ["Stock Analysis", "Daily Market Movers", "Real-Time Streaming", "Portfolio Tracking", "Strategy & Predictions"])

#########################
# STOCK ANALYSIS WITH INTERACTIVE CHARTING & NEWS/SENTIMENT
#########################
if feature == "Stock Analysis":
    st.title("Stock Analysis with Technical Indicators & News")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
    start_date = st.date_input("Start Date:", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date:", pd.to_datetime("2023-01-01"))
    
    # Allow user to select which indicators to display
    show_ma = st.checkbox("Show Moving Average", value=True)
    show_bbands = st.checkbox("Show Bollinger Bands", value=True)
    show_macd = st.checkbox("Show MACD", value=True)
    show_rsi = st.checkbox("Show RSI", value=True)
    
    if st.button("Analyze"):
        try:
            data = fetch_stock_data(ticker, start_date, end_date)
            if not data.empty:
                data = calculate_moving_average(data)
                data = calculate_rsi(data)
                data = calculate_bollinger_bands(data)
                data = calculate_macd(data)
                
                # Create two tabs: one for charts and one for news & sentiment
                tab1, tab2 = st.tabs(["Advanced Charting", "News & Sentiment"])
                
                with tab1:
                    st.subheader(f"{ticker} Price Chart with Indicators")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close", line=dict(color="blue")))
                    if show_ma:
                        fig.add_trace(go.Scatter(x=data.index, y=data['MA_20'], name="MA (20)", line=dict(color="orange")))
                    if show_bbands:
                        fig.add_trace(go.Scatter(x=data.index, y=data['Upper Band'], name="Upper Band", line=dict(color="green"), opacity=0.5))
                        fig.add_trace(go.Scatter(x=data.index, y=data['Lower Band'], name="Lower Band", line=dict(color="red"), opacity=0.5))
                    fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
                    fig.update_xaxes(rangeslider_visible=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if show_macd:
                        st.subheader(f"{ticker} MACD")
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], name="MACD", line=dict(color="purple")))
                        fig_macd.add_trace(go.Scatter(x=data.index, y=data['Signal Line'], name="Signal Line", line=dict(color="black")))
                        fig_macd.update_layout(title=f"{ticker} MACD", xaxis_title="Date", yaxis_title="MACD")
                        st.plotly_chart(fig_macd, use_container_width=True)
                    
                    if show_rsi:
                        st.subheader(f"{ticker} RSI")
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color="magenta")))
                        fig_rsi.update_layout(title=f"{ticker} RSI", xaxis_title="Date", yaxis_title="RSI")
                        st.plotly_chart(fig_rsi, use_container_width=True)
                
                with tab2:
                    st.subheader("News & Sentiment Analysis")
                    # Use your NewsAPI key (consider using an environment variable)
                    news_api_key = "f45ba8b0a730487dac4d4e8a15148964"
                    headlines = fetch_news(ticker, news_api_key)
                    if headlines:
                        st.write("Latest Headlines:")
                        for headline in headlines:
                            st.write("- " + headline)
                        sentiment = analyze_sentiment(headlines)
                        st.write("Sentiment Scores:")
                        for headline, scores in sentiment.items():
                            st.write(f"**{headline}**: {scores}")
                    else:
                        st.warning("No news found for this ticker or check your API key.")
            else:
                st.warning("No data available for the given ticker and date range.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

#########################
# DAILY MARKET MOVERS (Existing Feature)
#########################
elif feature == "Daily Market Movers":
    st.header("Daily Market Movers")
    popular_stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
    popular_cryptos = ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "XRP-USD"]
    all_tickers = popular_stocks + popular_cryptos

    def fetch_daily_changes(tickers):
        data = yf.download(tickers, period="2d", interval="1d", auto_adjust=True, group_by='ticker')
        pct_changes = {}
        if not isinstance(data.columns, pd.MultiIndex):
            data = {tickers[0]: data}
        for ticker in tickers:
            try:
                df = data[ticker]
            except Exception:
                continue
            if df is None or df.empty or len(df) < 2:
                continue
            first_price = df['Close'].iloc[0]
            last_price = df['Close'].iloc[-1]
            pct_change = ((last_price - first_price) / first_price) * 100
            pct_changes[ticker] = pct_change
        return pct_changes

    daily_changes = fetch_daily_changes(all_tickers)
    if daily_changes:
        df_changes = pd.DataFrame(list(daily_changes.items()), columns=["Ticker", "Daily Change (%)"])
        df_changes.sort_values("Daily Change (%)", ascending=False, inplace=True)

        fig = px.bar(
            df_changes,
            x="Ticker",
            y="Daily Change (%)",
            color="Daily Change (%)",
            color_continuous_scale="RdYlGn",
            title="Daily Performance of Popular Stocks and Cryptos"
        )
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig)
    else:
        st.warning("No data available for market movers.")

#########################
# REAL-TIME STREAMING (Existing Feature)
#########################
elif feature == "Real-Time Streaming":
    st.header("Real-Time AAPL Market Updates with Alerts")
    st_autorefresh(interval=1000, limit=1000, key="streamrefresh")
    st.write(st.session_state.latest_update)
    # Email alert configuration inputs
    alert_threshold = st.number_input("Set Alert Threshold (ask price)", value=150.0, step=1.0)
    email_alert_enabled = st.checkbox("Enable Email Alerts", value=False)
    from_email = st.text_input("From Email", "your_email@example.com")
    to_email = st.text_input("To Email", "recipient_email@example.com")
    smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
    smtp_port = st.number_input("SMTP port", value=465, step=1)
    smtp_username = st.text_input("SMTP Username", "your_email@example.com")
    smtp_password = st.text_input("SMTP Password", type="password")

    # Store alert configuration in session state for use in callbacks
    st.session_state.alert_threshold = alert_threshold
    st.session_state.email_alert_enabled = email_alert_enabled
    st.session_state.from_email = from_email
    st.session_state.to_email = to_email
    st.session_state.smtp_server = smtp_server
    st.session_state.smtp_port = smtp_port
    st.session_state.smtp_username = smtp_username
    st.session_state.smtp_password = smtp_password

#########################
# PORTFOLIO TRACKING
#########################
elif feature == "Portfolio Tracking":
    st.header("Portfolio Tracking")
    st.write("Enter your trades:")
    with st.form("trade_form"):
        trade_ticker = st.text_input("Ticker", "AAPL")
        trade_quantity = st.number_input("Quantity", value=1, min_value=1)
        trade_price = st.number_input("Buy Price", value=100.0, min_value=0.0, format="%.2f")
        submitted = st.form_submit_button("Add Trade")
        if submitted:
            st.session_state.portfolio.append({
                "Ticker": trade_ticker.upper(),
                "Quantity": trade_quantity,
                "Buy Price": trade_price
            })
            st.success("Trade added!")
    
    if st.session_state.portfolio:
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        st.write("Your Trades:")
        st.dataframe(portfolio_df)
        current_values = []
        for ticker in portfolio_df["Ticker"].unique():
            data = yf.download(ticker, period="1d", interval="1m", auto_adjust=True)
            if not data.empty:
                current_price = data['Close'].iloc[-1]
            else:
                current_price = None
            current_values.append({"Ticker": ticker, "Current Price": current_price})
        current_prices_df = pd.DataFrame(current_values)
        st.write("Current Prices:")
        st.dataframe(current_prices_df)
        portfolio_summary = portfolio_df.merge(current_prices_df, on="Ticker", how="left")
        portfolio_summary["Current Value"] = portfolio_summary["Quantity"] * portfolio_summary["Current Price"]
        portfolio_summary["Cost Basis"] = portfolio_summary["Quantity"] * portfolio_summary["Buy Price"]
        portfolio_summary["P/L"] = portfolio_summary["Current Value"] - portfolio_summary["Cost Basis"]
        st.write("Portfolio Summary:")
        st.dataframe(portfolio_summary)

#########################
# STRATEGY SIMULATION & ML PREDICTIONS
#########################
elif feature == "Strategy & Predictions":
    st.title("Strategy Simulation & ML Predictions")

    # User inputs for backtesting
    ticker = st.text_input("Enter Stock Ticker for backtesting", "AAPL")
    start_date = st.date_input("Backtesting Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Backtesting End date", pd.to_datetime("2023-01-01"))
    short_window = st.number_input("Short Moving Average Window", value=20, min_value=1)
    long_window = st.number_input("Long Moving Average Window", value=50, min_value=1)
    forecast_days = st.number_input("Forecast Days", value=10, min_value=1)
    alert_price = st.number_input("Set Alert Price", value=150.0, min_value=0.0, format="%.2f")

    if st.button("Run Backtest & Predict"):
        # Fetching historical data
        data = fetch_stock_data(ticker, start_date, end_date)
        if data.empty:
            st.warning("No data available for the given ticker and the date range.")
        else:
            # Backtesting: Simple moving average crossover strategy
            data['short_ma'] = data['Close'].rolling(window=short_window).mean()
            data['long_ma'] = data['Close'].rolling(window=long_window).mean()
            data = data.dropna()
            # Generating a simple signal: 1 if short_ma > long_ma, else 0
            data['signal'] = np.where(data['short_ma'] > data['long_ma'], 1, 0)
            data['positions'] = data['signal'].diff()
            initial_capital = 100000
            # Simplified simulation: Assume you hold a position when signal==1.
            # Portfolio value is updated based on daily returns.
            data['daily_return'] = data['Close'].pct_change().fillna(0)
            data['portfolio_value'] = initial_capital * (1 + data['daily_return'] * data['signal']).cumprod()
            cumulative_return = (data['portfolio_value'].iloc[-1] / initial_capital) - 1

            # Machine learning predictions: Linear regression forecast
            data['Data_ordinal'] = data.index.map(pd.Timestamp.toordinal)
            X = data[['Data_ordinal']]
            y = data['Close']
            model = LinearRegression()
            model.fit(X, y)
            last_date = data.index[-1]
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days+1)]
            future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1,1)
            predictions = model.predict(future_ordinals)
            pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions})

            # Create tabs for backtesting and ML predictions
            tab1, tab2 = st.tabs(["Backtesting", "ML Predictions & Alerts"])

            with tab1:
                st.subheader("Backtesting Results")
                st.write(f"Cumulative Return: {cumulative_return*100:.2f}%")
                st.line_chart(data['portfolio_value'])
                
            with tab2:
                st.subheader("ML Predictions")
                st.dataframe(pred_df)
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted_Close'],
                                              mode='lines+markers', name='Predicted Close'))
                fig_pred.update_layout(title="Predicted Future Prices", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_pred, use_container_width=True)

                # Show current price and alert if it exceeds threshold
                current_price = data['Close'].iloc[-1]
                st.write("Current Price:", current_price)
                if current_price > alert_price:
                    st.warning(f"Alert: {ticker} current price {current_price:.2f} exceeds threshold {alert_price:.2f}")
                else:
                    st.info(f"{ticker} current price {current_price:.2f} is below threshold {alert_price:.2f}")

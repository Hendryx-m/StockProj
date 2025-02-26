import streamlit as st
import pandas as pd
import yfinance as yf

st.title("Stock Market Analyzer")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.date_input("Start Date:", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date:", pd.to_datetime("2023-01-01"))

# Fetch and analyze data
if st.button("Analyze"):
    data = yf.download(ticker, start=start_date, end=end_date)
    if not data.empty:
        st.write(f"### {ticker} Stock Data")
        st.write(data.tail())
        
        st.write("### Stock Price and Moving Average")
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        st.line_chart(data[['Close', 'MA_20']])
    else:
        st.warning("No data available for the given ticker and date range.")
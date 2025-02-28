import streamlit as st
import pandas as pd
import yfinance as yf
from stock_analysis import fetch_stock_data, calculate_moving_average, calculate_rsi
import plotly.graph_objects as go


# Defining fetch_stock_data
def fetch_stock_data(ticker, start_data, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Defining calculate_moving_average
def calculate_moving_average(data, window=20):
    data[f'M_A{window}'] = data['Close'].rolling(window=window).mean()
    return data

# Defining calculate_rsi
def calculate_rsi(data, window=14):
    # Calculate price changes
    delta = data['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    data['RSI'] = 100 - (100 / (1 + rs))
    return data



st.title("Stock Market Analyzer")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.date_input("Start Date:", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date:", pd.to_datetime("2023-01-01"))

# Fetch and analyze data
if st.button("Analyze"):
    try:
        data = fetch_stock_data(ticker, start_date, end_date)
        if not data.empty:
            data = calculate_moving_average(data)
            data = calculate_rsi(data)
            
            st.write(f"### {ticker} Stock Data")
            st.write(data.tail())
            
            st.write("### Stock Price and Moving Average")
            if 'Close' in data.columns and 'MA_20' in data.columns:  # Check if columns exist
                st.line_chart(data[['Close', 'MA_20']])
            else:
                st.warning("Required columns ('Close' or 'MA_20') not found in the data.")
            
            st.write("### RSI Indicator")
            if 'RSI' in data.columns:  # Check if RSI column exists
                st.line_chart(data['RSI'])
            else:
                st.warning("RSI data not available.")
        else:
            st.warning("No data available for the given ticker and date range.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
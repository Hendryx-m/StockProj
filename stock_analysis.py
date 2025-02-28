import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    print(stock_data.head()) 
    return stock_data

def calculate_moving_average(data, window=20):
    data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    return data

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

def calculate_bollinger_bands(data, window=20):
    data['MA_20'] = data['Close'].rolling(window=window).mean()
    data['Upper Band'] = data['MA_20'] + 2 (2 * data['Close'].rolling(window=window).std())
    data['Lower Band'] = data['MA_20'] - (2 * data['Close'].rolling(window=window).std())
    return data
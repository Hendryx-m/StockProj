import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    # Flatten columns if they are a MultiIndex
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    print(stock_data.head())
    return stock_data

def calculate_moving_average(data, window=20):
    data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    return data

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_bollinger_bands(data, window=20):
    data['MA_20'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()
    data['Upper Band'] = data['MA_20'] + (2 * data['STD'])
    data['Lower Band'] = data['MA_20'] - (2 * data['STD'])
    return data

def calculate_macd(data, span_short=12, span_long=26, signal_span=9):
    data['EMA12'] = data['Close'].ewm(span=span_short, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=span_long, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal Line'] = data['MACD'].ewm(span=signal_span, adjust=False).mean()
    return data

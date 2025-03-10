import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
import asyncio
import os

# Retrieve your paper trading API credentials
API_KEY = os.getenv("APCA_API_KEY_ID", "PK72F14MY9AE5K6H4EJE")
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "9gwRVD6idycWU5byU8TsOgj3KXz2wvO10L6yZuYl")
BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading endpoint
DATA_FEED = "iex"  # Data feed for equities

# Define asynchronous callback functions for trades and quotes
async def on_trade(data):
    print("Trade Update:", data)

async def on_quote(data):
    print("Quote Update:", data)

async def run_stream():
    # Create a Stream instance with your credentials
    stream = Stream(API_KEY, API_SECRET, base_url=BASE_URL, data_feed=DATA_FEED)
    
    # Subscribe to trade and quote updates for AAPL
    stream.subscribe_trades(on_trade, "AAPL")
    stream.subscribe_quotes(on_quote, "AAPL")
    
    print("Subscribed to AAPL updates. Waiting for data...")
    
    # Run the stream indefinitely
    await stream._run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(run_stream())
    except KeyboardInterrupt:
        print("Stream stopped by user.")
        


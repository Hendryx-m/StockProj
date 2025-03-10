import os
from alpaca_trade_api.rest import REST

# Retrieve your paper trading API credentials
API_KEY = os.getenv("APCA_API_KEY_ID", "PK72F14MY9AE5K6H4EJE")
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "9gwRVD6idycWU5byU8TsOgj3KXz2wvO10L6yZuYl")
BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading endpoint

api = REST(API_KEY, API_SECRET, BASE_URL)
account = api.get_account()
print(account)

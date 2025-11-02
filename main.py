from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv

load_dotenv()


ALPACA_MARKET_API_KEY = os.getenv('ALPACA_MARKET_API_KEY')
ALPACA_MARKET_API_SECRET = os.getenv('ALPACA_MARKET_API_SECRET')


trading_client = TradingClient(ALPACA_MARKET_API_KEY, ALPACA_MARKET_API_SECRET)

print(trading_client.get_account().account_number)
print(trading_client.get_account().buying_power)
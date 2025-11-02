#!/usr/bin/env python3
"""Quick script to verify orders in Alpaca."""

from alpaca.trading.client import TradingClient
from config.settings import settings

client = TradingClient(settings.alpaca_api_key, settings.alpaca_api_secret, paper=True)

# Check orders
orders = client.get_orders()
print(f'Total orders in Alpaca: {len(orders)}')
print(f'\nMost recent orders:')
for order in orders[:3]:
    print(f'  - {order.symbol}: {order.side} {order.qty} shares, Status: {order.status}')
    print(f'    Order ID: {order.id}')
    print(f'    Submitted: {order.submitted_at}')

# Check account
account = client.get_account()
print(f'\nAccount Status:')
print(f'  Buying Power: ${float(account.buying_power):,.2f}')
print(f'  Cash: ${float(account.cash):,.2f}')

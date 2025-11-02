# Ticket 01: Market Data Streaming Service

**Created**: 2025-11-02
**Priority**: HIGH
**Estimate**: 4 hours
**Dependencies**: None
**Status**: TODO

## Description

Implement a service to stream live market data from Alpaca using their WebSocket API. This will provide real-time trades, quotes, and bars for AAPL stock.

## Requirements

1. Connect to Alpaca WebSocket stream
2. Subscribe to AAPL trades, quotes, and bars
3. Parse incoming messages
4. Emit events for downstream consumers
5. Handle reconnection on disconnect
6. Graceful shutdown

## Implementation Details

### File: `services/market_stream_service.py`

```python
import asyncio
from typing import Callable, Optional
from alpaca.data.live import StockDataStream
from alpaca.data.models import Trade, Quote, Bar
from utils.logger import get_logger

class MarketStreamService:
    """Service for streaming live market data from Alpaca."""

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.logger = get_logger("market_stream")
        self.stream = StockDataStream(api_key, api_secret, feed="iex" if paper else "sip")
        self.is_running = False

    async def subscribe_trades(self, symbols: list[str], callback: Callable):
        """Subscribe to trade updates."""
        async def trade_handler(trade: Trade):
            await callback(trade)

        self.stream.subscribe_trades(trade_handler, *symbols)

    async def subscribe_quotes(self, symbols: list[str], callback: Callable):
        """Subscribe to quote updates."""
        async def quote_handler(quote: Quote):
            await callback(quote)

        self.stream.subscribe_quotes(quote_handler, *symbols)

    async def subscribe_bars(self, symbols: list[str], callback: Callable):
        """Subscribe to bar updates (1-minute)."""
        async def bar_handler(bar: Bar):
            await callback(bar)

        self.stream.subscribe_bars(bar_handler, *symbols)

    async def start(self):
        """Start the stream."""
        self.logger.info("starting_market_stream")
        self.is_running = True
        await self.stream.run()

    async def stop(self):
        """Stop the stream."""
        self.logger.info("stopping_market_stream")
        self.is_running = False
        await self.stream.close()
```

### File: `models/market_data.py`

```python
from pydantic import BaseModel
from datetime import datetime

class MarketTrade(BaseModel):
    """Real-time trade data."""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    exchange: str

class MarketQuote(BaseModel):
    """Real-time quote data."""
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int

class MarketBar(BaseModel):
    """1-minute bar data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
```

## Acceptance Criteria

1. ✅ Service connects to Alpaca WebSocket successfully
2. ✅ Receives real-time trades for AAPL
3. ✅ Receives real-time quotes for AAPL
4. ✅ Receives 1-minute bars for AAPL
5. ✅ Callbacks fire on each update
6. ✅ Handles disconnects and reconnects automatically
7. ✅ Clean shutdown without hanging

## Testing

### File: `tests/test_market_stream_service.py`

```python
import pytest
import asyncio
from services.market_stream_service import MarketStreamService
from config.settings import settings

@pytest.mark.asyncio
async def test_market_stream_connection():
    """Should connect to Alpaca stream."""
    service = MarketStreamService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=True
    )

    assert service.stream is not None
    assert service.is_running is False

@pytest.mark.asyncio
async def test_subscribe_trades():
    """Should receive trade updates."""
    service = MarketStreamService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=True
    )

    trades_received = []

    async def trade_callback(trade):
        trades_received.append(trade)

    await service.subscribe_trades(["AAPL"], trade_callback)

    # Start stream for 10 seconds
    asyncio.create_task(service.start())
    await asyncio.sleep(10)
    await service.stop()

    # Should have received at least one trade (during market hours)
    assert len(trades_received) > 0

@pytest.mark.asyncio
async def test_subscribe_quotes():
    """Should receive quote updates."""
    service = MarketStreamService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=True
    )

    quotes_received = []

    async def quote_callback(quote):
        quotes_received.append(quote)

    await service.subscribe_quotes(["AAPL"], quote_callback)

    asyncio.create_task(service.start())
    await asyncio.sleep(10)
    await service.stop()

    assert len(quotes_received) > 0
```

## Manual Testing

```bash
# Test during market hours (9:30 AM - 4:00 PM ET)
python -c "
import asyncio
from services.market_stream_service import MarketStreamService
from config.settings import settings

async def test_stream():
    service = MarketStreamService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=True
    )

    async def print_trade(trade):
        print(f'Trade: {trade.symbol} @ ${trade.price} x {trade.size}')

    async def print_quote(quote):
        print(f'Quote: {quote.symbol} Bid: ${quote.bid_price} Ask: ${quote.ask_price}')

    await service.subscribe_trades(['AAPL'], print_trade)
    await service.subscribe_quotes(['AAPL'], print_quote)

    print('Streaming for 30 seconds...')
    task = asyncio.create_task(service.start())
    await asyncio.sleep(30)
    await service.stop()
    print('Done!')

asyncio.run(test_stream())
"
```

## Dependencies to Install

```bash
pip install alpaca-py[data]
```

## Estimated Completion Time

4 hours

## Notes

- Use IEX feed for paper trading (free)
- SIP feed for production (requires subscription)
- WebSocket will auto-reconnect on disconnect
- Rate limits: No explicit limits on WebSocket subscriptions

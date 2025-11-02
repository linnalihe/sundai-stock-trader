# Ticket 02: Market Data Storage

**Created**: 2025-11-02
**Priority**: HIGH
**Estimate**: 3 hours
**Dependencies**: Ticket 01
**Status**: TODO

## Description

Implement storage layer for streaming market data. Maintain in-memory cache for latest prices and persist to database for historical reference and analysis.

## Requirements

1. In-memory cache for latest market data
2. Database schema for historical storage
3. Cache management (prevent memory leaks)
4. Query interface for latest prices
5. Historical data retrieval

## Implementation Details

### File: `services/market_data_cache.py`

```python
from typing import Optional, Dict
from datetime import datetime, timedelta
from models.market_data import MarketTrade, MarketQuote, MarketBar
from utils.logger import get_logger

class MarketDataCache:
    """In-memory cache for latest market data."""

    def __init__(self, max_age_minutes: int = 60):
        self.logger = get_logger("market_cache")
        self.max_age = timedelta(minutes=max_age_minutes)

        # Latest data by symbol
        self.latest_trades: Dict[str, MarketTrade] = {}
        self.latest_quotes: Dict[str, MarketQuote] = {}
        self.latest_bars: Dict[str, MarketBar] = {}

        # Historical data (limited retention)
        self.trade_history: Dict[str, list[MarketTrade]] = {}
        self.max_history = 1000  # Keep last 1000 trades per symbol

    def update_trade(self, trade: MarketTrade):
        """Update latest trade."""
        symbol = trade.symbol
        self.latest_trades[symbol] = trade

        # Add to history
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []

        self.trade_history[symbol].append(trade)

        # Trim history
        if len(self.trade_history[symbol]) > self.max_history:
            self.trade_history[symbol] = self.trade_history[symbol][-self.max_history:]

    def update_quote(self, quote: MarketQuote):
        """Update latest quote."""
        self.latest_quotes[quote.symbol] = quote

    def update_bar(self, bar: MarketBar):
        """Update latest bar."""
        self.latest_bars[bar.symbol] = bar

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from trades or quotes."""
        # Prefer trades over quotes
        if symbol in self.latest_trades:
            trade = self.latest_trades[symbol]
            if datetime.utcnow() - trade.timestamp < self.max_age:
                return trade.price

        # Fall back to quote mid-price
        if symbol in self.latest_quotes:
            quote = self.latest_quotes[symbol]
            if datetime.utcnow() - quote.timestamp < self.max_age:
                return (quote.bid_price + quote.ask_price) / 2

        return None

    def get_latest_trade(self, symbol: str) -> Optional[MarketTrade]:
        """Get latest trade."""
        return self.latest_trades.get(symbol)

    def get_latest_quote(self, symbol: str) -> Optional[MarketQuote]:
        """Get latest quote."""
        return self.latest_quotes.get(symbol)

    def get_spread(self, symbol: str) -> Optional[float]:
        """Get bid-ask spread."""
        quote = self.latest_quotes.get(symbol)
        if quote:
            return quote.ask_price - quote.bid_price
        return None

    def clear_stale_data(self):
        """Remove data older than max_age."""
        now = datetime.utcnow()

        # Clear stale trades
        for symbol in list(self.latest_trades.keys()):
            if now - self.latest_trades[symbol].timestamp > self.max_age:
                del self.latest_trades[symbol]

        # Clear stale quotes
        for symbol in list(self.latest_quotes.keys()):
            if now - self.latest_quotes[symbol].timestamp > self.max_age:
                del self.latest_quotes[symbol]
```

### Update: `storage/schema.sql`

```sql
-- Market Data Tables

-- Real-time trades
CREATE TABLE IF NOT EXISTS market_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    price REAL NOT NULL,
    size INTEGER NOT NULL,
    exchange TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON market_trades(symbol, timestamp DESC);

-- Real-time quotes (sampled every minute)
CREATE TABLE IF NOT EXISTS market_quotes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    bid_price REAL NOT NULL,
    bid_size INTEGER NOT NULL,
    ask_price REAL NOT NULL,
    ask_size INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_quotes_symbol_time ON market_quotes(symbol, timestamp DESC);

-- 1-minute bars
CREATE TABLE IF NOT EXISTS market_bars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_bars_symbol_time ON market_bars(symbol, timestamp DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_bars_unique ON market_bars(symbol, timestamp);
```

### File: `services/market_data_store.py`

```python
from storage.database import db
from models.market_data import MarketTrade, MarketQuote, MarketBar
from sqlalchemy import text
from utils.logger import get_logger

class MarketDataStore:
    """Persist market data to database."""

    def __init__(self):
        self.logger = get_logger("market_store")

    def store_trade(self, trade: MarketTrade):
        """Store trade in database."""
        try:
            with db.get_session() as session:
                session.execute(
                    text("""
                        INSERT INTO market_trades (symbol, timestamp, price, size, exchange)
                        VALUES (:symbol, :timestamp, :price, :size, :exchange)
                    """),
                    {
                        "symbol": trade.symbol,
                        "timestamp": trade.timestamp,
                        "price": trade.price,
                        "size": trade.size,
                        "exchange": trade.exchange
                    }
                )
        except Exception as e:
            self.logger.error("failed_to_store_trade", error=str(e))

    def store_bar(self, bar: MarketBar):
        """Store bar in database (deduplicated)."""
        try:
            with db.get_session() as session:
                session.execute(
                    text("""
                        INSERT OR REPLACE INTO market_bars
                        (symbol, timestamp, open, high, low, close, volume)
                        VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)
                    """),
                    {
                        "symbol": bar.symbol,
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume
                    }
                )
        except Exception as e:
            self.logger.error("failed_to_store_bar", error=str(e))

    def get_recent_bars(self, symbol: str, limit: int = 100) -> list[MarketBar]:
        """Get recent bars for a symbol."""
        try:
            with db.get_session() as session:
                result = session.execute(
                    text("""
                        SELECT symbol, timestamp, open, high, low, close, volume
                        FROM market_bars
                        WHERE symbol = :symbol
                        ORDER BY timestamp DESC
                        LIMIT :limit
                    """),
                    {"symbol": symbol, "limit": limit}
                )

                bars = []
                for row in result:
                    bars.append(MarketBar(
                        symbol=row.symbol,
                        timestamp=row.timestamp,
                        open=row.open,
                        high=row.high,
                        low=row.low,
                        close=row.close,
                        volume=row.volume
                    ))
                return bars
        except Exception as e:
            self.logger.error("failed_to_get_bars", error=str(e))
            return []
```

## Acceptance Criteria

1. ✅ Cache stores latest trade, quote, bar for each symbol
2. ✅ Cache provides fast lookup (<1ms)
3. ✅ Stale data automatically cleared
4. ✅ Database stores trades and bars
5. ✅ Can retrieve historical data
6. ✅ No memory leaks (history trimmed)
7. ✅ Deduplication on bars (same timestamp)

## Testing

### File: `tests/test_market_data_cache.py`

```python
import pytest
from datetime import datetime, timedelta
from services.market_data_cache import MarketDataCache
from models.market_data import MarketTrade, MarketQuote

def test_cache_latest_price():
    """Should return latest price from cache."""
    cache = MarketDataCache(max_age_minutes=5)

    trade = MarketTrade(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        price=150.50,
        size=100,
        exchange="IEX"
    )

    cache.update_trade(trade)

    price = cache.get_latest_price("AAPL")
    assert price == 150.50

def test_cache_stale_data():
    """Should reject stale data."""
    cache = MarketDataCache(max_age_minutes=5)

    # Old trade
    old_trade = MarketTrade(
        symbol="AAPL",
        timestamp=datetime.utcnow() - timedelta(minutes=10),
        price=150.50,
        size=100,
        exchange="IEX"
    )

    cache.update_trade(old_trade)
    price = cache.get_latest_price("AAPL")

    assert price is None  # Too old

def test_cache_history_trimming():
    """Should trim history to prevent memory leaks."""
    cache = MarketDataCache()
    cache.max_history = 10

    # Add 20 trades
    for i in range(20):
        trade = MarketTrade(
            symbol="AAPL",
            timestamp=datetime.utcnow(),
            price=150.0 + i,
            size=100,
            exchange="IEX"
        )
        cache.update_trade(trade)

    # Should only keep last 10
    assert len(cache.trade_history["AAPL"]) == 10
```

## Manual Testing

```bash
# Test cache performance
python -c "
from services.market_data_cache import MarketDataCache
from models.market_data import MarketTrade
from datetime import datetime
import time

cache = MarketDataCache()

# Add trade
trade = MarketTrade(
    symbol='AAPL',
    timestamp=datetime.utcnow(),
    price=150.50,
    size=100,
    exchange='IEX'
)

cache.update_trade(trade)

# Benchmark lookup
start = time.time()
for _ in range(10000):
    price = cache.get_latest_price('AAPL')
elapsed = time.time() - start

print(f'10,000 lookups in {elapsed:.4f}s = {elapsed/10000*1000:.4f}ms per lookup')
print(f'Latest price: ${price}')
"
```

## Estimated Completion Time

3 hours

## Notes

- Cache must be thread-safe if used from multiple async tasks
- Consider using `asyncio.Lock` for concurrent access
- Database writes should be batched for performance
- Quote sampling: Store 1 quote per minute (not every update)

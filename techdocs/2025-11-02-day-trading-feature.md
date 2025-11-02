# Day Trading Feature - Technical Design

**Created**: 2025-11-02
**Status**: PLANNING
**Priority**: HIGH

## Overview

Build an automated day trading system that makes up to 5 trading decisions per day based on real-time news analysis and live market data from Alpaca's streaming API, focusing on AAPL stock.

## Goals

1. **Intraday Trading**: Execute multiple trades throughout the trading day (9:30 AM - 4:00 PM ET)
2. **Real-time Data**: Use Alpaca WebSocket streaming for live market data
3. **News-Driven**: Trigger analysis when new AAPL news arrives
4. **Risk Management**: Limit to 5 trades per day, manage position sizes
5. **Live Decision Making**: Make decisions based on current market price, not historical data

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Day Trading System                      │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐         ┌─────▼─────┐      ┌──────▼──────┐
   │ Market  │         │   News    │      │   Trading   │
   │ Stream  │         │  Monitor  │      │  Scheduler  │
   └────┬────┘         └─────┬─────┘      └──────┬──────┘
        │                    │                    │
        │              ┌─────▼─────┐              │
        │              │ Triggered │              │
        │              │ Analysis  │              │
        │              └─────┬─────┘              │
        │                    │                    │
        └────────────┬───────┴───────┬────────────┘
                     │               │
              ┌──────▼───────────────▼──────┐
              │  Real-time Decision Engine  │
              │  • Live price                │
              │  • Current position          │
              │  • Sentiment score           │
              │  • Daily trade count         │
              └──────┬───────────────────────┘
                     │
              ┌──────▼──────┐
              │  Execution  │
              │   Agent     │
              └─────────────┘
```

## Key Components

### 1. Market Data Streamer (New)
- **Purpose**: Stream live market data via Alpaca WebSocket
- **Data Types**:
  - Trades: Real-time trade executions
  - Quotes: Best bid/ask prices
  - Bars: Minute-level OHLCV data
- **Storage**: In-memory cache + SQLite for historical reference

### 2. News Monitor (New)
- **Purpose**: Poll for new AAPL news every 15 minutes
- **Trigger**: When new articles appear, trigger analysis
- **Rate Limiting**: Respect API limits, batch analysis

### 3. Trading Scheduler (New)
- **Purpose**: Manage trading windows and enforce limits
- **Responsibilities**:
  - Track market hours (9:30 AM - 4:00 PM ET)
  - Enforce 5 trades per day limit
  - Prevent trading in first/last 15 minutes (market volatility)
  - Track daily P&L and position

### 4. Intraday Decision Agent (Updated)
- **Changes from Current**:
  - Use live market price instead of expected price
  - Consider current position (can't buy if fully invested)
  - Factor in daily trade count (more conservative if limit approaching)
  - Real-time sentiment calculation (last 2 hours, not 72)

### 5. Position Manager (New)
- **Purpose**: Track and manage positions
- **Responsibilities**:
  - Current position size and avg entry price
  - Unrealized P&L
  - Position limits (max exposure per symbol)
  - Exit conditions (stop loss, take profit)

## Trading Strategy

### Entry Conditions (BUY)
- Positive sentiment score > 0.4 (last 2 hours of news)
- At least 2 high-impact positive articles
- Not at max position (limit: 100 shares AAPL)
- Trades remaining today > 0
- Not in first/last 15 min of trading day

### Exit Conditions (SELL)
- Negative sentiment score < -0.3 (last 2 hours)
- OR Stop loss hit (-2% from avg entry)
- OR Take profit hit (+3% from avg entry)
- OR End of day liquidation (3:45 PM)

### Hold Conditions
- Neutral sentiment (-0.3 to 0.4)
- No position to exit
- Daily trade limit reached

## Risk Management

1. **Position Limits**
   - Max position: 100 shares AAPL
   - Max position value: 30% of buying power
   - Trade size: 20-30 shares per trade

2. **Daily Limits**
   - Max 5 trades per day
   - Max loss per day: $500 (circuit breaker)
   - Max position changes: 3 (prevent overtrading)

3. **Time-Based**
   - No trading first 15 minutes (9:30-9:45 AM)
   - No trading last 15 minutes (3:45-4:00 PM)
   - Force close all positions by 3:45 PM

## Data Models

### MarketData (New)
```python
class MarketData(BaseModel):
    symbol: str
    timestamp: datetime
    price: float
    bid: float
    ask: float
    volume: int
    source: str  # "trade", "quote", "bar"
```

### PositionSnapshot (New)
```python
class PositionSnapshot(BaseModel):
    symbol: str
    quantity: int
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    timestamp: datetime
```

### TradingSession (New)
```python
class TradingSession(BaseModel):
    date: date
    symbol: str
    trades_count: int
    max_trades: int = 5
    realized_pnl: float
    max_loss: float = -500.0
    is_active: bool
```

## Implementation Tickets

We'll break this into 8 manageable tickets:

1. **Market Data Streaming Service** - Alpaca WebSocket integration
2. **Market Data Storage** - Cache and persist streaming data
3. **Trading Session Manager** - Track daily trading limits
4. **Position Manager** - Track positions and P&L
5. **News Monitor Service** - Poll and detect new articles
6. **Intraday Decision Logic** - Update decision agent for real-time
7. **Day Trading Orchestrator** - Coordinate all components
8. **Testing & Monitoring** - Test with paper trading, add dashboards

## Testing Strategy

### Unit Tests
- Each component isolated
- Mock WebSocket streams
- Verify position calculations
- Test risk management rules

### Integration Tests
- End-to-end with paper trading
- Simulate news arrivals
- Test circuit breakers
- Verify daily limits

### Manual Testing
- Run during market hours for 1-2 days
- Monitor execution quality
- Verify position tracking
- Check P&L accuracy

## Success Criteria

1. ✅ System makes 5 trading decisions per day
2. ✅ Uses live market data (not stale prices)
3. ✅ Responds to news within 2 minutes of arrival
4. ✅ Respects all risk limits (position, daily loss, trade count)
5. ✅ Accurate position tracking and P&L calculation
6. ✅ No trading violations (first/last 15 min)
7. ✅ Clean shutdown at market close with flat positions

## Rollout Plan

**Phase 1** (Tickets 1-2): Market data infrastructure
**Phase 2** (Tickets 3-4): Risk management layer
**Phase 3** (Tickets 5-6): News-driven trading logic
**Phase 4** (Tickets 7-8): Orchestration and testing

## Dependencies

- Alpaca WebSocket API access
- Real-time news feed (already available)
- Paper trading account (already configured)
- Market hours awareness (need timezone handling)

## Future Enhancements

- Multiple symbols (AAPL, MSFT, GOOGL, etc.)
- Machine learning for better entry/exit timing
- Options trading for hedging
- Backtesting engine with historical data
- Web dashboard for monitoring

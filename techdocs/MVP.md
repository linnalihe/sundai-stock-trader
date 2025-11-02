# Multi-Agent AI Trading Platform - Technical Document

## Overview
A multi-agent AI system that analyzes market news and makes automated trading decisions for stocks (starting with AAPL) via the Alpaca API. Built for extensibility to support multiple assets and trading strategies.

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Orchestrator Agent                       │
│                  (Coordinates all agents)                    │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
       ┌───────▼────────┐            ┌────────▼────────┐
       │  News Agent    │            │ Analysis Agent  │
       │ (Fetch & Filter)│            │(Sentiment, Tech)│
       └───────┬────────┘            └────────┬────────┘
               │                              │
               └──────────┬───────────────────┘
                          │
                  ┌───────▼────────┐
                  │ Decision Agent │
                  │  (Buy/Sell/Hold)│
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │ Execution Agent│
                  │(Alpaca Trading) │
                  └────────────────┘
```

### Agent Responsibilities

#### 1. News Agent
- **Purpose**: Fetch and filter market news
- **Data Sources**:
  - NewsAPI / Alpaca News API
  - Financial news RSS feeds
- **Output**: Filtered, relevant news articles with metadata
- **Extensibility**: Easy to add new news sources via adapter pattern

#### 2. Analysis Agent
- **Purpose**: Analyze news and market data
- **Functions**:
  - Sentiment analysis (LLM-based)
  - Technical indicator calculation (optional for MVP)
  - Context summarization
- **Output**: Analysis report with sentiment scores and key insights
- **Extensibility**: Pluggable analysis strategies

#### 3. Decision Agent
- **Purpose**: Make trading decisions based on analysis
- **Logic**:
  - Evaluate sentiment and market conditions
  - Apply risk management rules
  - Generate trading signals (BUY/SELL/HOLD)
  - Calculate position sizes
- **Output**: Trading decision with confidence level
- **Extensibility**: Rule-based initially, ML model integration later

#### 4. Execution Agent
- **Purpose**: Execute trades via Alpaca API
- **Functions**:
  - Order placement
  - Position monitoring
  - Error handling and retries
- **Output**: Trade confirmation and status
- **Extensibility**: Broker abstraction layer for multi-exchange support

#### 5. Orchestrator Agent
- **Purpose**: Coordinate agent workflow
- **Functions**:
  - Schedule news fetching
  - Trigger analysis pipeline
  - Log decisions and executions
  - Handle errors and state management
- **Output**: System logs and orchestration state

## Technology Stack

### Core Technologies
- **Language**: Python 3.11+
- **Agent Framework**: LangGraph or CrewAI (for agent orchestration)
- **LLM**: OpenAI GPT-4 or Anthropic Claude (for analysis)
- **Trading API**: Alpaca Markets API
- **News API**: NewsAPI.org or Alpaca News API

### Supporting Libraries
- **Data**: pandas, numpy
- **HTTP**: httpx (async requests)
- **Environment**: python-dotenv
- **Logging**: structlog
- **Testing**: pytest
- **Database**: SQLite (MVP) → PostgreSQL (production)
- **Task Queue**: None (MVP) → Celery/Redis (future)

## Data Models

### News Article
```python
{
    "id": str,
    "title": str,
    "content": str,
    "source": str,
    "published_at": datetime,
    "url": str,
    "symbols": List[str],  # e.g., ["AAPL"]
    "relevance_score": float
}
```

### Analysis Result
```python
{
    "article_id": str,
    "symbol": str,
    "sentiment": str,  # POSITIVE, NEGATIVE, NEUTRAL
    "sentiment_score": float,  # -1.0 to 1.0
    "key_points": List[str],
    "impact_level": str,  # HIGH, MEDIUM, LOW
    "analyzed_at": datetime
}
```

### Trading Decision
```python
{
    "symbol": str,
    "action": str,  # BUY, SELL, HOLD
    "quantity": int,
    "confidence": float,  # 0.0 to 1.0
    "reasoning": str,
    "price_limit": Optional[float],
    "created_at": datetime
}
```

### Trade Execution
```python
{
    "decision_id": str,
    "order_id": str,
    "symbol": str,
    "status": str,  # SUBMITTED, FILLED, REJECTED, CANCELLED
    "filled_qty": int,
    "filled_avg_price": float,
    "executed_at": datetime
}
```

## Project Structure

```
sundai-stock-trader/
├── README.md
├── TECH_DOC.md
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
│
├── main.py                    # Entry point for the system
├── config/
│   ├── __init__.py
│   ├── settings.py           # Configuration management
│   └── symbols.yaml          # Tradeable symbols config
│
├── agents/
│   ├── __init__.py
│   ├── base.py               # Base agent class
│   ├── orchestrator.py       # Main orchestrator
│   ├── news_agent.py         # News fetching
│   ├── analysis_agent.py     # Analysis & sentiment
│   ├── decision_agent.py     # Trading decisions
│   └── execution_agent.py    # Trade execution
│
├── services/
│   ├── __init__.py
│   ├── news_service.py       # News API integration
│   ├── alpaca_service.py     # Alpaca API wrapper
│   └── llm_service.py        # LLM API wrapper
│
├── models/
│   ├── __init__.py
│   ├── news.py               # News data models
│   ├── analysis.py           # Analysis models
│   └── trade.py              # Trading models
│
├── utils/
│   ├── __init__.py
│   ├── logger.py             # Logging setup
│   └── validators.py         # Input validation
│
├── storage/
│   ├── __init__.py
│   ├── database.py           # Database interface
│   └── schema.sql            # Database schema
│
└── tests/
    ├── __init__.py
    ├── test_agents/
    ├── test_services/
    └── fixtures/
```

## Development Phases

### Phase 1: Foundation (Week 1)
**Goal**: Set up core infrastructure and basic agent framework

**Tasks**:
1. Set up project structure
2. Configure environment and dependencies
3. Implement base agent class
4. Set up logging and error handling
5. Create data models (Pydantic)
6. Set up SQLite database with basic schema

**Deliverable**: Running framework with skeleton agents

### Phase 2: News Collection (Week 2)
**Goal**: Implement news fetching and storage

**Tasks**:
1. Integrate NewsAPI or Alpaca News API
2. Implement News Agent
3. Add news filtering logic (by symbol, relevance)
4. Store news articles in database
5. Create news retrieval utilities

**Deliverable**: News agent fetching and storing AAPL news

### Phase 3: Analysis Pipeline (Week 2-3)
**Goal**: Implement sentiment and news analysis

**Tasks**:
1. Integrate LLM service (OpenAI/Anthropic)
2. Implement Analysis Agent
3. Create sentiment analysis prompts
4. Extract key information from news
5. Score and rank news by impact

**Deliverable**: Analysis agent producing sentiment reports

### Phase 4: Decision Logic (Week 3-4)
**Goal**: Build trading decision engine

**Tasks**:
1. Implement Decision Agent
2. Create rule-based decision logic
3. Add risk management parameters
4. Implement position sizing logic
5. Add confidence scoring

**Deliverable**: Decision agent generating trading signals

### Phase 5: Trade Execution (Week 4)
**Goal**: Execute trades via Alpaca

**Tasks**:
1. Enhance Alpaca service wrapper
2. Implement Execution Agent
3. Add order placement logic
4. Implement order status tracking
5. Add error handling and retry logic

**Deliverable**: End-to-end trading from news to execution

### Phase 6: Orchestration (Week 5)
**Goal**: Coordinate all agents into unified workflow

**Tasks**:
1. Implement Orchestrator Agent
2. Create workflow scheduling
3. Add state management
4. Implement event logging
5. Create dashboard/monitoring

**Deliverable**: Fully automated trading system

### Phase 7: Testing & Refinement (Week 6)
**Goal**: Test, debug, and optimize

**Tasks**:
1. Write unit tests for all agents
2. Integration testing
3. Paper trading validation
4. Performance optimization
5. Documentation updates

**Deliverable**: Production-ready MVP

## Extensibility Design

### 1. Multi-Asset Support
**Current**: Hardcoded for AAPL
**Future**:
```python
# config/symbols.yaml
assets:
  stocks:
    - symbol: AAPL
      enabled: true
    - symbol: GOOGL
      enabled: true
  crypto:
    - symbol: BTCUSD
      enabled: false  # Future
```

### 2. Multi-Strategy Support
**Current**: Single sentiment-based strategy
**Future**:
```python
# Abstract strategy pattern
class TradingStrategy:
    def analyze(self, data) -> Decision:
        pass

class SentimentStrategy(TradingStrategy):
    pass

class TechnicalStrategy(TradingStrategy):
    pass

class HybridStrategy(TradingStrategy):
    pass
```

### 3. Multi-Broker Support
**Current**: Alpaca only
**Future**:
```python
# Broker abstraction
class BrokerInterface:
    def place_order(self, order) -> OrderResult:
        pass

class AlpacaBroker(BrokerInterface):
    pass

class InteractiveBrokersBroker(BrokerInterface):
    pass
```

### 4. Plugin Architecture
```python
# Future: Agent plugins
class AgentPlugin:
    def register(self, orchestrator):
        pass

    def execute(self, context):
        pass
```

## Configuration Management

### Environment Variables
```bash
# .env
ALPACA_MARKET_API_KEY=your_key
ALPACA_MARKET_API_SECRET=your_secret
OPENAI_API_KEY=your_key
NEWS_API_KEY=your_key

# Trading Config
PAPER_TRADING=true
MAX_POSITION_SIZE=100
RISK_PERCENTAGE=0.02

# Agent Config
NEWS_FETCH_INTERVAL=300  # 5 minutes
ANALYSIS_THRESHOLD=0.6
```

### Config File
```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    alpaca_api_key: str
    alpaca_api_secret: str
    openai_api_key: str
    news_api_key: str

    paper_trading: bool = True
    max_position_size: int = 100
    risk_percentage: float = 0.02

    news_fetch_interval: int = 300
    analysis_threshold: float = 0.6

    class Config:
        env_file = ".env"
```

## Risk Management

### MVP Risk Controls
1. **Paper Trading Only**: All trades in simulation mode
2. **Position Limits**: Max 100 shares per trade
3. **Portfolio Limit**: Max 20% portfolio in single position
4. **Confidence Threshold**: Only execute high-confidence decisions (>0.7)
5. **Daily Trade Limit**: Max 5 trades per day
6. **Loss Prevention**: Stop trading if daily loss exceeds 2%

### Future Enhancements
- Stop-loss orders
- Take-profit targets
- Portfolio rebalancing
- Volatility-based position sizing
- Multi-timeframe analysis

## Monitoring & Logging

### What to Log
1. **News Events**: All fetched articles with timestamps
2. **Analysis Results**: Sentiment scores and reasoning
3. **Decisions**: All trading decisions with confidence
4. **Executions**: Order status and fill prices
5. **Errors**: All exceptions and failures
6. **Performance**: P&L, win rate, Sharpe ratio

### Log Format
```python
# Use structured logging
{
    "timestamp": "2025-11-02T10:30:00Z",
    "level": "INFO",
    "agent": "decision_agent",
    "event": "decision_made",
    "symbol": "AAPL",
    "action": "BUY",
    "confidence": 0.85,
    "reasoning": "Strong positive sentiment from earnings report"
}
```

## Testing Strategy

### Unit Tests
- Test each agent independently
- Mock external API calls
- Test edge cases and error handling

### Integration Tests
- Test agent communication
- Test end-to-end workflows
- Test database operations

### Validation Tests
- Backtest on historical news data
- Compare against buy-and-hold strategy
- Validate risk controls

## Deployment (Future)

### MVP: Local Execution
- Run as Python script on local machine
- Cron job for scheduling

### Future: Cloud Deployment
- Containerize with Docker
- Deploy to AWS/GCP
- Use managed task scheduler
- Set up monitoring dashboards

## Success Metrics

### MVP Goals
1. Successfully fetch AAPL news every 5 minutes
2. Generate sentiment analysis for each article
3. Make trading decisions based on sentiment
4. Execute at least 1 paper trade per day
5. Maintain 100% uptime during market hours
6. Log all activities for review

### Performance Metrics (Future)
- Win rate > 55%
- Sharpe ratio > 1.0
- Max drawdown < 10%
- Average holding period

## Next Steps

1. **Review this document** and validate architecture
2. **Set up development environment** with all dependencies
3. **Start Phase 1**: Build foundation and base classes
4. **Iterate quickly**: Build, test, refine each phase
5. **Paper trade extensively** before any real capital

## Questions to Consider

1. Which LLM provider? (OpenAI GPT-4 vs Anthropic Claude)
2. News API preference? (NewsAPI vs Alpaca News vs others)
3. Agent framework? (LangGraph vs CrewAI vs custom)
4. Database for production? (PostgreSQL vs MongoDB)
5. Monitoring tools? (Grafana, Prometheus, or simple logging)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
**Author**: Technical Architecture Team
**Status**: Draft - Ready for Review

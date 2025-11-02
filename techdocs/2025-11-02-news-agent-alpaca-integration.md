# Feature: News Agent - Alpaca News API Integration

**Date**: 2025-11-02
**Status**: ✅ COMPLETED
**Estimated Time**: 2-3 hours
**Actual Time**: ~2.5 hours

## Overview
Build the News Agent to fetch financial news from Alpaca News API, filter articles relevant to AAPL stock, calculate relevance scores, and store them in the database. This is the first data collection component of the trading pipeline.

## What We're Building
1. News service wrapper for Alpaca News API
2. News Agent that orchestrates fetching and storage
3. Relevance scoring logic (keyword-based for MVP)
4. Duplicate detection to avoid storing same article twice
5. Symbol configuration file for managing tradeable assets
6. Database storage and retrieval functions

## Implementation Details

### 1. Symbol Configuration (`config/symbols.yaml`)
Create a YAML file to manage which stocks to track:

```yaml
# Trading Symbols Configuration
assets:
  stocks:
    - symbol: AAPL
      name: Apple Inc.
      enabled: true
      max_position_size: 100

    - symbol: GOOGL
      name: Alphabet Inc.
      enabled: false
      max_position_size: 50

  crypto:
    - symbol: BTCUSD
      name: Bitcoin
      enabled: false
      max_position_size: 0.1
```

**Create `config/symbol_config.py`**:
- `SymbolConfig` class to load and parse YAML
- `get_enabled_symbols(asset_type)` method
- `get_symbol_info(symbol)` method
- Global instance for easy importing

### 2. Alpaca News Service (`services/alpaca_news_service.py`)
Create a service to interact with Alpaca News API:

**Key Methods**:
- `__init__(api_key, api_secret)` - Initialize Alpaca NewsClient
- `fetch_news(symbols, hours_back=24, limit=50)` - Fetch news for symbols
- `_parse_news_item(item, symbol)` - Convert Alpaca news to NewsArticle model

**Implementation Notes**:
- Use `alpaca.data.historical.NewsClient`
- Use `alpaca.data.requests.NewsRequest` for queries
- Handle pagination if needed
- Parse response into our NewsArticle model
- Handle API errors gracefully (rate limits, connection errors)

### 3. News Agent (`agents/news_agent.py`)
Create the agent that orchestrates news collection:

**Inherits from**: `BaseAgent`

**Key Methods**:
- `execute(symbols, hours_back=24, max_articles=50)` - Main entry point
- `_filter_duplicates(articles)` - Remove duplicates by URL
- `_score_relevance(articles, symbols)` - Calculate relevance scores
- `_store_articles(articles)` - Save to database
- `get_recent_articles(symbol, limit=10, min_relevance=0.0)` - Retrieve from DB

**Relevance Scoring Logic** (Simple keyword-based for MVP):
- Symbol in title: +0.5 points
- Company name in title: +0.3 points
- Financial keywords in title: +0.1 each ("earnings", "revenue", "profit", "stock", "shares")
- Same keywords in content: +0.05 each
- Normalize to 0.0-1.0 range

**Duplicate Detection**:
- Check if URL already exists in database
- Use URL as unique identifier

### 4. Database Operations
**Storage** (`_store_articles` method):
- Check if article URL already exists (skip if duplicate)
- Insert new articles with all fields
- Store symbols as JSON array
- Handle database errors

**Retrieval** (`get_recent_articles` method):
- Query by symbol (using LIKE on JSON array)
- Filter by minimum relevance score
- Order by published_at DESC
- Limit results

### 5. Update main.py
Add function to run news agent:

```python
async def run_news_agent():
    """Run news agent to fetch latest articles."""
    # Initialize service and agent
    # Get enabled symbols from config
    # Fetch news
    # Display summary
```

## Success Criteria

### Functional Requirements
- [ ] Symbol configuration loads from YAML file
- [ ] Can get enabled symbols (should return ["AAPL"])
- [ ] Alpaca News Service connects to API successfully
- [ ] Can fetch news for AAPL symbol
- [ ] News articles parsed into NewsArticle model correctly
- [ ] Relevance scores calculated (should be between 0.0-1.0)
- [ ] Duplicate articles filtered out (same URL not stored twice)
- [ ] Articles stored in database with all fields
- [ ] Can retrieve articles from database by symbol
- [ ] News agent executes end-to-end without errors

### Data Quality
- [ ] All required NewsArticle fields populated
- [ ] Timestamps in correct format (datetime objects)
- [ ] URLs are valid HTTP/HTTPS
- [ ] Symbols array contains correct stock symbols
- [ ] Relevance scores make sense (AAPL news has higher score)

### Error Handling
- [ ] Handles API connection errors gracefully
- [ ] Handles missing/invalid data in API response
- [ ] Handles database errors (duplicate keys, etc.)
- [ ] Logs errors with context

## Tests

### Test File: `tests/test_symbol_config.py`
```python
def test_load_symbols_config():
    """Should load symbols from YAML file"""
    from config.symbol_config import symbol_config
    enabled = symbol_config.get_enabled_symbols("stocks")
    assert "AAPL" in enabled

def test_get_symbol_info():
    """Should get info for specific symbol"""
    from config.symbol_config import symbol_config
    info = symbol_config.get_symbol_info("AAPL")
    assert info["name"] == "Apple Inc."
    assert info["enabled"] == True

def test_disabled_symbols_not_included():
    """Disabled symbols should not be in enabled list"""
    from config.symbol_config import symbol_config
    enabled = symbol_config.get_enabled_symbols("stocks")
    assert "GOOGL" not in enabled  # Disabled in config
```

### Test File: `tests/test_alpaca_news_service.py`
```python
import pytest
from services.alpaca_news_service import AlpacaNewsService
from config.settings import settings

@pytest.mark.asyncio
async def test_alpaca_news_service_init():
    """Should initialize Alpaca News Service"""
    service = AlpacaNewsService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret
    )
    assert service.client is not None

@pytest.mark.asyncio
async def test_fetch_news_for_aapl():
    """Should fetch news for AAPL"""
    service = AlpacaNewsService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret
    )
    articles = await service.fetch_news(["AAPL"], hours_back=24, limit=10)

    assert isinstance(articles, list)
    # May be 0 if no news in last 24 hours, but should not error
    if len(articles) > 0:
        assert articles[0].symbols[0] == "AAPL"
        assert articles[0].url is not None

@pytest.mark.asyncio
async def test_news_article_structure():
    """Fetched news should have correct structure"""
    service = AlpacaNewsService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret
    )
    articles = await service.fetch_news(["AAPL"], hours_back=72, limit=5)

    if len(articles) > 0:
        article = articles[0]
        assert article.id is not None
        assert article.title is not None
        assert article.source is not None
        assert article.published_at is not None
```

### Test File: `tests/test_news_agent.py`
```python
import pytest
from agents.news_agent import NewsAgent
from services.alpaca_news_service import AlpacaNewsService
from models.news import NewsArticle
from datetime import datetime
from config.settings import settings

@pytest.fixture
def news_service():
    return AlpacaNewsService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret
    )

@pytest.fixture
def news_agent(news_service):
    return NewsAgent(news_service)

def test_news_agent_initialization(news_agent):
    """News agent should initialize correctly"""
    assert news_agent.name == "news_agent"
    assert news_agent.news_service is not None

def test_duplicate_filtering(news_agent):
    """Should filter duplicate articles by URL"""
    articles = [
        NewsArticle(
            id="1", title="Test", content="Content", source="Source",
            published_at=datetime.utcnow(), url="https://example.com/1",
            symbols=["AAPL"]
        ),
        NewsArticle(
            id="2", title="Test2", content="Content", source="Source",
            published_at=datetime.utcnow(), url="https://example.com/1",  # Duplicate
            symbols=["AAPL"]
        ),
        NewsArticle(
            id="3", title="Test3", content="Content", source="Source",
            published_at=datetime.utcnow(), url="https://example.com/3",
            symbols=["AAPL"]
        ),
    ]

    unique = news_agent._filter_duplicates(articles)
    assert len(unique) == 2  # Only 2 unique URLs

def test_relevance_scoring(news_agent):
    """Should score articles based on keywords"""
    articles = [
        NewsArticle(
            id="1",
            title="AAPL reports strong earnings revenue profit",
            content="Apple Inc. showed strong performance",
            source="Source",
            published_at=datetime.utcnow(),
            url="https://example.com/1",
            symbols=["AAPL"]
        ),
        NewsArticle(
            id="2",
            title="Random tech news",
            content="Some random content",
            source="Source",
            published_at=datetime.utcnow(),
            url="https://example.com/2",
            symbols=["AAPL"]
        ),
    ]

    scored = news_agent._score_relevance(articles, ["AAPL"])

    # First article should have higher relevance (has keywords)
    assert scored[0].relevance_score > scored[1].relevance_score
    assert 0.0 <= scored[0].relevance_score <= 1.0
    assert 0.0 <= scored[1].relevance_score <= 1.0

@pytest.mark.asyncio
async def test_news_agent_execute(news_agent):
    """Should execute full news fetching pipeline"""
    articles = await news_agent.execute(
        symbols=["AAPL"],
        hours_back=72,
        max_articles=10
    )

    assert isinstance(articles, list)
    # Should not error even if no articles found

@pytest.mark.asyncio
async def test_store_and_retrieve(news_agent):
    """Should store articles and retrieve them"""
    # Fetch and store
    await news_agent.execute(["AAPL"], hours_back=72, max_articles=5)

    # Retrieve
    recent = news_agent.get_recent_articles("AAPL", limit=5)

    # Should have some articles (if API returned any)
    assert isinstance(recent, list)
```

### Integration Test
```python
@pytest.mark.asyncio
async def test_full_news_pipeline():
    """Test complete news pipeline end-to-end"""
    from config.symbol_config import symbol_config
    from services.alpaca_news_service import AlpacaNewsService
    from agents.news_agent import NewsAgent
    from config.settings import settings

    # 1. Get symbols from config
    symbols = symbol_config.get_enabled_symbols("stocks")
    assert len(symbols) > 0

    # 2. Initialize service and agent
    service = AlpacaNewsService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret
    )
    agent = NewsAgent(service)

    # 3. Fetch news
    articles = await agent.execute(symbols, hours_back=72, max_articles=20)

    # 4. Verify results
    assert isinstance(articles, list)

    # 5. Retrieve from database
    if len(articles) > 0:
        recent = agent.get_recent_articles(symbols[0], limit=5)
        assert len(recent) > 0
```

## Manual Verification Commands

```bash
# 1. Test symbol config
python -c "from config.symbol_config import symbol_config; print(symbol_config.get_enabled_symbols('stocks'))"

# 2. Test news service
python -c "
import asyncio
from services.alpaca_news_service import AlpacaNewsService
from config.settings import settings

async def test():
    service = AlpacaNewsService(settings.alpaca_api_key, settings.alpaca_api_secret)
    articles = await service.fetch_news(['AAPL'], hours_back=24, limit=5)
    print(f'Fetched {len(articles)} articles')
    if articles:
        print(f'First article: {articles[0].title}')

asyncio.run(test())
"

# 3. Run news agent via main.py
python main.py

# 4. Check database
sqlite3 trading.db "SELECT COUNT(*) FROM news_articles"
sqlite3 trading.db "SELECT title, relevance_score, source FROM news_articles WHERE symbols LIKE '%AAPL%' ORDER BY published_at DESC LIMIT 5"

# 5. Run tests
pytest tests/test_symbol_config.py -v
pytest tests/test_alpaca_news_service.py -v
pytest tests/test_news_agent.py -v
```

## Files to Create/Modify

### New Files:
- `config/symbols.yaml` - Symbol configuration
- `config/symbol_config.py` - Symbol config loader
- `services/alpaca_news_service.py` - Alpaca News API wrapper
- `agents/news_agent.py` - News fetching agent
- `tests/test_symbol_config.py` - Symbol config tests
- `tests/test_alpaca_news_service.py` - News service tests
- `tests/test_news_agent.py` - News agent tests

### Modified Files:
- `main.py` - Add news agent execution
- `requirements.txt` - Already has alpaca-py

## Dependencies
- ✅ Phase 1 completed (foundation)
- ✅ Alpaca API credentials in .env
- ✅ Database schema created
- ✅ NewsArticle model defined

## Blockers
None - all dependencies are met

## Notes
- Start simple with keyword-based relevance scoring
- Can enhance with ML-based relevance in future
- Alpaca News API has rate limits - handle gracefully
- Store raw news data - don't filter too aggressively
- Focus on AAPL only for MVP, but design for extensibility

## Implementation Order
1. Create `config/symbols.yaml` and `config/symbol_config.py`
2. Create `services/alpaca_news_service.py`
3. Create `agents/news_agent.py` with all methods
4. Update `main.py` to run news agent
5. Create all test files
6. Run tests and verify
7. Manual testing with real API

## Expected Output
After completion, running `python main.py` should:
- Connect to Alpaca
- Fetch news for AAPL
- Store articles in database
- Log summary (e.g., "Fetched 15 articles, stored 12 new ones")
- Display sample article titles with relevance scores

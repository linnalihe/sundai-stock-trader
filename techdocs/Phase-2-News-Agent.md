# Phase 2: News Agent

## Overview
Implement the News Agent to fetch, filter, and store market news relevant to configured stock symbols. This agent is the first data collection component of the trading pipeline.

## Timeline
**Estimated Duration**: 3-4 days

## Objectives
1. Integrate with news API (NewsAPI or Alpaca News)
2. Implement news fetching with symbol filtering
3. Add relevance scoring
4. Store news articles in database
5. Create news retrieval utilities
6. Schedule periodic news fetching

## Dependencies
- Phase 1 completed (foundation, models, database)
- News API key (NewsAPI.org or Alpaca News)
- Base agent class

## Implementation Tasks

### 1. News Service Integration
**File**: `services/news_service.py`

**Work**:
```python
import httpx
from datetime import datetime, timedelta
from typing import List, Optional
from models.news import NewsArticle
from config.settings import settings
from utils.logger import get_logger
import uuid

logger = get_logger("news_service")

class NewsService:
    """Service for fetching news from external APIs."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("news_service_initialized")

    async def fetch_news(
        self,
        symbols: List[str],
        hours_back: int = 24,
        max_articles: int = 50
    ) -> List[NewsArticle]:
        """
        Fetch news articles for given symbols.

        Args:
            symbols: List of stock symbols (e.g., ["AAPL"])
            hours_back: How many hours back to search
            max_articles: Maximum articles to return

        Returns:
            List of NewsArticle objects
        """
        articles = []

        for symbol in symbols:
            try:
                symbol_articles = await self._fetch_for_symbol(
                    symbol, hours_back, max_articles
                )
                articles.extend(symbol_articles)
                logger.info(
                    "news_fetched_for_symbol",
                    symbol=symbol,
                    count=len(symbol_articles)
                )
            except Exception as e:
                logger.error(
                    "news_fetch_failed",
                    symbol=symbol,
                    error=str(e)
                )

        return articles

    async def _fetch_for_symbol(
        self,
        symbol: str,
        hours_back: int,
        max_articles: int
    ) -> List[NewsArticle]:
        """Fetch news for a single symbol from NewsAPI."""

        # Calculate time range
        from_time = datetime.utcnow() - timedelta(hours=hours_back)

        # Build query
        query = f"{symbol} OR {self._get_company_name(symbol)}"

        params = {
            "q": query,
            "from": from_time.isoformat(),
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": min(max_articles, 100),
            "apiKey": self.api_key
        }

        response = await self.client.get(
            f"{self.base_url}/everything",
            params=params
        )
        response.raise_for_status()

        data = response.json()

        # Convert to NewsArticle objects
        articles = []
        for article in data.get("articles", []):
            try:
                news_article = NewsArticle(
                    id=f"news_{uuid.uuid4().hex[:12]}",
                    title=article["title"] or "No Title",
                    content=article["description"] or article.get("content", ""),
                    source=article["source"]["name"],
                    published_at=datetime.fromisoformat(
                        article["publishedAt"].replace("Z", "+00:00")
                    ),
                    url=article["url"],
                    symbols=[symbol],
                    relevance_score=0.0  # Will be calculated later
                )
                articles.append(news_article)
            except Exception as e:
                logger.warning(
                    "article_parsing_failed",
                    error=str(e),
                    article_title=article.get("title")
                )
                continue

        return articles

    def _get_company_name(self, symbol: str) -> str:
        """Get company name for symbol (can be expanded)."""
        company_map = {
            "AAPL": "Apple",
            "GOOGL": "Google",
            "MSFT": "Microsoft",
            "AMZN": "Amazon",
            "TSLA": "Tesla",
        }
        return company_map.get(symbol, symbol)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
```

**Alternative - Alpaca News Service**:
```python
# If using Alpaca News instead
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

class AlpacaNewsService:
    """Service for fetching news from Alpaca."""

    def __init__(self, api_key: str, api_secret: str):
        self.client = NewsClient(api_key, api_secret)
        logger.info("alpaca_news_service_initialized")

    async def fetch_news(
        self,
        symbols: List[str],
        hours_back: int = 24,
        max_articles: int = 50
    ) -> List[NewsArticle]:
        """Fetch news using Alpaca News API."""

        start_time = datetime.utcnow() - timedelta(hours=hours_back)

        request = NewsRequest(
            symbols=symbols,
            start=start_time,
            limit=max_articles
        )

        news = self.client.get_news(request)

        articles = []
        for item in news:
            article = NewsArticle(
                id=f"news_{item.id}",
                title=item.headline,
                content=item.summary,
                source=item.source,
                published_at=item.created_at,
                url=item.url,
                symbols=item.symbols,
                relevance_score=0.0
            )
            articles.append(article)

        return articles
```

**Success Criteria**:
- [ ] Can fetch news from API
- [ ] Articles parsed correctly into NewsArticle model
- [ ] Error handling for API failures
- [ ] Rate limiting respected
- [ ] Test with multiple symbols

---

### 2. News Agent Implementation
**File**: `agents/news_agent.py`

**Work**:
```python
from typing import List, Optional
from agents.base import BaseAgent
from services.news_service import NewsService
from models.news import NewsArticle
from storage.database import db
from config.settings import settings
from sqlalchemy import text
import json

class NewsAgent(BaseAgent):
    """Agent responsible for fetching and storing news articles."""

    def __init__(self, news_service: NewsService):
        super().__init__("news_agent")
        self.news_service = news_service

    async def execute(
        self,
        symbols: List[str],
        hours_back: int = 24,
        max_articles: int = 50
    ) -> List[NewsArticle]:
        """
        Fetch and store news articles.

        Args:
            symbols: Stock symbols to fetch news for
            hours_back: Time window for news
            max_articles: Maximum articles per symbol

        Returns:
            List of newly fetched articles
        """
        self._log_event(
            "news_fetch_started",
            symbols=symbols,
            hours_back=hours_back
        )

        # Fetch news
        articles = await self.news_service.fetch_news(
            symbols=symbols,
            hours_back=hours_back,
            max_articles=max_articles
        )

        # Filter duplicates
        unique_articles = self._filter_duplicates(articles)

        # Score relevance
        scored_articles = self._score_relevance(unique_articles, symbols)

        # Store in database
        stored_count = self._store_articles(scored_articles)

        self._log_event(
            "news_fetch_completed",
            total_fetched=len(articles),
            unique=len(unique_articles),
            stored=stored_count
        )

        return scored_articles

    def _filter_duplicates(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on URL or title."""
        seen_urls = set()
        unique = []

        for article in articles:
            if str(article.url) not in seen_urls:
                seen_urls.add(str(article.url))
                unique.append(article)

        return unique

    def _score_relevance(
        self,
        articles: List[NewsArticle],
        target_symbols: List[str]
    ) -> List[NewsArticle]:
        """
        Score article relevance based on content.

        Simple keyword-based scoring for MVP.
        Can be enhanced with ML in future.
        """
        for article in articles:
            score = 0.0

            # Check if symbols in title (highest weight)
            title_lower = article.title.lower()
            for symbol in target_symbols:
                if symbol.lower() in title_lower:
                    score += 0.5

            # Check content
            content_lower = article.content.lower()
            keywords = ["earnings", "revenue", "profit", "stock", "shares",
                       "acquisition", "merger", "ceo", "product launch"]

            for keyword in keywords:
                if keyword in title_lower:
                    score += 0.1
                elif keyword in content_lower:
                    score += 0.05

            # Normalize
            article.relevance_score = min(score, 1.0)

        return articles

    def _store_articles(self, articles: List[NewsArticle]) -> int:
        """Store articles in database."""
        stored = 0

        with db.get_session() as session:
            for article in articles:
                try:
                    # Check if already exists
                    result = session.execute(
                        text("SELECT id FROM news_articles WHERE url = :url"),
                        {"url": str(article.url)}
                    )

                    if result.fetchone():
                        continue

                    # Insert new article
                    session.execute(
                        text("""
                            INSERT INTO news_articles
                            (id, title, content, source, published_at, url, symbols, relevance_score)
                            VALUES (:id, :title, :content, :source, :published_at, :url, :symbols, :relevance_score)
                        """),
                        {
                            "id": article.id,
                            "title": article.title,
                            "content": article.content,
                            "source": article.source,
                            "published_at": article.published_at,
                            "url": str(article.url),
                            "symbols": json.dumps(article.symbols),
                            "relevance_score": article.relevance_score
                        }
                    )
                    stored += 1

                except Exception as e:
                    self.logger.error(
                        "article_storage_failed",
                        article_id=article.id,
                        error=str(e)
                    )

        return stored

    def get_recent_articles(
        self,
        symbol: str,
        limit: int = 10,
        min_relevance: float = 0.0
    ) -> List[NewsArticle]:
        """Retrieve recent articles from database."""

        with db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT * FROM news_articles
                    WHERE symbols LIKE :symbol
                    AND relevance_score >= :min_relevance
                    ORDER BY published_at DESC
                    LIMIT :limit
                """),
                {
                    "symbol": f"%{symbol}%",
                    "min_relevance": min_relevance,
                    "limit": limit
                }
            )

            rows = result.fetchall()

            articles = []
            for row in rows:
                article = NewsArticle(
                    id=row[0],
                    title=row[1],
                    content=row[2],
                    source=row[3],
                    published_at=row[4],
                    url=row[5],
                    symbols=json.loads(row[6]),
                    relevance_score=row[7]
                )
                articles.append(article)

            return articles
```

**Success Criteria**:
- [ ] Agent fetches news successfully
- [ ] Duplicate filtering works
- [ ] Relevance scoring assigns scores
- [ ] Articles stored in database
- [ ] Can retrieve articles from database
- [ ] Error handling for failed API calls

---

### 3. Configuration for Symbols
**File**: `config/symbols.yaml`

**Work**:
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

    - symbol: MSFT
      name: Microsoft Corporation
      enabled: false
      max_position_size: 75

  crypto:
    - symbol: BTCUSD
      name: Bitcoin
      enabled: false
      max_position_size: 0.1
```

**File**: `config/symbol_config.py`

**Work**:
```python
import yaml
from pathlib import Path
from typing import List, Dict
from utils.logger import get_logger

logger = get_logger("symbol_config")

class SymbolConfig:
    """Manage trading symbol configuration."""

    def __init__(self):
        self.config_path = Path(__file__).parent / "symbols.yaml"
        self._config = self._load_config()

    def _load_config(self) -> Dict:
        """Load symbols from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("symbol_config_loaded", path=str(self.config_path))
        return config

    def get_enabled_symbols(self, asset_type: str = "stocks") -> List[str]:
        """Get list of enabled symbols."""
        assets = self._config.get("assets", {}).get(asset_type, [])
        enabled = [
            asset["symbol"]
            for asset in assets
            if asset.get("enabled", False)
        ]
        return enabled

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get configuration for a specific symbol."""
        for asset_type in ["stocks", "crypto"]:
            assets = self._config.get("assets", {}).get(asset_type, [])
            for asset in assets:
                if asset["symbol"] == symbol:
                    return asset
        return {}

# Global instance
symbol_config = SymbolConfig()
```

**Success Criteria**:
- [ ] YAML file loads correctly
- [ ] Can get enabled symbols
- [ ] Can get symbol-specific config
- [ ] Easy to add new symbols

---

### 4. Update Main Entry Point
**File**: `main.py`

**Work**:
```python
import asyncio
from utils.logger import setup_logging, get_logger
from config.settings import settings
from config.symbol_config import symbol_config
from storage.database import db
from services.news_service import NewsService
from agents.news_agent import NewsAgent

setup_logging()
logger = get_logger("main")

async def run_news_agent():
    """Run news agent to fetch latest articles."""
    logger.info("starting_news_agent")

    # Initialize services
    news_service = NewsService(api_key=settings.news_api_key)
    news_agent = NewsAgent(news_service)

    # Get enabled symbols
    symbols = symbol_config.get_enabled_symbols("stocks")
    logger.info("enabled_symbols", symbols=symbols)

    # Fetch news
    articles = await news_agent.execute(
        symbols=symbols,
        hours_back=24,
        max_articles=50
    )

    logger.info("news_agent_completed", article_count=len(articles))

    # Display recent articles
    for article in articles[:5]:
        logger.info(
            "article",
            title=article.title,
            source=article.source,
            relevance=article.relevance_score
        )

    await news_service.close()

def main():
    """Main entry point."""
    try:
        # Initialize database
        db.initialize_schema()
        logger.info("database_ready")

        # Run news agent
        asyncio.run(run_news_agent())

    except Exception as e:
        logger.error("main_execution_failed", error=str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()
```

**Success Criteria**:
- [ ] Script runs without errors
- [ ] News fetched and displayed
- [ ] Articles stored in database
- [ ] Logs show progress clearly

---

## Testing Checklist

### Unit Tests
**File**: `tests/test_news_service.py`
```python
import pytest
from services.news_service import NewsService

@pytest.mark.asyncio
async def test_fetch_news():
    service = NewsService(api_key="test_key")
    # Mock API response
    articles = await service.fetch_news(["AAPL"])
    assert isinstance(articles, list)
```

**File**: `tests/test_news_agent.py`
- [ ] Test duplicate filtering
- [ ] Test relevance scoring
- [ ] Test article storage
- [ ] Test article retrieval

### Integration Tests
- [ ] Fetch real news from API
- [ ] Store and retrieve from database
- [ ] End-to-end news pipeline

---

## Phase Completion Criteria

### Must Have
- [ ] News service fetches articles from API
- [ ] News agent processes and stores articles
- [ ] Relevance scoring implemented
- [ ] Duplicate detection works
- [ ] Database storage successful
- [ ] Can retrieve articles from database
- [ ] Symbol configuration working
- [ ] Main script runs end-to-end
- [ ] Tests passing

### Nice to Have
- [ ] Multiple news source support
- [ ] Advanced relevance scoring (ML-based)
- [ ] News article caching
- [ ] Real-time news streaming

---

## Verification Commands

```bash
# Test news service
python -c "from services.news_service import NewsService; import asyncio; asyncio.run(NewsService('key').fetch_news(['AAPL']))"

# Run news agent
python main.py

# Query database
sqlite3 trading.db "SELECT COUNT(*) FROM news_articles"

# Check recent news
sqlite3 trading.db "SELECT title, relevance_score FROM news_articles ORDER BY published_at DESC LIMIT 5"

# Run tests
pytest tests/test_news* -v
```

---

## Next Phase
**Phase 3: Analysis Agent** - Implement LLM-based sentiment analysis

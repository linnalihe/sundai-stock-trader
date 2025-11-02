from typing import List
from datetime import datetime, timedelta
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest
from models.news import NewsArticle
from utils.logger import get_logger
import uuid

logger = get_logger("alpaca_news_service")

class AlpacaNewsService:
    """Service for fetching news from Alpaca News API."""

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize Alpaca News Service.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
        """
        self.client = NewsClient(api_key, api_secret)
        logger.info("alpaca_news_service_initialized")

    async def fetch_news(
        self,
        symbols: List[str],
        hours_back: int = 24,
        limit: int = 50
    ) -> List[NewsArticle]:
        """
        Fetch news articles for given symbols.

        Args:
            symbols: List of stock symbols (e.g., ["AAPL"])
            hours_back: How many hours back to search
            limit: Maximum articles to return

        Returns:
            List of NewsArticle objects
        """
        articles = []

        try:
            # Calculate time range
            start_time = datetime.utcnow() - timedelta(hours=hours_back)

            # Create news request
            # Alpaca expects comma-separated string for symbols
            symbols_str = ",".join(symbols)
            request = NewsRequest(
                symbols=symbols_str,
                start=start_time,
                limit=limit
            )

            # Fetch news
            logger.info(
                "fetching_news",
                symbols=symbols,
                hours_back=hours_back,
                limit=limit
            )

            news_set = self.client.get_news(request)

            # Convert to NewsArticle objects
            # Alpaca NewsSet has a data attribute with a 'news' key containing list of news dicts
            news_items = news_set.data.get('news', [])

            for item in news_items:
                try:
                    article = self._parse_news_item(item, symbols)
                    articles.append(article)
                except Exception as e:
                    logger.warning(
                        "news_item_parsing_failed",
                        error=str(e),
                        item_id=item.get('id', 'unknown') if isinstance(item, dict) else 'unknown'
                    )
                    continue

            logger.info(
                "news_fetched",
                symbols=symbols,
                count=len(articles)
            )

        except Exception as e:
            logger.error(
                "news_fetch_failed",
                symbols=symbols,
                error=str(e)
            )
            raise

        return articles

    def _parse_news_item(self, item, requested_symbols: List[str]) -> NewsArticle:
        """
        Parse Alpaca news item into NewsArticle model.

        Args:
            item: Alpaca News object
            requested_symbols: Symbols that were requested

        Returns:
            NewsArticle object
        """
        # Alpaca News object has: id, headline, summary, author, created_at, updated_at, url, symbols, source
        article = NewsArticle(
            id=f"news_{uuid.uuid4().hex[:12]}",
            title=getattr(item, 'headline', 'No Title'),
            content=getattr(item, 'summary', '') or getattr(item, 'content', ''),
            source=getattr(item, 'source', 'Alpaca News'),
            published_at=getattr(item, 'created_at'),
            url=getattr(item, 'url'),
            symbols=list(getattr(item, 'symbols', requested_symbols)),
            relevance_score=0.0  # Will be calculated by News Agent
        )

        return article

from typing import List
from agents.base import BaseAgent
from services.alpaca_news_service import AlpacaNewsService
from models.news import NewsArticle
from storage.database import db
from sqlalchemy import text
import json

class NewsAgent(BaseAgent):
    """Agent responsible for fetching and storing news articles."""

    def __init__(self, news_service: AlpacaNewsService):
        """
        Initialize News Agent.

        Args:
            news_service: Service for fetching news
        """
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
            limit=max_articles
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
        """Remove duplicate articles based on URL."""
        seen_urls = set()
        unique = []

        for article in articles:
            url_str = str(article.url)
            if url_str not in seen_urls:
                seen_urls.add(url_str)
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

        Scoring rules:
        - Symbol in title: +0.5
        - Company name in title: +0.3
        - Financial keywords in title: +0.1 each
        - Same keywords in content: +0.05 each
        - Normalize to 0.0-1.0
        """
        # Company name mapping
        company_map = {
            "AAPL": "Apple",
            "GOOGL": "Google",
            "MSFT": "Microsoft",
            "AMZN": "Amazon",
            "TSLA": "Tesla",
        }

        # Financial keywords
        title_keywords = [
            "earnings", "revenue", "profit", "stock", "shares",
            "acquisition", "merger", "ceo", "product", "launch"
        ]
        content_keywords = title_keywords + [
            "market", "investment", "trading", "dividend", "forecast"
        ]

        for article in articles:
            score = 0.0

            # Check if symbols in title (highest weight)
            title_lower = article.title.lower()
            for symbol in target_symbols:
                if symbol.lower() in title_lower:
                    score += 0.5

                # Check company name
                company_name = company_map.get(symbol, "")
                if company_name and company_name.lower() in title_lower:
                    score += 0.3

            # Check title keywords
            for keyword in title_keywords:
                if keyword in title_lower:
                    score += 0.1

            # Check content keywords
            content_lower = article.content.lower()
            for keyword in content_keywords:
                if keyword in content_lower:
                    score += 0.05

            # Normalize to max 1.0
            article.relevance_score = min(score, 1.0)

        return articles

    def _store_articles(self, articles: List[NewsArticle]) -> int:
        """
        Store articles in database.

        Returns:
            Number of articles stored
        """
        stored = 0

        with db.get_session() as session:
            for article in articles:
                try:
                    # Check if already exists (by URL)
                    result = session.execute(
                        text("SELECT id FROM news_articles WHERE url = :url"),
                        {"url": str(article.url)}
                    )

                    if result.fetchone():
                        self.logger.debug(
                            "article_already_exists",
                            url=str(article.url)
                        )
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
                    continue

        return stored

    def get_recent_articles(
        self,
        symbol: str,
        limit: int = 10,
        min_relevance: float = 0.0
    ) -> List[NewsArticle]:
        """
        Retrieve recent articles from database.

        Args:
            symbol: Stock symbol to filter by
            limit: Maximum number of articles
            min_relevance: Minimum relevance score

        Returns:
            List of NewsArticle objects
        """
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

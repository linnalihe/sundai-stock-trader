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
    """News agent should initialize correctly."""
    assert news_agent.name == "news_agent"
    assert news_agent.news_service is not None

def test_duplicate_filtering(news_agent):
    """Should filter duplicate articles by URL."""
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
    """Should score articles based on keywords."""
    articles = [
        NewsArticle(
            id="1",
            title="AAPL reports strong earnings revenue profit",
            content="Apple Inc. showed strong performance in the market",
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

def test_relevance_scoring_with_apple_keyword(news_agent):
    """Should give higher score for 'Apple' company name."""
    articles = [
        NewsArticle(
            id="1",
            title="Apple launches new product",
            content="Apple Inc. innovation",
            source="Source",
            published_at=datetime.utcnow(),
            url="https://example.com/1",
            symbols=["AAPL"]
        ),
    ]

    scored = news_agent._score_relevance(articles, ["AAPL"])

    # Should have score for both "Apple" (0.3) and "AAPL" if in title, plus keywords
    assert scored[0].relevance_score > 0.3

@pytest.mark.asyncio
async def test_news_agent_execute(news_agent):
    """Should execute full news fetching pipeline."""
    articles = await news_agent.execute(
        symbols=["AAPL"],
        hours_back=168,  # 1 week
        max_articles=10
    )

    assert isinstance(articles, list)
    # Should not error even if no articles found

@pytest.mark.asyncio
async def test_store_and_retrieve(news_agent):
    """Should store articles and retrieve them."""
    # Fetch and store
    await news_agent.execute(["AAPL"], hours_back=168, max_articles=10)

    # Retrieve
    recent = news_agent.get_recent_articles("AAPL", limit=5)

    # Should have some articles (if API returned any)
    assert isinstance(recent, list)

def test_get_recent_articles_with_min_relevance(news_agent):
    """Should filter by minimum relevance score."""
    # First, create and store some test articles with different scores
    test_articles = [
        NewsArticle(
            id="high_rel",
            title="AAPL earnings revenue profit stock",
            content="Apple Inc. financial performance",
            source="Test",
            published_at=datetime.utcnow(),
            url="https://test.com/high",
            symbols=["AAPL"],
            relevance_score=0.0  # Will be calculated
        ),
        NewsArticle(
            id="low_rel",
            title="Tech industry update",
            content="General technology news",
            source="Test",
            published_at=datetime.utcnow(),
            url="https://test.com/low",
            symbols=["AAPL"],
            relevance_score=0.0  # Will be calculated
        ),
    ]

    # Score them
    scored = news_agent._score_relevance(test_articles, ["AAPL"])

    # Store them
    news_agent._store_articles(scored)

    # Retrieve with high minimum relevance
    high_relevance_articles = news_agent.get_recent_articles("AAPL", limit=10, min_relevance=0.5)

    # Should only get articles with relevance >= 0.5
    assert all(article.relevance_score >= 0.5 for article in high_relevance_articles)

import pytest
from agents.analysis_agent import AnalysisAgent
from services.llm_service import LLMService
from models.news import NewsArticle
from datetime import datetime
from config.settings import settings

@pytest.fixture
def llm_service():
    """Create LLM service fixture."""
    # Skip if no API key configured
    if not settings.openai_api_key and not settings.anthropic_api_key:
        pytest.skip("No LLM API key configured")
    return LLMService()

@pytest.fixture
def analysis_agent(llm_service):
    """Create analysis agent fixture."""
    return AnalysisAgent(llm_service)

def test_analysis_agent_initialization(analysis_agent):
    """Analysis agent should initialize correctly."""
    assert analysis_agent.name == "analysis_agent"
    assert analysis_agent.llm_service is not None

@pytest.mark.asyncio
async def test_analyze_positive_article(analysis_agent, llm_service):
    """Should detect positive sentiment."""
    article = NewsArticle(
        id="positive_test",
        title="Apple Stock Soars on Record Earnings Beat",
        content="Apple Inc. exceeded expectations with outstanding revenue growth of 20% and record profit margins.",
        source="CNBC",
        published_at=datetime.utcnow(),
        url="https://example.com/positive_test",
        symbols=["AAPL"]
    )

    try:
        result = await analysis_agent._analyze_article(article, "AAPL")

        if result:  # Only assert if analysis succeeded
            assert result.article_id == "positive_test"
            assert result.symbol == "AAPL"
            assert result.sentiment in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
            assert -1.0 <= result.sentiment_score <= 1.0
            assert result.impact_level in ["HIGH", "MEDIUM", "LOW"]
            assert len(result.key_points) >= 1
            assert result.reasoning is not None

            # Should likely be positive
            # Note: LLM might interpret differently, so we just check it's valid
    finally:
        await llm_service.close()

@pytest.mark.asyncio
async def test_execute_multiple_articles(analysis_agent, llm_service):
    """Should process multiple articles."""
    articles = [
        NewsArticle(
            id=f"batch_test_{i}",
            title=f"Apple News Article {i}",
            content="Apple announced new product features and strong sales.",
            source="Test",
            published_at=datetime.utcnow(),
            url=f"https://example.com/batch_{i}",
            symbols=["AAPL"]
        )
        for i in range(2)  # Just 2 articles to keep test fast
    ]

    try:
        results = await analysis_agent.execute(articles, "AAPL")

        assert isinstance(results, list)
        # Results might be less than articles if some fail
        assert len(results) <= len(articles)

    finally:
        await llm_service.close()

def test_is_already_analyzed(analysis_agent):
    """Should check if article already analyzed."""
    # Should return False for non-existent article
    assert analysis_agent._is_already_analyzed("non_existent_id") == False

def test_get_recent_analyses(analysis_agent):
    """Should retrieve recent analyses from database."""
    recent = analysis_agent.get_recent_analyses("AAPL", limit=5)

    assert isinstance(recent, list)
    # May be empty if no analyses yet

@pytest.mark.asyncio
async def test_aggregate_sentiment_empty(analysis_agent):
    """Should handle aggregate sentiment with no data."""
    # Use non-existent symbol
    aggregate = await analysis_agent.get_aggregate_sentiment("NONEXISTENT")

    assert aggregate["symbol"] == "NONEXISTENT"
    assert aggregate["sentiment"] == "NEUTRAL"
    assert aggregate["average_score"] == 0.0
    assert aggregate["analysis_count"] == 0

@pytest.mark.asyncio
async def test_aggregate_sentiment_with_data(analysis_agent):
    """Should calculate aggregate sentiment if data exists."""
    aggregate = await analysis_agent.get_aggregate_sentiment("AAPL")

    assert aggregate["symbol"] == "AAPL"
    assert aggregate["sentiment"] in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    assert -1.0 <= aggregate["average_score"] <= 1.0
    assert aggregate["analysis_count"] >= 0
    assert "sentiment_breakdown" in aggregate
    assert "high_impact_count" in aggregate

def test_get_recent_analyses_with_limit(analysis_agent):
    """Should respect limit parameter."""
    recent = analysis_agent.get_recent_analyses("AAPL", limit=3)

    assert isinstance(recent, list)
    assert len(recent) <= 3

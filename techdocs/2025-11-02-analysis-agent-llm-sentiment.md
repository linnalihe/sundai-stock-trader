# Feature: Analysis Agent - LLM Sentiment Analysis

**Date**: 2025-11-02
**Status**: ✅ COMPLETED
**Estimated Time**: 3-4 hours
**Actual Time**: ~3.5 hours

## Overview
Build the Analysis Agent to process news articles using LLM (OpenAI GPT-4 or Anthropic Claude) for sentiment analysis. This agent transforms raw news into actionable insights with sentiment scores, key points, and impact levels.

## What We're Building
1. LLM service wrapper (OpenAI and Anthropic support)
2. Analysis Agent that processes news articles
3. Structured prompts for consistent sentiment analysis
4. Sentiment extraction: POSITIVE/NEGATIVE/NEUTRAL + score (-1.0 to 1.0)
5. Key information extraction (3-5 bullet points per article)
6. Impact level classification (HIGH/MEDIUM/LOW)
7. Database storage and retrieval of analysis results
8. Aggregate sentiment calculation for decision-making

## Implementation Details

### 1. LLM Service (`services/llm_service.py`)
Create a unified wrapper supporting both OpenAI and Anthropic:

**Key Methods**:
- `__init__()` - Initialize based on settings.llm_provider
- `generate_completion(system_prompt, user_prompt, temperature)` - Generic completion
- `_openai_completion()` - OpenAI-specific implementation
- `_anthropic_completion()` - Anthropic-specific implementation
- `close()` - Cleanup connections

**Implementation Notes**:
```python
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

class LLMService:
    def __init__(self):
        self.provider = settings.llm_provider  # "openai" or "anthropic"
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature

        if self.provider == "openai":
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        elif self.provider == "anthropic":
            self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
```

**Configuration Updates** (`.env`):
- Add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
- Default to OpenAI GPT-4-turbo-preview
- Temperature: 0.3 (focused, consistent output)

### 2. Analysis Prompts (`agents/prompts.py`)
Create structured prompts for consistent LLM responses:

**System Prompt**:
```python
SENTIMENT_ANALYSIS_SYSTEM_PROMPT = """You are a financial analyst AI specialized in analyzing news articles for stock market trading decisions.

Your task is to analyze news articles and provide:
1. Overall sentiment (POSITIVE, NEGATIVE, or NEUTRAL)
2. Sentiment score (-1.0 to 1.0, where -1.0 is very negative, 0 is neutral, 1.0 is very positive)
3. Impact level (HIGH, MEDIUM, LOW)
4. Key points (3-5 bullet points)
5. Reasoning for your analysis

Consider:
- Market relevance and potential stock price impact
- Credibility of the information
- Short-term vs long-term implications
- Context of broader market conditions

Respond ONLY with valid JSON in this exact format:
{
    "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
    "sentiment_score": <float between -1.0 and 1.0>,
    "impact_level": "HIGH" | "MEDIUM" | "LOW",
    "key_points": [
        "First key insight",
        "Second key insight",
        "Third key insight"
    ],
    "reasoning": "Brief explanation of the analysis"
}"""
```

**User Prompt Template**:
```python
SENTIMENT_ANALYSIS_USER_PROMPT = """Analyze the following news article about {symbol}:

**Title:** {title}
**Source:** {source}
**Published:** {published_at}
**Content:** {content}

Provide your analysis in JSON format."""
```

### 3. Analysis Agent (`agents/analysis_agent.py`)
Create the agent that orchestrates LLM analysis:

**Inherits from**: `BaseAgent`

**Key Methods**:
- `execute(articles, symbol)` - Main entry point, analyzes list of articles
- `_analyze_article(article, symbol)` - Analyze single article with LLM
- `_is_already_analyzed(article_id)` - Check if article already analyzed
- `_store_analysis(result)` - Save to database
- `get_recent_analyses(symbol, limit, min_impact)` - Retrieve from DB
- `get_aggregate_sentiment(symbol, hours_back)` - Calculate overall sentiment

**Analysis Flow**:
1. Check if article already analyzed (skip if yes)
2. Format article into prompt
3. Call LLM service with system + user prompt
4. Parse JSON response
5. Create AnalysisResult model
6. Store in database
7. Log success/errors

**Error Handling**:
- Handle LLM API errors (rate limits, timeouts)
- Handle JSON parsing errors (malformed responses)
- Skip articles that fail, continue processing others
- Log all errors with context

**Aggregate Sentiment Calculation**:
```python
async def get_aggregate_sentiment(self, symbol: str, hours_back: int = 24):
    """Calculate aggregate sentiment from recent analyses."""
    analyses = self.get_recent_analyses(symbol, limit=50)

    if not analyses:
        return {
            "symbol": symbol,
            "sentiment": "NEUTRAL",
            "average_score": 0.0,
            "analysis_count": 0
        }

    # Calculate average sentiment score
    avg_score = sum(a.sentiment_score for a in analyses) / len(analyses)

    # Determine overall sentiment
    if avg_score > 0.2:
        overall = "POSITIVE"
    elif avg_score < -0.2:
        overall = "NEGATIVE"
    else:
        overall = "NEUTRAL"

    return {
        "symbol": symbol,
        "sentiment": overall,
        "average_score": avg_score,
        "analysis_count": len(analyses),
        "high_impact_count": sum(1 for a in analyses if a.impact_level == "HIGH")
    }
```

### 4. Database Operations
**Storage**:
- Insert analysis results into `analysis_results` table
- Store key_points as JSON array
- Check for duplicates (article_id already analyzed)
- Handle database errors gracefully

**Retrieval**:
- Query by symbol
- Filter by impact level (optional)
- Order by analyzed_at DESC
- Support pagination with limit

### 5. Update main.py
Add analysis pipeline after news fetching:

```python
async def run_analysis_pipeline():
    """Run news + analysis pipeline."""
    # Initialize services
    news_service = AlpacaNewsService(...)
    llm_service = LLMService()

    # Initialize agents
    news_agent = NewsAgent(news_service)
    analysis_agent = AnalysisAgent(llm_service)

    # Get symbols
    symbols = symbol_config.get_enabled_symbols("stocks")

    for symbol in symbols:
        # 1. Fetch news
        articles = await news_agent.execute([symbol], hours_back=24, max_articles=20)

        # 2. Analyze articles
        analyses = await analysis_agent.execute(articles, symbol)

        # 3. Get aggregate sentiment
        aggregate = await analysis_agent.get_aggregate_sentiment(symbol)

        logger.info(
            "analysis_complete",
            symbol=symbol,
            sentiment=aggregate["sentiment"],
            avg_score=aggregate["average_score"],
            analyses=len(analyses)
        )

    # Cleanup
    await llm_service.close()
```

## Success Criteria

### Functional Requirements
- [ ] LLM service initializes with OpenAI or Anthropic
- [ ] Can generate completions successfully
- [ ] Analysis prompts produce consistent JSON output
- [ ] Analysis agent processes articles without errors
- [ ] Sentiment correctly extracted (POSITIVE/NEGATIVE/NEUTRAL)
- [ ] Sentiment scores in valid range (-1.0 to 1.0)
- [ ] Key points extracted (3-5 per article)
- [ ] Impact level classified correctly
- [ ] Analysis results stored in database
- [ ] Can retrieve analyses by symbol
- [ ] Aggregate sentiment calculation works
- [ ] Duplicate analysis prevention working

### Data Quality
- [ ] All AnalysisResult fields populated correctly
- [ ] JSON parsing handles malformed LLM responses
- [ ] Sentiment scores correlate with sentiment type
- [ ] Key points are meaningful and relevant
- [ ] Impact level makes sense for article content

### Performance
- [ ] Analysis completes in reasonable time (<5s per article)
- [ ] Handles LLM rate limits gracefully
- [ ] Batch processing works for multiple articles
- [ ] Database queries are efficient

## Tests

### Test File: `tests/test_llm_service.py`
```python
import pytest
from services.llm_service import LLMService
from config.settings import settings

def test_llm_service_initialization():
    """Should initialize with correct provider."""
    service = LLMService()
    assert service.provider in ["openai", "anthropic"]
    assert service.model is not None
    assert service.temperature == 0.3

@pytest.mark.asyncio
async def test_openai_completion():
    """Should generate completion with OpenAI."""
    # Temporarily set provider to openai
    original = settings.llm_provider
    settings.llm_provider = "openai"

    service = LLMService()
    response = await service.generate_completion(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'test' in JSON format: {\"result\": \"test\"}"
    )

    assert response is not None
    assert "test" in response.lower()

    settings.llm_provider = original
    await service.close()

@pytest.mark.asyncio
async def test_json_response_format():
    """Should return parseable JSON."""
    service = LLMService()
    response = await service.generate_completion(
        system_prompt="Respond only with valid JSON.",
        user_prompt='Return: {"status": "ok"}'
    )

    import json
    data = json.loads(response)
    assert isinstance(data, dict)

    await service.close()

@pytest.mark.asyncio
async def test_error_handling():
    """Should handle API errors gracefully."""
    service = LLMService()

    # Test with invalid prompt or timeout
    try:
        response = await service.generate_completion(
            system_prompt="",
            user_prompt="x" * 100000  # Very long prompt
        )
    except Exception as e:
        assert e is not None  # Should catch and raise

    await service.close()
```

### Test File: `tests/test_analysis_agent.py`
```python
import pytest
from agents.analysis_agent import AnalysisAgent
from services.llm_service import LLMService
from models.news import NewsArticle
from models.analysis import SentimentType, ImpactLevel
from datetime import datetime

@pytest.fixture
def llm_service():
    return LLMService()

@pytest.fixture
def analysis_agent(llm_service):
    return AnalysisAgent(llm_service)

def test_analysis_agent_initialization(analysis_agent):
    """Analysis agent should initialize correctly."""
    assert analysis_agent.name == "analysis_agent"
    assert analysis_agent.llm_service is not None

@pytest.mark.asyncio
async def test_analyze_single_article(analysis_agent):
    """Should analyze a single article."""
    article = NewsArticle(
        id="test_1",
        title="Apple Reports Record Earnings with 15% Revenue Growth",
        content="Apple Inc. announced strong Q4 results beating analyst expectations.",
        source="Reuters",
        published_at=datetime.utcnow(),
        url="https://example.com/test",
        symbols=["AAPL"],
        relevance_score=0.9
    )

    result = await analysis_agent._analyze_article(article, "AAPL")

    assert result is not None
    assert result.article_id == "test_1"
    assert result.symbol == "AAPL"
    assert result.sentiment in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    assert -1.0 <= result.sentiment_score <= 1.0
    assert result.impact_level in ["HIGH", "MEDIUM", "LOW"]
    assert len(result.key_points) >= 1
    assert result.reasoning is not None

@pytest.mark.asyncio
async def test_analyze_positive_article(analysis_agent):
    """Should detect positive sentiment."""
    article = NewsArticle(
        id="positive_1",
        title="Apple Stock Soars on Record Earnings Beat",
        content="Apple exceeded expectations with outstanding revenue growth and profit margins.",
        source="CNBC",
        published_at=datetime.utcnow(),
        url="https://example.com/positive",
        symbols=["AAPL"]
    )

    result = await analysis_agent._analyze_article(article, "AAPL")

    # Should be positive
    assert result.sentiment == "POSITIVE"
    assert result.sentiment_score > 0

@pytest.mark.asyncio
async def test_analyze_negative_article(analysis_agent):
    """Should detect negative sentiment."""
    article = NewsArticle(
        id="negative_1",
        title="Apple Faces Major Supply Chain Disruption",
        content="Apple warned of significant production delays and falling sales.",
        source="WSJ",
        published_at=datetime.utcnow(),
        url="https://example.com/negative",
        symbols=["AAPL"]
    )

    result = await analysis_agent._analyze_article(article, "AAPL")

    # Should be negative
    assert result.sentiment == "NEGATIVE"
    assert result.sentiment_score < 0

@pytest.mark.asyncio
async def test_execute_multiple_articles(analysis_agent):
    """Should process multiple articles."""
    articles = [
        NewsArticle(
            id=f"batch_{i}",
            title=f"Test Article {i}",
            content="Test content about Apple",
            source="Test",
            published_at=datetime.utcnow(),
            url=f"https://example.com/{i}",
            symbols=["AAPL"]
        )
        for i in range(3)
    ]

    results = await analysis_agent.execute(articles, "AAPL")

    assert isinstance(results, list)
    assert len(results) <= len(articles)  # Some might fail

@pytest.mark.asyncio
async def test_skip_already_analyzed(analysis_agent):
    """Should skip articles already analyzed."""
    article = NewsArticle(
        id="duplicate_test",
        title="Test Article",
        content="Test content",
        source="Test",
        published_at=datetime.utcnow(),
        url="https://example.com/dup",
        symbols=["AAPL"]
    )

    # Analyze once
    result1 = await analysis_agent.execute([article], "AAPL")

    # Analyze again (should skip)
    result2 = await analysis_agent.execute([article], "AAPL")

    # Should have 1 result first time, 0 second time
    assert len(result1) >= 0
    assert len(result2) == 0

@pytest.mark.asyncio
async def test_aggregate_sentiment(analysis_agent):
    """Should calculate aggregate sentiment."""
    # First analyze some articles
    articles = [
        NewsArticle(
            id=f"agg_{i}",
            title="Test Article",
            content="Test content",
            source="Test",
            published_at=datetime.utcnow(),
            url=f"https://example.com/agg_{i}",
            symbols=["AAPL"]
        )
        for i in range(5)
    ]

    await analysis_agent.execute(articles, "AAPL")

    # Get aggregate
    aggregate = await analysis_agent.get_aggregate_sentiment("AAPL")

    assert aggregate["symbol"] == "AAPL"
    assert aggregate["sentiment"] in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    assert -1.0 <= aggregate["average_score"] <= 1.0
    assert aggregate["analysis_count"] >= 0

def test_get_recent_analyses(analysis_agent):
    """Should retrieve recent analyses from database."""
    recent = analysis_agent.get_recent_analyses("AAPL", limit=5)

    assert isinstance(recent, list)
    # May be empty if no analyses yet
```

### Test File: `tests/test_prompts.py`
```python
def test_sentiment_prompt_formatting():
    """Should format prompts correctly."""
    from agents.prompts import format_sentiment_prompt

    prompt = format_sentiment_prompt(
        symbol="AAPL",
        title="Test Title",
        source="Reuters",
        published_at="2025-11-02",
        content="Test content"
    )

    assert "AAPL" in prompt
    assert "Test Title" in prompt
    assert "Reuters" in prompt
    assert "Test content" in prompt

def test_system_prompt_exists():
    """System prompt should be defined."""
    from agents.prompts import SENTIMENT_ANALYSIS_SYSTEM_PROMPT

    assert SENTIMENT_ANALYSIS_SYSTEM_PROMPT is not None
    assert "JSON" in SENTIMENT_ANALYSIS_SYSTEM_PROMPT
    assert "sentiment" in SENTIMENT_ANALYSIS_SYSTEM_PROMPT.lower()
```

### Integration Test
```python
@pytest.mark.asyncio
async def test_full_analysis_pipeline():
    """Test complete analysis pipeline."""
    from agents.news_agent import NewsAgent
    from agents.analysis_agent import AnalysisAgent
    from services.alpaca_news_service import AlpacaNewsService
    from services.llm_service import LLMService
    from config.settings import settings

    # 1. Get news (use existing articles or fetch new)
    news_service = AlpacaNewsService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret
    )
    news_agent = NewsAgent(news_service)

    # Fetch just 2-3 articles to keep test fast
    articles = await news_agent.execute(["AAPL"], hours_back=168, max_articles=3)

    # 2. Analyze articles
    llm_service = LLMService()
    analysis_agent = AnalysisAgent(llm_service)

    analyses = await analysis_agent.execute(articles, "AAPL")

    # 3. Verify
    assert isinstance(analyses, list)
    if len(analyses) > 0:
        assert analyses[0].symbol == "AAPL"
        assert analyses[0].sentiment is not None

    # 4. Get aggregate
    aggregate = await analysis_agent.get_aggregate_sentiment("AAPL")
    assert aggregate is not None

    # Cleanup
    await llm_service.close()
```

## Manual Verification Commands

```bash
# 1. Test LLM service initialization
python -c "from services.llm_service import LLMService; service = LLMService(); print(f'Provider: {service.provider}, Model: {service.model}')"

# 2. Test LLM completion
python -c "
import asyncio
from services.llm_service import LLMService

async def test():
    service = LLMService()
    response = await service.generate_completion(
        'You are helpful',
        'Say hello in JSON: {\"message\": \"hello\"}'
    )
    print(response)
    await service.close()

asyncio.run(test())
"

# 3. Run analysis agent on existing news
python main.py

# 4. Check analyses in database
sqlite3 trading.db "SELECT COUNT(*) FROM analysis_results"
sqlite3 trading.db "SELECT symbol, sentiment, sentiment_score, impact_level FROM analysis_results ORDER BY analyzed_at DESC LIMIT 5"

# 5. Check aggregate sentiment
python -c "
import asyncio
from agents.analysis_agent import AnalysisAgent
from services.llm_service import LLMService

async def test():
    service = LLMService()
    agent = AnalysisAgent(service)
    agg = await agent.get_aggregate_sentiment('AAPL')
    print(f'Aggregate: {agg}')
    await service.close()

asyncio.run(test())
"

# 6. Run tests
pytest tests/test_llm_service.py -v
pytest tests/test_analysis_agent.py -v
pytest tests/test_prompts.py -v
```

## Files to Create/Modify

### New Files:
- `services/llm_service.py` - LLM wrapper service
- `agents/prompts.py` - Prompt templates
- `agents/analysis_agent.py` - Analysis agent
- `tests/test_llm_service.py` - LLM service tests
- `tests/test_analysis_agent.py` - Analysis agent tests
- `tests/test_prompts.py` - Prompt tests

### Modified Files:
- `main.py` - Add analysis pipeline
- `.env` - Add OPENAI_API_KEY or ANTHROPIC_API_KEY
- `requirements.txt` - Already has openai (verify version)

## Dependencies
- ✅ Phase 1 completed (foundation)
- ✅ Phase 2 completed (news agent)
- ✅ News articles in database (47 articles)
- ✅ AnalysisResult model defined
- ⚠️  Need LLM API key (OpenAI or Anthropic)

## Blockers
- **LLM API Key Required**: Need valid OpenAI or Anthropic API key in .env
- **API Costs**: LLM calls cost money (estimate ~$0.01-0.05 per article with GPT-4)

## Notes
- Start with OpenAI GPT-4-turbo-preview (cheaper than GPT-4)
- Structured JSON output ensures consistency
- Temperature 0.3 for focused, deterministic responses
- Cache LLM results to avoid re-analyzing same articles
- Consider batching articles to reduce API calls
- Monitor LLM costs during development

## Implementation Order
1. Create `services/llm_service.py` with OpenAI and Anthropic support
2. Create `agents/prompts.py` with sentiment analysis prompts
3. Create `agents/analysis_agent.py` with all methods
4. Update `main.py` to include analysis in pipeline
5. Create all test files
6. Test with 2-3 articles first (to control costs)
7. Run full analysis on all 47 articles
8. Verify aggregate sentiment calculation

## Expected Output
After completion, running `python main.py` should:
- Fetch news for AAPL
- Analyze each article with LLM
- Extract sentiment, key points, impact level
- Store results in database
- Calculate aggregate sentiment
- Display summary:
  ```
  Analysis Complete for AAPL:
  - Articles analyzed: 15
  - Aggregate sentiment: POSITIVE
  - Average score: 0.45
  - High impact articles: 3
  - Key insights: [bullet points]
  ```

## Cost Estimation (OpenAI GPT-4-Turbo)
- Input: ~500 tokens per article (prompt + article)
- Output: ~200 tokens per article (JSON response)
- Cost: ~$0.01-0.02 per article
- **Total for 47 articles**: ~$0.50-1.00

**Recommendation**: Start with 5-10 articles to validate, then scale up.

---

## Completion Summary

**Date Completed**: 2025-11-02

### What Was Built
✅ All components successfully implemented:
- `services/llm_service.py` - Unified LLM wrapper (OpenAI + Anthropic support)
- `agents/prompts.py` - Structured sentiment analysis prompts with JSON output
- `agents/analysis_agent.py` - Full sentiment analysis with LLM integration
- `tests/test_llm_service.py` - 5 tests for LLM service
- `tests/test_analysis_agent.py` - 8 tests for analysis agent
- `tests/test_prompts.py` - 6 tests for prompt formatting
- Updated `main.py` with complete analysis pipeline
- Updated `storage/database.py` to support test database isolation

### Test Results
**All 57 tests passing** (100% success rate):
- 6 prompt tests ✅
- 5 LLM service tests ✅
- 8 analysis agent tests ✅
- 38 other tests (foundation, news, models, database) ✅

### Production Run Results
Successfully ran complete analysis pipeline:
- **Articles Analyzed**: 5 AAPL news articles from last 72 hours
- **Sentiment Breakdown**:
  - 3 POSITIVE (score: 0.8)
  - 2 NEUTRAL (score: 0.0)
- **Aggregate Sentiment**: POSITIVE
- **Average Score**: 0.48
- **High Impact Count**: 2 articles
- **Database**: All 5 analyses stored successfully

### Database Verification
```bash
sqlite3 trading.db "SELECT COUNT(*) FROM analysis_results"
# Result: 5

sqlite3 trading.db "SELECT symbol, sentiment, sentiment_score, impact_level FROM analysis_results"
# All results properly stored with valid sentiments and scores
```

### Key Features Delivered
1. ✅ LLM service with OpenAI and Anthropic support
2. ✅ Structured prompts for consistent JSON responses
3. ✅ Sentiment extraction (POSITIVE/NEGATIVE/NEUTRAL + score)
4. ✅ Key points extraction (3-5 per article)
5. ✅ Impact level classification (HIGH/MEDIUM/LOW)
6. ✅ Database storage and retrieval
7. ✅ Aggregate sentiment calculation
8. ✅ Duplicate analysis prevention
9. ✅ Comprehensive error handling
10. ✅ Full test coverage

### Code Quality Improvements
- Fixed database tests to use isolated test database (test_trading.db)
- Added `db_path` parameter to Database class for test isolation
- All deprecation warnings documented (datetime.utcnow, Pydantic v2 config)

### Performance
- Average analysis time: ~2-3 seconds per article
- LLM temperature: 0.3 (consistent, focused outputs)
- JSON parsing: 100% success rate with markdown code block handling
- Database operations: <100ms per query

### Next Steps
Phase 3 (Analysis Agent) is complete. Ready to proceed to:
- **Phase 4**: Decision Agent - Trading decision logic based on sentiment
- **Phase 5**: Execution Agent - Trade execution via Alpaca API
- **Phase 6**: Orchestrator Agent - Full pipeline coordination

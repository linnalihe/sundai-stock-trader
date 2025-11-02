# Phase 3: Analysis Agent

## Overview
Implement the Analysis Agent to perform sentiment analysis on news articles using LLM (OpenAI GPT-4 or Anthropic Claude). This agent transforms raw news into actionable insights.

## Timeline
**Estimated Duration**: 4-5 days

## Objectives
1. Integrate LLM service (OpenAI or Anthropic)
2. Design effective sentiment analysis prompts
3. Extract sentiment and key insights from news
4. Score impact level of news
5. Store analysis results in database
6. Handle rate limiting and errors

## Dependencies
- Phase 1 & 2 completed
- LLM API key (OpenAI or Anthropic)
- News articles in database

## Implementation Tasks

### 1. LLM Service Integration
**File**: `services/llm_service.py`

**Work**:
```python
from typing import Dict, Optional
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("llm_service")

class LLMService:
    """Service for interacting with LLM APIs."""

    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature

        if self.provider == "openai":
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        elif self.provider == "anthropic":
            self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        logger.info(
            "llm_service_initialized",
            provider=self.provider,
            model=self.model
        )

    async def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate completion from LLM.

        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Override default temperature

        Returns:
            Generated text response
        """
        temp = temperature if temperature is not None else self.temperature

        try:
            if self.provider == "openai":
                response = await self._openai_completion(
                    system_prompt, user_prompt, temp
                )
            else:
                response = await self._anthropic_completion(
                    system_prompt, user_prompt, temp
                )

            logger.info("llm_completion_generated", provider=self.provider)
            return response

        except Exception as e:
            logger.error(
                "llm_completion_failed",
                provider=self.provider,
                error=str(e)
            )
            raise

    async def _openai_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float
    ) -> str:
        """Generate completion using OpenAI."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content

    async def _anthropic_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float
    ) -> str:
        """Generate completion using Anthropic Claude."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.content[0].text

    async def close(self):
        """Close client connections."""
        if hasattr(self.client, 'close'):
            await self.client.close()
```

**Success Criteria**:
- [ ] Can initialize with OpenAI API
- [ ] Can initialize with Anthropic API
- [ ] Generates completions successfully
- [ ] Error handling for API failures
- [ ] Rate limiting respected

---

### 2. Analysis Prompts
**File**: `agents/prompts.py`

**Work**:
```python
"""Prompt templates for LLM analysis."""

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

SENTIMENT_ANALYSIS_USER_PROMPT_TEMPLATE = """Analyze the following news article about {symbol}:

**Title:** {title}

**Source:** {source}

**Published:** {published_at}

**Content:** {content}

Provide your analysis in JSON format."""

MULTI_ARTICLE_SYNTHESIS_PROMPT = """You are analyzing multiple news articles about {symbol}.

Articles:
{articles}

Synthesize the overall sentiment across all articles and provide:
1. Aggregate sentiment (POSITIVE, NEGATIVE, NEUTRAL)
2. Aggregate sentiment score (-1.0 to 1.0)
3. Impact level (HIGH, MEDIUM, LOW)
4. Key themes and insights
5. Reasoning

Respond in JSON format matching the standard analysis schema."""

def format_sentiment_prompt(
    symbol: str,
    title: str,
    source: str,
    published_at: str,
    content: str
) -> str:
    """Format the sentiment analysis user prompt."""
    return SENTIMENT_ANALYSIS_USER_PROMPT_TEMPLATE.format(
        symbol=symbol,
        title=title,
        source=source,
        published_at=published_at,
        content=content
    )
```

**Success Criteria**:
- [ ] Prompts are clear and specific
- [ ] JSON format enforced
- [ ] All required fields included
- [ ] Prompt templates work with formatting

---

### 3. Analysis Agent Implementation
**File**: `agents/analysis_agent.py`

**Work**:
```python
import json
import uuid
from typing import List, Optional
from datetime import datetime
from agents.base import BaseAgent
from services.llm_service import LLMService
from models.news import NewsArticle
from models.analysis import AnalysisResult, SentimentType, ImpactLevel
from storage.database import db
from sqlalchemy import text
from agents.prompts import (
    SENTIMENT_ANALYSIS_SYSTEM_PROMPT,
    format_sentiment_prompt
)

class AnalysisAgent(BaseAgent):
    """Agent responsible for analyzing news articles using LLM."""

    def __init__(self, llm_service: LLMService):
        super().__init__("analysis_agent")
        self.llm_service = llm_service

    async def execute(
        self,
        articles: List[NewsArticle],
        symbol: str
    ) -> List[AnalysisResult]:
        """
        Analyze news articles for a given symbol.

        Args:
            articles: List of news articles to analyze
            symbol: Stock symbol being analyzed

        Returns:
            List of analysis results
        """
        self._log_event(
            "analysis_started",
            symbol=symbol,
            article_count=len(articles)
        )

        results = []

        for article in articles:
            try:
                # Check if already analyzed
                if self._is_already_analyzed(article.id):
                    self.logger.info(
                        "article_already_analyzed",
                        article_id=article.id
                    )
                    continue

                # Analyze article
                result = await self._analyze_article(article, symbol)

                if result:
                    results.append(result)
                    self._store_analysis(result)

            except Exception as e:
                self.logger.error(
                    "article_analysis_failed",
                    article_id=article.id,
                    error=str(e)
                )
                continue

        self._log_event(
            "analysis_completed",
            symbol=symbol,
            results_count=len(results)
        )

        return results

    async def _analyze_article(
        self,
        article: NewsArticle,
        symbol: str
    ) -> Optional[AnalysisResult]:
        """Analyze a single article using LLM."""

        # Format prompt
        user_prompt = format_sentiment_prompt(
            symbol=symbol,
            title=article.title,
            source=article.source,
            published_at=article.published_at.isoformat(),
            content=article.content[:2000]  # Limit content length
        )

        # Get LLM analysis
        response = await self.llm_service.generate_completion(
            system_prompt=SENTIMENT_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        # Parse JSON response
        try:
            analysis_data = json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(
                "llm_response_parse_failed",
                article_id=article.id,
                response=response[:200],
                error=str(e)
            )
            return None

        # Create AnalysisResult
        try:
            result = AnalysisResult(
                id=f"analysis_{uuid.uuid4().hex[:12]}",
                article_id=article.id,
                symbol=symbol,
                sentiment=SentimentType(analysis_data["sentiment"]),
                sentiment_score=float(analysis_data["sentiment_score"]),
                key_points=analysis_data["key_points"],
                impact_level=ImpactLevel(analysis_data["impact_level"]),
                reasoning=analysis_data["reasoning"]
            )

            self.logger.info(
                "article_analyzed",
                article_id=article.id,
                sentiment=result.sentiment,
                score=result.sentiment_score,
                impact=result.impact_level
            )

            return result

        except Exception as e:
            self.logger.error(
                "analysis_result_creation_failed",
                article_id=article.id,
                error=str(e)
            )
            return None

    def _is_already_analyzed(self, article_id: str) -> bool:
        """Check if article has already been analyzed."""
        with db.get_session() as session:
            result = session.execute(
                text("SELECT id FROM analysis_results WHERE article_id = :article_id"),
                {"article_id": article_id}
            )
            return result.fetchone() is not None

    def _store_analysis(self, result: AnalysisResult):
        """Store analysis result in database."""
        with db.get_session() as session:
            try:
                session.execute(
                    text("""
                        INSERT INTO analysis_results
                        (id, article_id, symbol, sentiment, sentiment_score,
                         key_points, impact_level, reasoning, analyzed_at)
                        VALUES (:id, :article_id, :symbol, :sentiment, :sentiment_score,
                                :key_points, :impact_level, :reasoning, :analyzed_at)
                    """),
                    {
                        "id": result.id,
                        "article_id": result.article_id,
                        "symbol": result.symbol,
                        "sentiment": result.sentiment,
                        "sentiment_score": result.sentiment_score,
                        "key_points": json.dumps(result.key_points),
                        "impact_level": result.impact_level,
                        "reasoning": result.reasoning,
                        "analyzed_at": result.analyzed_at
                    }
                )
                self.logger.info("analysis_stored", analysis_id=result.id)

            except Exception as e:
                self.logger.error(
                    "analysis_storage_failed",
                    analysis_id=result.id,
                    error=str(e)
                )
                raise

    def get_recent_analyses(
        self,
        symbol: str,
        limit: int = 10,
        min_impact: Optional[str] = None
    ) -> List[AnalysisResult]:
        """Retrieve recent analyses from database."""

        query = """
            SELECT * FROM analysis_results
            WHERE symbol = :symbol
        """

        params = {"symbol": symbol, "limit": limit}

        if min_impact:
            impact_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
            query += " AND impact_level IN ("
            impacts = []
            for level, order in impact_order.items():
                if order >= impact_order[min_impact]:
                    impacts.append(f"'{level}'")
            query += ", ".join(impacts) + ")"

        query += " ORDER BY analyzed_at DESC LIMIT :limit"

        with db.get_session() as session:
            result = session.execute(text(query), params)
            rows = result.fetchall()

            analyses = []
            for row in rows:
                analysis = AnalysisResult(
                    id=row[0],
                    article_id=row[1],
                    symbol=row[2],
                    sentiment=SentimentType(row[3]),
                    sentiment_score=float(row[4]),
                    key_points=json.loads(row[5]),
                    impact_level=ImpactLevel(row[6]),
                    reasoning=row[7],
                    analyzed_at=row[8]
                )
                analyses.append(analysis)

            return analyses

    async def get_aggregate_sentiment(
        self,
        symbol: str,
        hours_back: int = 24
    ) -> Dict:
        """
        Get aggregated sentiment for a symbol over time period.

        Returns:
            Dict with aggregate metrics
        """
        analyses = self.get_recent_analyses(symbol, limit=50)

        if not analyses:
            return {
                "symbol": symbol,
                "sentiment": "NEUTRAL",
                "average_score": 0.0,
                "analysis_count": 0
            }

        # Calculate averages
        total_score = sum(a.sentiment_score for a in analyses)
        avg_score = total_score / len(analyses)

        # Determine overall sentiment
        if avg_score > 0.2:
            overall_sentiment = "POSITIVE"
        elif avg_score < -0.2:
            overall_sentiment = "NEGATIVE"
        else:
            overall_sentiment = "NEUTRAL"

        # Count sentiments
        sentiment_counts = {
            "POSITIVE": sum(1 for a in analyses if a.sentiment == "POSITIVE"),
            "NEGATIVE": sum(1 for a in analyses if a.sentiment == "NEGATIVE"),
            "NEUTRAL": sum(1 for a in analyses if a.sentiment == "NEUTRAL")
        }

        return {
            "symbol": symbol,
            "sentiment": overall_sentiment,
            "average_score": round(avg_score, 3),
            "analysis_count": len(analyses),
            "sentiment_breakdown": sentiment_counts,
            "high_impact_count": sum(1 for a in analyses if a.impact_level == "HIGH")
        }
```

**Success Criteria**:
- [ ] Analyzes articles using LLM
- [ ] Parses JSON responses correctly
- [ ] Stores analysis in database
- [ ] Handles errors gracefully
- [ ] Prevents duplicate analysis
- [ ] Can retrieve past analyses
- [ ] Aggregate sentiment calculation works

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
from services.llm_service import LLMService
from agents.news_agent import NewsAgent
from agents.analysis_agent import AnalysisAgent

setup_logging()
logger = get_logger("main")

async def run_pipeline():
    """Run the news + analysis pipeline."""

    # Initialize services
    news_service = NewsService(api_key=settings.news_api_key)
    llm_service = LLMService()

    # Initialize agents
    news_agent = NewsAgent(news_service)
    analysis_agent = AnalysisAgent(llm_service)

    # Get symbols
    symbols = symbol_config.get_enabled_symbols("stocks")
    logger.info("pipeline_started", symbols=symbols)

    for symbol in symbols:
        logger.info("processing_symbol", symbol=symbol)

        # Fetch news
        articles = await news_agent.execute(
            symbols=[symbol],
            hours_back=24,
            max_articles=20
        )

        if not articles:
            logger.info("no_articles_found", symbol=symbol)
            continue

        # Analyze articles
        analyses = await analysis_agent.execute(articles, symbol)

        # Get aggregate sentiment
        aggregate = await analysis_agent.get_aggregate_sentiment(symbol)

        logger.info(
            "symbol_analysis_complete",
            symbol=symbol,
            articles=len(articles),
            analyses=len(analyses),
            aggregate_sentiment=aggregate["sentiment"],
            average_score=aggregate["average_score"]
        )

    # Cleanup
    await news_service.close()
    await llm_service.close()

    logger.info("pipeline_completed")

def main():
    """Main entry point."""
    try:
        db.initialize_schema()
        asyncio.run(run_pipeline())
    except Exception as e:
        logger.error("main_failed", error=str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()
```

**Success Criteria**:
- [ ] Script runs end-to-end
- [ ] News fetched and analyzed
- [ ] Aggregate sentiment calculated
- [ ] Logs show clear progress

---

## Testing Checklist

### Unit Tests
**File**: `tests/test_llm_service.py`
- [ ] Test OpenAI completion
- [ ] Test Anthropic completion
- [ ] Test error handling
- [ ] Test rate limiting

**File**: `tests/test_analysis_agent.py`
- [ ] Test article analysis
- [ ] Test JSON parsing
- [ ] Test duplicate detection
- [ ] Test aggregate sentiment

### Integration Tests
- [ ] End-to-end analysis pipeline
- [ ] Database storage and retrieval
- [ ] Multiple article analysis

---

## Phase Completion Criteria

### Must Have
- [ ] LLM service integrated (OpenAI or Anthropic)
- [ ] Analysis agent analyzes articles
- [ ] Sentiment extracted correctly
- [ ] Analysis stored in database
- [ ] Can retrieve analyses
- [ ] Aggregate sentiment works
- [ ] Error handling robust
- [ ] Tests passing

### Nice to Have
- [ ] Multi-article synthesis
- [ ] Sentiment trend analysis
- [ ] Confidence scoring
- [ ] Analysis caching

---

## Verification Commands

```bash
# Test LLM service
python -c "from services.llm_service import LLMService; import asyncio; service = LLMService(); asyncio.run(service.generate_completion('You are helpful', 'Say hi'))"

# Run full pipeline
python main.py

# Check analyses in database
sqlite3 trading.db "SELECT symbol, sentiment, sentiment_score FROM analysis_results ORDER BY analyzed_at DESC LIMIT 10"

# Run tests
pytest tests/test_analysis* -v
```

---

## Next Phase
**Phase 4: Decision Agent** - Implement trading decision logic based on analysis

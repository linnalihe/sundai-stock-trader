import json
import uuid
from typing import List, Optional, Dict
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
        """
        Initialize Analysis Agent.

        Args:
            llm_service: LLM service for sentiment analysis
        """
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
        """
        Analyze a single article using LLM.

        Args:
            article: News article to analyze
            symbol: Stock symbol

        Returns:
            AnalysisResult or None if analysis fails
        """
        # Format prompt
        user_prompt = format_sentiment_prompt(
            symbol=symbol,
            title=article.title,
            source=article.source,
            published_at=article.published_at.isoformat(),
            content=article.content
        )

        # Get LLM analysis
        try:
            response = await self.llm_service.generate_completion(
                system_prompt=SENTIMENT_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
        except Exception as e:
            self.logger.error(
                "llm_analysis_failed",
                article_id=article.id,
                error=str(e)
            )
            return None

        # Parse JSON response
        try:
            # Clean response - sometimes LLM adds markdown code blocks
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            analysis_data = json.loads(cleaned_response)
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
                key_points=analysis_data.get("key_points", []),
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
                error=str(e),
                data=analysis_data
            )
            return None

    def _is_already_analyzed(self, article_id: str) -> bool:
        """
        Check if article has already been analyzed.

        Args:
            article_id: Article ID to check

        Returns:
            True if already analyzed, False otherwise
        """
        with db.get_session() as session:
            result = session.execute(
                text("SELECT id FROM analysis_results WHERE article_id = :article_id"),
                {"article_id": article_id}
            )
            return result.fetchone() is not None

    def _store_analysis(self, result: AnalysisResult):
        """
        Store analysis result in database.

        Args:
            result: AnalysisResult to store
        """
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
        """
        Retrieve recent analyses from database.

        Args:
            symbol: Stock symbol to filter by
            limit: Maximum number of analyses
            min_impact: Minimum impact level (LOW, MEDIUM, HIGH)

        Returns:
            List of AnalysisResult objects
        """
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
                    key_points=json.loads(row[5]) if row[5] else [],
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

        Args:
            symbol: Stock symbol
            hours_back: Time window (not used in current impl, uses recent analyses)

        Returns:
            Dict with aggregate metrics
        """
        analyses = self.get_recent_analyses(symbol, limit=50)

        if not analyses:
            return {
                "symbol": symbol,
                "sentiment": "NEUTRAL",
                "average_score": 0.0,
                "analysis_count": 0,
                "sentiment_breakdown": {
                    "POSITIVE": 0,
                    "NEGATIVE": 0,
                    "NEUTRAL": 0
                },
                "high_impact_count": 0
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

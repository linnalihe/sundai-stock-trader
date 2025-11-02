from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class SentimentType(str, Enum):
    """Sentiment classification."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

class ImpactLevel(str, Enum):
    """Expected market impact level."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class AnalysisResult(BaseModel):
    """Analysis result data model."""

    id: str = Field(..., description="Unique analysis identifier")
    article_id: str = Field(..., description="Related article ID")
    symbol: str = Field(..., description="Stock symbol")
    sentiment: SentimentType = Field(..., description="Overall sentiment")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score")
    key_points: list[str] = Field(default_factory=list, description="Key takeaways")
    impact_level: ImpactLevel = Field(..., description="Expected impact level")
    reasoning: str = Field(..., description="Analysis reasoning")
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

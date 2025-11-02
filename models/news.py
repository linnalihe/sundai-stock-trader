from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from typing import Optional

class NewsArticle(BaseModel):
    """News article data model."""

    id: str = Field(..., description="Unique article identifier")
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content/summary")
    source: str = Field(..., description="News source name")
    published_at: datetime = Field(..., description="Publication timestamp")
    url: HttpUrl = Field(..., description="Article URL")
    symbols: list[str] = Field(default_factory=list, description="Related stock symbols")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "news_123",
                "title": "Apple Reports Record Earnings",
                "content": "Apple Inc. reported record quarterly earnings...",
                "source": "Reuters",
                "published_at": "2025-11-02T10:00:00Z",
                "url": "https://example.com/article",
                "symbols": ["AAPL"],
                "relevance_score": 0.95
            }
        }

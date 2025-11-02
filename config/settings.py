from pydantic_settings import BaseSettings
from typing import Literal, Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    alpaca_api_key: str
    alpaca_api_secret: str
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    news_api_key: Optional[str] = None

    # Trading Configuration
    paper_trading: bool = True
    max_position_size: int = 100
    max_portfolio_percent: float = 0.20
    risk_percentage: float = 0.02
    max_daily_trades: int = 5
    max_daily_loss_percent: float = 0.02

    # Agent Configuration
    news_fetch_interval: int = 300  # seconds
    analysis_threshold: float = 0.6
    decision_confidence_threshold: float = 0.7

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4-turbo-preview"
    llm_temperature: float = 0.3

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Database
    database_url: str = "sqlite:///./trading.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

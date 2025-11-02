import pytest
from agents.prompts import (
    SENTIMENT_ANALYSIS_SYSTEM_PROMPT,
    SENTIMENT_ANALYSIS_USER_PROMPT_TEMPLATE,
    format_sentiment_prompt
)

def test_system_prompt_exists():
    """System prompt should be defined."""
    assert SENTIMENT_ANALYSIS_SYSTEM_PROMPT is not None
    assert len(SENTIMENT_ANALYSIS_SYSTEM_PROMPT) > 0

def test_system_prompt_contains_json():
    """System prompt should mention JSON format."""
    assert "JSON" in SENTIMENT_ANALYSIS_SYSTEM_PROMPT
    assert "sentiment" in SENTIMENT_ANALYSIS_SYSTEM_PROMPT.lower()

def test_system_prompt_has_required_fields():
    """System prompt should specify all required fields."""
    assert "sentiment" in SENTIMENT_ANALYSIS_SYSTEM_PROMPT.lower()
    assert "sentiment_score" in SENTIMENT_ANALYSIS_SYSTEM_PROMPT.lower()
    assert "impact_level" in SENTIMENT_ANALYSIS_SYSTEM_PROMPT.lower()
    assert "key_points" in SENTIMENT_ANALYSIS_SYSTEM_PROMPT.lower()
    assert "reasoning" in SENTIMENT_ANALYSIS_SYSTEM_PROMPT.lower()

def test_user_prompt_template_exists():
    """User prompt template should be defined."""
    assert SENTIMENT_ANALYSIS_USER_PROMPT_TEMPLATE is not None
    assert "{symbol}" in SENTIMENT_ANALYSIS_USER_PROMPT_TEMPLATE
    assert "{title}" in SENTIMENT_ANALYSIS_USER_PROMPT_TEMPLATE
    assert "{content}" in SENTIMENT_ANALYSIS_USER_PROMPT_TEMPLATE

def test_format_sentiment_prompt():
    """Should format prompts correctly."""
    prompt = format_sentiment_prompt(
        symbol="AAPL",
        title="Test Title",
        source="Reuters",
        published_at="2025-11-02",
        content="Test content about Apple stock"
    )

    assert "AAPL" in prompt
    assert "Test Title" in prompt
    assert "Reuters" in prompt
    assert "Test content" in prompt

def test_format_prompt_limits_content_length():
    """Should limit content to 2000 characters."""
    long_content = "x" * 5000

    prompt = format_sentiment_prompt(
        symbol="AAPL",
        title="Test",
        source="Test",
        published_at="2025-11-02",
        content=long_content
    )

    # Content should be truncated
    assert len(prompt) < 3000  # Total prompt should be reasonable

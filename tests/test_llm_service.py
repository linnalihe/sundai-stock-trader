import pytest
from services.llm_service import LLMService
from config.settings import settings

def test_llm_service_initialization():
    """Should initialize with correct provider."""
    # Skip if no API key configured
    if not settings.openai_api_key and not settings.anthropic_api_key:
        pytest.skip("No LLM API key configured")

    service = LLMService()
    assert service.provider in ["openai", "anthropic"]
    assert service.model is not None
    assert service.temperature == 0.3

@pytest.mark.asyncio
async def test_generate_completion():
    """Should generate completion."""
    # Skip if no API key configured
    if not settings.openai_api_key and not settings.anthropic_api_key:
        pytest.skip("No LLM API key configured")

    service = LLMService()

    try:
        response = await service.generate_completion(
            system_prompt="You are a helpful assistant. Respond only with valid JSON.",
            user_prompt='Return this exact JSON: {"test": "success"}'
        )

        assert response is not None
        assert len(response) > 0

    finally:
        await service.close()

@pytest.mark.asyncio
async def test_json_response_parsing():
    """Should return response that can be parsed as JSON."""
    # Skip if no API key configured
    if not settings.openai_api_key and not settings.anthropic_api_key:
        pytest.skip("No LLM API key configured")

    service = LLMService()

    try:
        response = await service.generate_completion(
            system_prompt="Respond only with valid JSON.",
            user_prompt='Return: {"status": "ok", "value": 42}'
        )

        # Should contain JSON-like content
        assert "{" in response
        assert "}" in response

    finally:
        await service.close()

def test_missing_api_key_raises_error():
    """Should raise error if API key not configured."""
    # Temporarily clear API keys
    original_openai = settings.openai_api_key
    original_anthropic = settings.anthropic_api_key

    settings.openai_api_key = None
    settings.anthropic_api_key = None

    with pytest.raises(ValueError, match="API key not configured"):
        LLMService()

    # Restore
    settings.openai_api_key = original_openai
    settings.anthropic_api_key = original_anthropic

def test_invalid_provider_raises_error():
    """Should raise error for invalid provider."""
    original_provider = settings.llm_provider

    settings.llm_provider = "invalid_provider"

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        LLMService()

    # Restore
    settings.llm_provider = original_provider

from typing import Optional
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("llm_service")

class LLMService:
    """Service for interacting with LLM APIs (OpenAI or Anthropic)."""

    def __init__(self):
        """Initialize LLM service based on settings."""
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature

        if self.provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        elif self.provider == "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        logger.info(
            "llm_service_initialized",
            provider=self.provider,
            model=self.model,
            temperature=self.temperature
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
        logger.info("llm_service_closed")

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

def format_sentiment_prompt(
    symbol: str,
    title: str,
    source: str,
    published_at: str,
    content: str
) -> str:
    """
    Format the sentiment analysis user prompt.

    Args:
        symbol: Stock symbol
        title: Article title
        source: News source
        published_at: Publication date
        content: Article content

    Returns:
        Formatted prompt string
    """
    return SENTIMENT_ANALYSIS_USER_PROMPT_TEMPLATE.format(
        symbol=symbol,
        title=title,
        source=source,
        published_at=published_at,
        content=content[:2000]  # Limit content to 2000 chars to control token usage
    )

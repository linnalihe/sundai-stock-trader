import pytest
from agents.decision_agent import DecisionAgent
from agents.analysis_agent import AnalysisAgent
from services.llm_service import LLMService
from services.alpaca_market_service import AlpacaMarketService
from agents.rules import TradingRules
from config.settings import settings


@pytest.fixture
def llm_service():
    """Create LLM service fixture."""
    if not settings.openai_api_key and not settings.anthropic_api_key:
        pytest.skip("No LLM API key configured")
    return LLMService()


@pytest.fixture
def market_service():
    """Create market service fixture."""
    return AlpacaMarketService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=True
    )


@pytest.fixture
def analysis_agent(llm_service):
    """Create analysis agent fixture."""
    return AnalysisAgent(llm_service)


@pytest.fixture
def decision_agent(analysis_agent, market_service):
    """Create decision agent fixture."""
    return DecisionAgent(analysis_agent, market_service)


def test_decision_agent_initialization(decision_agent):
    """Decision agent should initialize correctly."""
    assert decision_agent.name == "decision_agent"
    assert decision_agent.analysis_agent is not None
    assert decision_agent.market_service is not None
    assert decision_agent.rules is not None


def test_custom_rules(market_service):
    """Should accept custom trading rules."""
    if not settings.openai_api_key and not settings.anthropic_api_key:
        pytest.skip("No LLM API key configured")

    llm_service = LLMService()
    analysis_agent = AnalysisAgent(llm_service)

    custom_rules = TradingRules(
        buy_sentiment_threshold=0.5,
        max_position_size=50
    )

    agent = DecisionAgent(analysis_agent, market_service, rules=custom_rules)

    assert agent.rules.buy_sentiment_threshold == 0.5
    assert agent.rules.max_position_size == 50


@pytest.mark.asyncio
async def test_execute_with_insufficient_analyses(decision_agent, llm_service):
    """Should return None if insufficient analyses."""
    try:
        # Use a symbol with no analyses
        decision = await decision_agent.execute("NONEXISTENT123")

        # Should return None due to insufficient analyses
        assert decision is None
    finally:
        await llm_service.close()


@pytest.mark.asyncio
async def test_execute_with_real_data(decision_agent, llm_service):
    """Should make decision based on real sentiment data."""
    try:
        # Use AAPL which has sentiment data from Phase 3
        decision = await decision_agent.execute("AAPL", hours_back=72)

        if decision:
            assert decision.symbol == "AAPL"
            assert decision.action in ["BUY", "SELL", "HOLD"]
            assert decision.quantity >= 0
            assert decision.expected_price > 0
            assert decision.confidence in ["HIGH", "MEDIUM", "LOW"]
            assert decision.reasoning is not None
            assert len(decision.reasoning) > 0
            assert decision.sentiment_score is not None
            assert decision.analysis_count >= 0
    finally:
        await llm_service.close()


def test_position_size_calculation(decision_agent):
    """Should calculate position size correctly."""
    # Test with high confidence signal
    size = decision_agent._calculate_position_size(
        score=0.8,
        high_impact_count=3,
        analysis_count=5
    )

    assert size > decision_agent.rules.base_position_size
    assert size <= decision_agent.rules.max_position_size

    # Test with weak signal
    size_weak = decision_agent._calculate_position_size(
        score=0.35,
        high_impact_count=0,
        analysis_count=3
    )

    assert size_weak >= decision_agent.rules.base_position_size
    assert size_weak < size  # Weaker signal = smaller position


def test_position_size_respects_max(decision_agent):
    """Position size should never exceed max."""
    # Even with very strong signal
    size = decision_agent._calculate_position_size(
        score=1.0,
        high_impact_count=10,
        analysis_count=20
    )

    assert size <= decision_agent.rules.max_position_size


def test_confidence_calculation_high(decision_agent):
    """Should calculate HIGH confidence correctly."""
    conf = decision_agent._calculate_confidence(
        score=0.8,
        high_impact_count=3,
        analysis_count=5
    )
    assert conf == "HIGH"


def test_confidence_calculation_medium(decision_agent):
    """Should calculate MEDIUM confidence correctly."""
    # Medium score with some impact
    conf = decision_agent._calculate_confidence(
        score=0.4,
        high_impact_count=1,
        analysis_count=5
    )
    assert conf == "MEDIUM"

    # Lower score but has high impact
    conf2 = decision_agent._calculate_confidence(
        score=0.2,
        high_impact_count=1,
        analysis_count=5
    )
    assert conf2 == "MEDIUM"


def test_confidence_calculation_low(decision_agent):
    """Should calculate LOW confidence correctly."""
    conf = decision_agent._calculate_confidence(
        score=0.2,
        high_impact_count=0,
        analysis_count=5
    )
    assert conf == "LOW"


def test_get_recent_decisions(decision_agent):
    """Should retrieve recent decisions."""
    decisions = decision_agent.get_recent_decisions("AAPL", limit=5)

    assert isinstance(decisions, list)
    assert len(decisions) <= 5

    # If we have decisions, check their structure
    for decision in decisions:
        assert hasattr(decision, 'symbol')
        assert hasattr(decision, 'action')
        assert hasattr(decision, 'quantity')


@pytest.mark.asyncio
async def test_decision_reasoning_includes_key_info(decision_agent, llm_service):
    """Should include clear reasoning in decision."""
    try:
        decision = await decision_agent.execute("AAPL", hours_back=72)

        if decision:
            assert decision.reasoning is not None
            assert len(decision.reasoning) > 0

            # Reasoning should mention key concepts
            reasoning_lower = decision.reasoning.lower()
            has_relevant_terms = any(word in reasoning_lower
                                    for word in ["sentiment", "position", "shares",
                                                 "holding", "buying", "selling",
                                                 "positive", "negative", "neutral"])
            assert has_relevant_terms
    finally:
        await llm_service.close()


def test_position_size_scales_with_sentiment(decision_agent):
    """Position size should scale with sentiment strength."""
    # Weak positive
    size_weak = decision_agent._calculate_position_size(
        score=0.35,
        high_impact_count=0,
        analysis_count=5
    )

    # Strong positive
    size_strong = decision_agent._calculate_position_size(
        score=0.8,
        high_impact_count=2,
        analysis_count=5
    )

    # Strong should be larger
    assert size_strong > size_weak


def test_high_impact_multiplier_applied(decision_agent):
    """High impact articles should increase position size."""
    # Same score, different high impact counts
    size_no_impact = decision_agent._calculate_position_size(
        score=0.6,
        high_impact_count=1,
        analysis_count=5
    )

    size_with_impact = decision_agent._calculate_position_size(
        score=0.6,
        high_impact_count=3,  # Above threshold
        analysis_count=5
    )

    # With more high impact should be larger
    assert size_with_impact > size_no_impact


def test_decision_agent_has_correct_attributes(decision_agent):
    """Decision agent should have all required attributes."""
    assert hasattr(decision_agent, 'analysis_agent')
    assert hasattr(decision_agent, 'market_service')
    assert hasattr(decision_agent, 'rules')
    assert hasattr(decision_agent, 'execute')
    assert hasattr(decision_agent, 'get_recent_decisions')


@pytest.mark.asyncio
async def test_decision_stored_in_database(decision_agent, llm_service):
    """Decision should be stored in database after execution."""
    try:
        # Make a decision
        decision = await decision_agent.execute("AAPL", hours_back=72)

        if decision:
            # Try to retrieve it
            recent = decision_agent.get_recent_decisions("AAPL", limit=1)

            assert len(recent) >= 1
            # Most recent should match our decision
            latest = recent[0]
            assert latest.symbol == decision.symbol
            assert latest.action == decision.action
    finally:
        await llm_service.close()

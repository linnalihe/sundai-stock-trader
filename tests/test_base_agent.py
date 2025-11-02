import pytest
from agents.base import BaseAgent

def test_base_agent_is_abstract():
    """Cannot instantiate BaseAgent directly."""
    with pytest.raises(TypeError):
        agent = BaseAgent("test")

def test_custom_agent_requires_execute():
    """Custom agent must implement execute()."""
    class BadAgent(BaseAgent):
        pass

    with pytest.raises(TypeError):
        agent = BadAgent("bad")

@pytest.mark.asyncio
async def test_agent_with_execute():
    """Agent with execute() method can be instantiated."""
    class TestAgent(BaseAgent):
        async def execute(self):
            return "done"

    agent = TestAgent("test")
    assert agent.name == "test"
    result = await agent.execute()
    assert result == "done"

@pytest.mark.asyncio
async def test_agent_callable():
    """Agent should be callable via __call__."""
    class TestAgent(BaseAgent):
        async def execute(self):
            return "completed"

    agent = TestAgent("test")
    result = await agent()
    assert result == "completed"

@pytest.mark.asyncio
async def test_agent_error_handling():
    """Agent should handle errors properly."""
    class ErrorAgent(BaseAgent):
        async def execute(self):
            raise ValueError("Test error")

    agent = ErrorAgent("error_test")
    with pytest.raises(ValueError, match="Test error"):
        await agent()

@pytest.mark.asyncio
async def test_agent_log_event():
    """Agent should have _log_event helper."""
    class TestAgent(BaseAgent):
        async def execute(self):
            self._log_event("test_event", key="value")
            return "ok"

    agent = TestAgent("test")
    result = await agent()
    assert result == "ok"

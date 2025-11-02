from abc import ABC, abstractmethod
from utils.logger import get_logger
from typing import Any

class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        """
        Initialize base agent.

        Args:
            name: Agent name for logging
        """
        self.name = name
        self.logger = get_logger(f"agent.{name}")
        self.logger.info("agent_initialized", agent=name)

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """
        Execute the agent's main task.

        Must be implemented by subclasses.
        """
        pass

    async def __call__(self, *args, **kwargs) -> Any:
        """Allow agent to be called directly."""
        self.logger.info("agent_execution_started", agent=self.name)
        try:
            result = await self.execute(*args, **kwargs)
            self.logger.info("agent_execution_completed", agent=self.name)
            return result
        except Exception as e:
            self.logger.error(
                "agent_execution_failed",
                agent=self.name,
                error=str(e),
                exc_info=True
            )
            raise

    def _log_event(self, event: str, **kwargs):
        """Helper method to log agent events."""
        self.logger.info(event, agent=self.name, **kwargs)

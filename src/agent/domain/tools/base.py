from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def _execute_impl(self, context: Any, **kwargs) -> Dict[str, Any]:
        """Execute the tool's main logic."""
        pass

    async def execute(self, context: Any, **kwargs) -> Dict[str, Any]:
        """Public method to execute the tool, wrapping the abstract implementation."""
        return await self._execute_impl(context, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, description={self.description!r})"

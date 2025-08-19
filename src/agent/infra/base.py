# src/stock_assistant/infra/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any


class HealthCheckable(ABC):
    """Abstract interface for health checkable components in infrastructure layer."""
    
    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Check the health of the component and return status and details."""
        pass